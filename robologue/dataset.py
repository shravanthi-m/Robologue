"""Archive preparation and an evaluator-only inventory for IndustReal."""

import csv
import io
import json
import re
import shutil
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from .perception.video import write_json
from .industreal_labels import load_raw_states, load_procedure_info


def extract_selected(archive, root, select):
    """Only regular safe members are extracted; ZipFile reads validate CRC."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    count = total = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir() or not select(info.filename):
                continue
            path = root / info.filename
            if path.resolve().is_relative_to(root) is False or "\\" in info.filename:
                raise ValueError("Unsafe archive member")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive links are not supported")
            path.parent.mkdir(parents=True, exist_ok=True)
            # A completed extraction is reusable; partial files are replaced.
            reusable = path.exists() and path.stat().st_size == info.file_size
            if reusable:
                crc = 0
                with path.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                reusable = crc == info.CRC
            if not reusable:
                temporary = path.with_suffix(path.suffix + ".partial")
                with source.open(info) as stream, temporary.open("wb") as target:
                    shutil.copyfileobj(stream, target, 1024 * 1024)
                temporary.replace(path)
            count += 1
            total += info.file_size
    return {"members": count, "bytes": total}


def prepare(labels_zip, native_zip, videos_zip, output, procedure_info):
    root = Path(output).resolve()
    extracted = {
        "action_labels": extract_selected(
            labels_zip,
            root / "action_labels",
            lambda n: n in ("train.csv", "val.csv", "test.csv"),
        ),
        "native_rgb_and_annotations": extract_selected(
            native_zip,
            root / "native",
            lambda n: "/rgb/" in n
            or n.endswith(("/PSR_labels_raw.csv", "/AR_labels.csv")),
        ),
        "videos": extract_selected(
            videos_zip, root / "videos", lambda n: n.endswith(".mp4")
        ),
    }
    components = load_procedure_info(procedure_info)
    catalog = {
        "schema_version": 1,
        "source": "IndustReal/PSR/procedure_info.json",
        "components": {f"component-{i}": d for i, d in sorted(components.items())},
    }
    write_json(root / "components.json", catalog)
    write_json(root / "extraction.json", extracted)
    report = inventory(root)
    write_json(root / "inventory.json", report)
    return report


def action_rows(path):
    rows = []
    with Path(path).open(newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) != 5:
                raise ValueError(f"Invalid action row {line}")
            rec, ident, label, start, end = row
            if not re.fullmatch(r"[0-9]{2}_(assy|main)_[0-9]+_[0-9]+", rec):
                raise ValueError("Unknown recording ID format")
            try:
                ident, start, end = (
                    int(ident),
                    int(Path(start).stem),
                    int(Path(end).stem),
                )
            except ValueError:
                raise ValueError("Invalid action coordinates") from None
            if ident < 0 or not label or not 0 <= start <= end:
                raise ValueError("Invalid action interval")
            rows.append(
                {
                    "recording_id": rec,
                    "class_id": ident,
                    "action": label,
                    "start_frame": start,
                    "end_frame": end,
                }
            )
    return rows


def inventory(root):
    root = Path(root)
    videos = {p.stem for p in (root / "videos").glob("*.mp4")}
    splits = {}
    all_actions = []
    rows_by_split = {}
    for split in ("train", "val", "test"):
        rows = action_rows(root / "action_labels" / f"{split}.csv")
        rows_by_split[split] = rows
        recordings = sorted({r["recording_id"] for r in rows})
        splits[split] = {
            "rows": len(rows),
            "recordings": recordings,
            "classes": len({r["class_id"] for r in rows}),
        }
        all_actions.extend(rows)
    sets = {k: set(v["recordings"]) for k, v in splits.items()}
    overlap = {
        a + "_" + b: sorted(sets[a] & sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    names = {}
    for r in all_actions:
        names.setdefault(r["class_id"], set()).add(r["action"])
    native = []
    for path in sorted((root / "native").glob("*")):
        if not path.is_dir():
            continue
        ids = sorted(int(p.stem) for p in (path / "rgb").glob("*.jpg"))
        rows = load_raw_states(path / "PSR_labels_raw.csv")
        points = [f for f, _, _ in rows]
        ar = action_rows(path / "AR_labels.csv")
        global_test = [
            r for r in rows_by_split["test"] if r["recording_id"] == path.name
        ]
        native.append(
            {
                "recording_id": path.name,
                "rgb_frames": len(ids),
                "first_frame": ids[0],
                "last_frame": ids[-1],
                "missing_source_frames": len(
                    set(range(ids[0], ids[-1] + 1)) - set(ids)
                ),
                "checkpoint_frames": points,
                "components": len(rows[0][2]),
                "reference_cases": sum(len(s) for _, _, s in rows),
                "reference_states": dict(
                    Counter(s for _, _, states in rows for s in states)
                ),
                "checkpoints_missing_rgb": sorted(set(points) - set(ids)),
                "action_rows": len(ar),
                "action_matches_global_test": ar == global_test,
                "has_video": path.name in videos,
            }
        )
    return {
        "schema_version": 1,
        "videos": len(videos),
        "video_recording_ids": sorted(videos),
        "action_splits": splits,
        "split_overlaps": overlap,
        "class_mapping_conflicts": {
            str(k): sorted(v) for k, v in names.items() if len(v) > 1
        },
        "extra_videos_without_action_labels": sorted(
            videos - set().union(*sets.values())
        ),
        "action_recordings_without_video": sorted(set().union(*sets.values()) - videos),
        "native_recordings": native,
        "total_native_rgb_frames": sum(r["rgb_frames"] for r in native),
        "total_state_reference_cases": sum(r["reference_cases"] for r in native),
        "warning": "Native archives are from official test. Any internal adaptation split is an exploratory demonstration, not the official untouched test benchmark.",
    }
