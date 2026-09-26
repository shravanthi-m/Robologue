"""Full MP4 decode inventory and sampled RGB-to-video alignment checks."""

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .perception.video import frame_index, register_video, write_json


def _decode(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,duration,avg_frame_rate,nb_read_frames,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    stream = json.loads(result.stdout).get("streams", [{}])[0] if result.stdout else {}
    return {
        "recording_id": path.stem,
        "bytes": path.stat().st_size,
        "stream": stream,
        "decode_ok": result.returncode == 0
        and not result.stderr.strip()
        and int(stream.get("nb_read_frames", 0)) > 0,
        "decode_error": bool(result.stderr.strip()),
    }


def audit_media(root, output, *, workers=2):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    videos = sorted((root / "videos").glob("*.mp4"))
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_decode, path): path for path in videos}
        for job in as_completed(jobs):
            records.append(job.result())
            if len(records) % 10 == 0:
                print(
                    f"[media] fully decoded {len(records)}/{len(videos)} videos",
                    flush=True,
                )
    records.sort(key=lambda r: r["recording_id"])
    report = {
        "videos": records,
        "total": len(records),
        "decode_ok": sum(r["decode_ok"] for r in records),
    }
    write_json(output / "video_audit.json", report)
    alignment = []
    from PIL import Image
    import numpy as np

    for native in sorted((root / "native").glob("*")):
        if not native.is_dir():
            continue
        rec = native.name
        record = register_video(
            output / "video_manifest.json",
            rec,
            root / "videos" / f"{rec}.mp4",
            source="IndustReal",
            replace=True,
        )
        index = frame_index(record, output / "cache")
        ids = sorted(int(p.stem) for p in (native / "rgb").glob("*.jpg"))
        if len(index["timestamps"]) != len(ids) or ids != list(range(len(ids))):
            raise ValueError(
                f"Video/native frame count mismatch for {rec}; no time map fabricated"
            )
        selected = sorted(
            {0, len(ids) // 4, len(ids) // 2, 3 * len(ids) // 4, len(ids) - 1}
        )
        with tempfile.TemporaryDirectory() as temp:
            selection = "+".join(f"eq(n\\,{i})" for i in selected)
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    record["video_path"],
                    "-vf",
                    f"select={selection}",
                    "-fps_mode",
                    "vfr",
                    "-frames:v",
                    str(len(selected)),
                    str(Path(temp) / "%03d.png"),
                ],
                capture_output=True,
                timeout=180,
            )
            if result.returncode:
                raise ValueError("Alignment frame decode failed")
            errors = []
            for ident, path in zip(selected, sorted(Path(temp).glob("*.png"))):
                with Image.open(native / "rgb" / f"{ident:06}.jpg") as image:
                    a = np.asarray(image.convert("RGB").resize((128, 72)), dtype=float)
                with Image.open(path) as image:
                    b = np.asarray(image.convert("RGB").resize((128, 72)), dtype=float)
                errors.append(float(np.abs(a - b).mean()))
        verified = len(errors) == len(selected) and max(errors) < 12
        alignment.append(
            {
                "recording_id": rec,
                "native_frames": len(ids),
                "video_frames": len(index["timestamps"]),
                "sample_frames": selected,
                "mean_absolute_pixel_errors": errors,
                "verified_sample_alignment": verified,
                "time_basis": "decoded MP4 PTS; rendered timeline, not sensor capture clock",
            }
        )
        if not verified:
            raise ValueError(f"Sampled RGB/video alignment failed for {rec}")
        mapping = {
            "schema_version": 1,
            "verified": True,
            "recording_id": rec,
            "mapping_id": "decoded-pts:" + record["sha256"],
            "frame_seconds": {str(i): t for i, t in enumerate(index["timestamps"])},
            "verification": "Equal decoded/native counts plus five pixel-alignment spot checks.",
            "time_basis": "rendered MP4 timeline; no physical capture-time claim",
        }
        write_json(output / "time_maps" / f"{rec}.json", mapping)
        print(
            f"[media] {rec}: {len(ids)} RGB frames aligned to decoded video timeline",
            flush=True,
        )
    report["native_alignment"] = alignment
    write_json(output / "video_audit.json", report)
    return report
