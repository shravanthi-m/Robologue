"""Person 1: native RGB frame evidence, without reading reference annotations.

Register an explicitly selected RGB directory. Numeric JPEG stems are frame IDs,
not seconds. The runtime must supply observed_until_frame from its trusted cursor.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from .perception import inspect as vision
from .perception.video import json_key, run_media, write_json

ADAPTER_VERSION = "industreal-rgb-v1"
DEFAULT_MANIFEST = "work/industreal/recordings.json"
DEFAULT_CACHE = "work/industreal/cache"


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def load_manifest(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "adapter": ADAPTER_VERSION, "recordings": {}}
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1 or value.get("adapter") != ADAPTER_VERSION
            or not isinstance(value.get("recordings"), dict)):
        raise ValueError("Not an IndustReal RGB manifest; use a separate manifest from video registration")
    return value


def register_frames(manifest_path, recording_id, rgb_dir, *, replace=False, source="industreal"):
    """Index file names/stats only; hash and decode selected images at sampling time.

    Explicitly register RGB or rgb, never the whole recording/annotation directory.
    This snapshots a partial download. Re-register with replace=True as files arrive.
    """
    _text(recording_id, "recording_id")
    if source not in ("industreal", "synthetic"):
        raise ValueError("source must be industreal or synthetic")
    directory = Path(rgb_dir).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("rgb_dir must be a directory of JPEG images")
    manifest = load_manifest(manifest_path)
    if recording_id in manifest["recordings"] and not replace:
        raise ValueError("Recording already registered; explicitly use --replace to refresh the snapshot")
    frames = []
    seen = set()
    for path in directory.iterdir():
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            continue  # No annotation, CSV, sensor, or nested directory is opened.
        if path.is_symlink() or not path.is_file() or not re.fullmatch(r"[0-9]+", path.stem):
            raise ValueError("RGB frames must be regular JPEG files with numeric stems (no symlinks)")
        frame_id = int(path.stem)
        if frame_id in seen:
            raise ValueError(f"Duplicate numeric frame ID: {frame_id}")
        seen.add(frame_id)
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError("Empty JPEG found; wait for download completion before registering")
        frames.append({"frame_id": frame_id, "filename": path.name,
                       "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not frames:
        raise ValueError("No numeric JPEG frames found; select the RGB image folder, not its parent")
    frames.sort(key=lambda f: f["frame_id"])
    record = {"recording_id": recording_id, "rgb_dir": str(directory), "source": source,
              "frame_count": len(frames), "first_frame": frames[0]["frame_id"],
              "last_frame": frames[-1]["frame_id"], "frames": frames,
              "index_hash": json_key(frames), "annotation_alignment": "unverified",
              "time_basis": "numeric_source_filename; seconds_not_inferred",
              "motion_status": "unavailable"}
    manifest["recordings"][recording_id] = record
    write_json(manifest_path, manifest)
    return {k: v for k, v in record.items() if k != "frames"}


def sample_frames(manifest_path, recording_id, start_frame, end_frame, observed_until_frame,
                  *, n_frames=3, cache_dir=DEFAULT_CACHE):
    """Select <=3 existing source IDs; never invent frames in numbering gaps."""
    for name, value in (("start_frame", start_frame), ("end_frame", end_frame),
                        ("observed_until_frame", observed_until_frame), ("n_frames", n_frames)):
        _integer(value, name)
    if not 1 <= n_frames <= 3:
        raise ValueError("n_frames must be between 1 and 3")
    if not start_frame <= end_frame <= observed_until_frame:
        raise ValueError("Require start_frame <= end_frame <= trusted observed_until_frame")
    records = load_manifest(manifest_path)["recordings"]
    if recording_id not in records:
        raise ValueError("Unknown recording; register its RGB directory first")
    record = records[recording_id]
    if start_frame < record["first_frame"] or observed_until_frame > record["last_frame"]:
        raise ValueError("Requested range/cursor exceeds the registered frame snapshot")
    available = [f for f in record["frames"] if start_frame <= f["frame_id"] <= end_frame]
    if not available:
        raise ValueError("No available frames in the requested interval")
    count = min(n_frames, len(available))
    indices = ([len(available) - 1] if count == 1 else
               [i * (len(available) - 1) // (count - 1) for i in range(count)])
    directory = Path(record["rgb_dir"])
    output = []
    for index in indices:
        frame = available[index]
        filename = frame["filename"]
        if Path(filename).name != filename:
            raise ValueError("Invalid source filename in manifest")
        path = directory / filename
        if path.is_symlink() or path.resolve().parent != directory:
            raise ValueError("Source frame must remain inside its registered RGB directory")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (frame["size_bytes"], frame["mtime_ns"]):
            raise ValueError("Selected frame changed; re-register this recording explicitly")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        cached = Path(cache_dir).resolve() / "frames" / f"{digest}.jpg"
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists() or cached.read_bytes() != content:
            cached.write_bytes(content)
        # Decode selected images; a renamed/truncated file must never reach the VLM.
        run_media(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(cached),
                   "-frames:v", "1", "-f", "null", "-"], timeout=30)
        output.append({"frame_id": frame["frame_id"], "source_filename": filename,
                       "source_path": str(path), "path": str(cached), "sha256": digest})
    return {"recording_id": recording_id, "source": record["source"],
            "window_frames": [start_frame, end_frame], "cursor_frame": observed_until_frame,
            "frames": output, "index_hash": record["index_hash"],
            "uncertainties": ["Annotation alignment has not been verified.",
                              "Sparse frames do not establish continuous movement."]}


def get_evidence(manifest_path, recording_id, checkpoint_id, component_id, start_frame, end_frame,
                 observed_until_frame, *, model=None, n_frames=3, cache_dir=DEFAULT_CACHE,
                 mock=False, client=None):
    """Return Interface A. Mock mode is explicit and never describes an assembly.

    Person 2 reserves the logical call budget and supplies a trusted cursor. This
    function never reads reference labels or produces a correctness verdict.
    """
    _text(checkpoint_id, "checkpoint_id")
    _text(component_id, "component_id")
    model = "mock" if mock else model or os.environ.get("OPENROUTER_MODEL")
    _text(model, "model")
    sample = sample_frames(manifest_path, recording_id, start_frame, end_frame,
                           observed_until_frame, n_frames=n_frames, cache_dir=cache_dir)
    if sample["source"] == "synthetic" and not mock:
        raise ValueError("Synthetic fixtures require --mock; no paid model request is made")
    identity = {"adapter": ADAPTER_VERSION, "model": model, "mock": mock,
                "client_version": "bounded-json-v1" if client is not None else "original-v1",
                "endpoint": vision.ENDPOINT, "prompt_hash": json_key(vision.PROMPT),
                "prompt_version": vision.PROMPT_VERSION, "max_tokens": 1000,
                "temperature": 0, "frames": [{"frame_id": f["frame_id"], "sha256": f["sha256"]}
                                             for f in sample["frames"]]}
    observation_id = json_key(identity)
    path = Path(cache_dir) / "observations" / f"{observation_id}.json"
    hit = path.exists()
    if hit:
        cached = json.loads(path.read_text())
        if cached.get("identity") != identity:
            raise ValueError("Observation cache identity mismatch; remove the corrupt entry")
        value = vision.validate_observations(cached["value"])
        call = cached["call"]
    else:
        if mock:
            value = {"observations": ["MOCK: observation provider was not called."],
                     "uncertainties": ["No assembly conclusion can be drawn from this mock packet."]}
            call = {"provider": "mock", "usage": {}, "latency_seconds": 0}
        else:
            # Send only content-addressed images and numeric IDs, never local names,
            # component instructions, recording names, annotations, or expected states.
            frames = [{"path": f["path"], "frame_id": f["frame_id"]} for f in sample["frames"]]
            value, call = vision.openrouter_inspect(frames, model, request_timeout=30, client=client)
            vision.validate_observations(value)
        write_json(path, {"identity": identity, "value": value, "call": call})
    packet_identity = {"recording_id": recording_id, "checkpoint_id": checkpoint_id,
                       "component_id": component_id, "cursor_frame": observed_until_frame,
                       "window_frames": sample["window_frames"], "observation_id": observation_id}
    return {"schema_version": 1, "source_kind": "mock" if mock else "rgb_vlm",
            "evidence_id": json_key(packet_identity), **packet_identity,
            "frame_ids": [f["frame_id"] for f in sample["frames"]],
            "observations": value["observations"],
            "uncertainties": value["uncertainties"] + sample["uncertainties"],
            "provider": {"model_id": model, "prompt_version": vision.PROMPT_VERSION,
                         "adapter_version": ADAPTER_VERSION, "call": call},
            "frames": sample["frames"], "cache_hit": hit,
            "new_api_calls": 0 if mock or hit else 1, "logical_tool_calls": 1,
            "annotation_alignment": "unverified",
            "motion_summary": {"status": "unavailable", "reason": "Not part of the RGB MVP."}}


def create_fixture(output_dir):
    """Create small colored JPEG fixtures, registration and a MOCK packet offline."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rgb = root / "RGB"
    rgb.mkdir()
    for frame_id, color in ((10, "red"), (20, "green"), (40, "blue")):
        run_media(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                   f"color=c={color}:s=160x120", "-frames:v", "1", "-update", "1",
                   str(rgb / f"{frame_id:06}.jpg")], timeout=30)
    manifest = root / "recordings.json"
    register_frames(manifest, "synthetic-demo", rgb, source="synthetic")
    result = get_evidence(manifest, "synthetic-demo", "checkpoint-1", "demo-component",
                          10, 40, 40, mock=True, cache_dir=root / "cache")
    write_json(root / "evidence.json", result)
    return {"manifest": str(manifest), "evidence": str(root / "evidence.json"), "source_kind": "mock"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="Offline synthetic smoke test; no API key or download")
    fixture.add_argument("output_dir")
    reg = sub.add_parser("register", help="Index an explicit RGB JPEG directory")
    reg.add_argument("rgb_dir")
    reg.add_argument("--recording", required=True)
    reg.add_argument("--manifest", default=DEFAULT_MANIFEST)
    reg.add_argument("--replace", action="store_true")
    for name in ("sample", "inspect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--manifest", default=DEFAULT_MANIFEST)
        cmd.add_argument("--recording", required=True)
        cmd.add_argument("--start-frame", type=int, required=True)
        cmd.add_argument("--end-frame", type=int, required=True)
        cmd.add_argument("--observed-until-frame", type=int, required=True)
        cmd.add_argument("--frames", type=int, default=3)
        cmd.add_argument("--cache-dir", default=DEFAULT_CACHE)
        cmd.add_argument("--output", help="New JSON output file; refuses overwrite")
        if name == "inspect":
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--component", required=True)
            cmd.add_argument("--model")
            cmd.add_argument("--mock", action="store_true")
            cmd.add_argument("--env-file", help="Explicit ignored credentials file (Person 2 loader)")
    args = parser.parse_args()
    try:
        if getattr(args, "env_file", None):
            from .storage_common import load_env_file
            load_env_file(args.env_file)
        if getattr(args, "output", None) and Path(args.output).exists():
            raise ValueError("Output already exists; choose a new filename")
        if args.command == "fixture":
            result = create_fixture(args.output_dir)
        elif args.command == "register":
            result = register_frames(args.manifest, args.recording, args.rgb_dir, replace=args.replace)
        else:
            positional = (args.manifest, args.recording)
            window = (args.start_frame, args.end_frame, args.observed_until_frame)
            options = {"n_frames": args.frames, "cache_dir": args.cache_dir}
            if args.command == "sample":
                result = sample_frames(*positional, *window, **options)
            else:
                result = get_evidence(*positional, args.checkpoint, args.component, *window,
                                      model=args.model, mock=args.mock, **options)
        if getattr(args, "output", None):
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Evidence adapter: {exc}\n")


if __name__ == "__main__":
    main()
