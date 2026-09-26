"""Local video registry and exact decoded-frame selection using FFmpeg tools."""
import bisect
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

SAMPLING_VERSION = "decoded-frame-v1"


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
    temporary.replace(path)


def run_media(command, timeout=180):
    if not shutil.which(command[0]):
        raise ValueError(f"{command[0]} is required. Install FFmpeg (which includes ffprobe).")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{command[0]} timed out; use a shorter recording or inspect the media locally.") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{command[0]} failed; confirm that the media file is readable and contains video.") from exc
    return result.stdout


def load_manifest(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "recordings": {}}
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("recordings"), dict):
        raise ValueError("Unsupported recording manifest")
    return manifest


def register_video(manifest_path, recording_id, video_path, *, source="local", replace=False):
    """Register a local video. Times are relative to the first decoded video frame."""
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("recording_id must be nonempty")
    video = Path(video_path).expanduser().resolve(strict=True)
    metadata = json.loads(run_media([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,duration,avg_frame_rate:format=duration", "-of", "json", str(video)
    ]))
    if not metadata.get("streams"):
        raise ValueError("No video stream found")
    stream = metadata["streams"][0]
    duration = float(stream.get("duration", metadata.get("format", {}).get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video has no usable finite duration")
    manifest = load_manifest(manifest_path)
    if recording_id in manifest["recordings"] and not replace:
        raise ValueError("Recording ID already registered. Use a new ID or explicitly pass --replace.")
    stat = video.stat()
    recording = {"recording_id": recording_id, "video_path": str(video),
                 "sha256": digest_file(video), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "duration_seconds": duration, "width": stream["width"], "height": stream["height"],
                 "avg_frame_rate": stream.get("avg_frame_rate"), "source": source,
                 "time_origin": "first_decoded_video_frame", "annotation_alignment": "unverified",
                 "sensor_status": {"spatial": "not_validated", "depth": "not_validated"}}
    manifest["recordings"][recording_id] = recording
    write_json(manifest_path, manifest)
    return recording


def get_recording(manifest_path, recording_id):
    manifest = load_manifest(manifest_path)
    if recording_id not in manifest["recordings"]:
        raise ValueError(f"Recording is not registered: {recording_id}")
    recording = manifest["recordings"][recording_id]
    stat = Path(recording["video_path"]).stat()
    if stat.st_size != recording["size_bytes"] or stat.st_mtime_ns != recording["mtime_ns"]:
        raise ValueError("Video changed since registration. Re-register explicitly before inspecting it.")
    return recording


def frame_index(recording, cache_dir):
    """Store actual presentation timestamps, not timestamps inferred from nominal FPS."""
    path = Path(cache_dir) / "indexes" / (recording["sha256"] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    payload = json.loads(run_media([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", recording["video_path"]
    ]))
    frames = payload.get("frames", [])
    if not frames or any("best_effort_timestamp_time" not in frame for frame in frames):
        raise ValueError("Every decoded frame must have a presentation timestamp")
    raw = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    times = [value - raw[0] for value in raw]
    if any(not math.isfinite(t) for t in times) or any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("Video timestamps are not finite and monotonic")
    index = {"first_pts_seconds": raw[0], "timestamps": times}
    write_json(path, index)
    return index


def validate_window(start, end, observed_until, duration, n_frames):
    for value in (start, end, observed_until):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Window times must be finite numbers")
    if not 0 <= start < end <= observed_until <= duration:
        raise ValueError("Require 0 <= start < end <= observed_until <= video duration")
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or not 1 <= n_frames <= 8:
        raise ValueError("n_frames must be an integer from 1 to 8")


def sample_video(manifest_path, recording_id, t_start, t_end, n_frames, observed_until,
                 *, cache_dir="work/perception"):
    recording = get_recording(manifest_path, recording_id)
    validate_window(t_start, t_end, observed_until, recording["duration_seconds"], n_frames)
    times = frame_index(recording, cache_dir)["timestamps"]
    left, right = bisect.bisect_left(times, t_start), bisect.bisect_right(times, t_end)
    if left == right:
        raise ValueError("No decoded frames exist inside the permitted interval")
    # Evenly spaced target times; choose the nearest in-window decoded frames.
    targets = [(t_start + t_end) / 2] if n_frames == 1 else [
        t_start + (t_end - t_start) * i / (n_frames - 1) for i in range(n_frames)]
    chosen = set()
    for target in targets:
        position = bisect.bisect_left(times, target, left, right)
        candidates = [i for i in (position - 1, position) if left <= i < right]
        chosen.add(min(candidates, key=lambda i: abs(times[i] - target)))
    indices = sorted(chosen)
    identity = {"sha256": recording["sha256"], "recording_id": recording_id,
                "start": t_start, "end": t_end, "n_frames": n_frames,
                "sampling_version": SAMPLING_VERSION, "max_width": 960}
    key = json_key(identity)
    folder = Path(cache_dir).resolve() / "samples" / key
    packet_path = folder / "sample.json"
    if packet_path.exists():
        packet = json.loads(packet_path.read_text())
        if all(Path(frame["path"]).exists() and digest_file(frame["path"]) == frame["sha256"] for frame in packet["frames"]):
            return packet
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as temporary:
        temporary = Path(temporary)
        selection = "+".join(f"eq(n\\,{i})" for i in indices)
        run_media(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", recording["video_path"],
                   "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
                   f"select={selection},scale=w='min(960,iw)':h=-2", "-fps_mode", "vfr",
                   "-frames:v", str(len(indices)), "-q:v", "3", str(temporary / "%03d.jpg")])
        images = sorted(temporary.glob("*.jpg"))
        if len(images) != len(indices):
            raise ValueError("Decoded frame count did not match the requested sample")
        folder.mkdir(parents=True, exist_ok=True)
        evidence = []
        for image, index in zip(images, indices):
            target = folder / image.name
            image.replace(target)
            evidence.append({"frame_id": f"{recording['sha256'][:16]}:{index}",
                             "frame_index": index, "timestamp": times[index], "path": str(target),
                             "sha256": digest_file(target)})
        packet = {"sample_id": key, "recording_id": recording_id, "window": [t_start, t_end],
                  "sampling_version": SAMPLING_VERSION, "source_sha256": recording["sha256"],
                  "requested_frames": n_frames, "frames": evidence,
                  "uncertainties": ["Sparse frames may miss intermediate actions."]}
        if len(indices) < n_frames:
            packet["uncertainties"].append("Fewer distinct frames were available than requested.")
        write_json(packet_path, packet)
        return packet
