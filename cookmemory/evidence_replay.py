"""Persist Person 1 evidence and feed Person 2's durable recording replay.

No label reader, correctness classifier or simulator is invoked. JPEG frame IDs
need an explicit verified time map; video packets already carry decoded PTS.
"""
import argparse
import json
import math
import tempfile
from pathlib import Path

from db import load_env_file
from .harness import advance, replay, validate
from .perception.inspect import observation_event
from .store import AtlasStore, SQLiteStore


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("Evidence times must be finite nonnegative seconds")
    return value


def _descriptions(packet):
    for key in ("observations", "uncertainties"):
        values = packet.get(key)
        if not isinstance(values, list) or len(values) > 40 or any(
                not isinstance(v, str) or not v.strip() or len(v) > 2000 for v in values):
            raise ValueError("Packet observations/uncertainties must be bounded string lists")
    if not packet["observations"] and not packet["uncertainties"]:
        raise ValueError("Empty inspection")


def to_event(packet, *, frame_times=None, allow_mock=False):
    """Map verified time coordinates; never infer fps or copy issue/label fields."""
    _descriptions(packet)
    mock = packet.get("source_kind") == "mock"
    if mock and not allow_mock:
        raise ValueError("Mock evidence requires --allow-mock and must not be presented as real")
    if "inspection_id" in packet:  # existing video adapter's exact decoded timestamps
        start, end = packet["window"]
        _number(start); _number(end)
        observed = _number(packet["observed_until_at_creation"])
        if not start < end <= observed or not packet["evidence"]:
            raise ValueError("Invalid video evidence window")
        for frame in packet["evidence"]:
            if not start <= _number(frame["timestamp"]) <= end:
                raise ValueError("Video frame is outside the allowed interval")
        event = observation_event(packet)
    else:
        if packet.get("schema_version") != 1 or packet.get("source_kind") not in ("rgb_vlm", "mock"):
            raise ValueError("Unsupported evidence packet")
        ids, cursor = packet.get("frame_ids"), packet.get("cursor_frame")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 3
                or any(type(v) is not int or v < 0 for v in ids)
                or type(cursor) is not int or cursor < 0
                or ids != sorted(set(ids)) or ids[-1] > cursor):
            raise ValueError("Frame IDs must be ordered, unique and no later than the trusted cursor")
        if (not isinstance(frame_times, dict) or frame_times.get("schema_version") != 1
                or frame_times.get("verified") is not True
                or frame_times.get("recording_id") != packet["recording_id"]):
            raise ValueError("Native frames require a verified time map for the same recording; no FPS is assumed")
        if not isinstance(frame_times.get("mapping_id"), str) or not frame_times["mapping_id"]:
            raise ValueError("Time map requires a provenance mapping_id")
        times = frame_times.get("frame_seconds")
        if not isinstance(times, dict):
            raise ValueError("Time map requires frame_seconds")
        try:
            positions = sorted(set(ids + [cursor]))
            mapped = [_number(times[str(i)]) for i in positions]
        except KeyError:
            raise ValueError("Time map is missing a selected frame or cursor") from None
        if any(a >= b for a, b in zip(mapped, mapped[1:])):
            raise ValueError("Distinct frame IDs must have strictly increasing timestamps")
        timestamp = times[str(cursor)]
        text = "Observed: " + "; ".join(packet["observations"])
        text += " | Uncertain: " + "; ".join(packet["uncertainties"])
        text += " | Evidence inspection: " + packet["evidence_id"]
        event = {"schema_version": 1, "event_id": packet["evidence_id"],
                 "recording_id": packet["recording_id"], "timestamp": timestamp,
                 "observation": text, "evidence_refs": [{
                     "recording_id": packet["recording_id"], "start_seconds": times[str(ids[0])],
                     "end_seconds": times[str(ids[-1])], "frame_ids": [str(i) for i in ids]}]}
    if mock:
        event["observation"] = "MOCK EVIDENCE - " + event["observation"]
    # Deliberately no completed_steps/issue/resolves: a verifier must supply those.
    validate(event)
    return event


def ingest(store, session, packets, *, frame_times=None, allow_mock=False):
    """Preflight the batch, save evidence, then use the existing receipt/outbox loop.

    A crash after evidence insertion can leave an unprocessed inspection, which is
    safe to re-ingest. Durable decisions are projected by Person 2's replay.
    """
    if not isinstance(session, str) or not session.strip():
        raise ValueError("A nonempty recording session is required")
    if not isinstance(packets, list) or not 1 <= len(packets) <= 1000:
        raise ValueError("Supply 1-1000 evidence packets")
    if len(json.dumps(packets, allow_nan=False).encode()) > 8_000_000:
        raise ValueError("Evidence batch exceeds 8 MB; use smaller batches")
    events = [to_event(p, frame_times=frame_times, allow_mock=allow_mock) for p in packets]
    # Check ordering/session compatibility before any evidence insertion.
    state = store.load(session)
    for event in events:
        state, _ = advance(state, event)
    for packet, event in zip(packets, events):
        stable = {k: v for k, v in packet.items()
                  if k not in ("cache_hit", "new_api_calls", "logical_tool_calls")}
        doc = {"schema_version": 1, "kind": "inspection_only", "packet": stable, "event": event}
        if "inspection_id" not in packet:
            # Retain only the mapped coordinates needed for this packet.
            needed = set(packet["frame_ids"] + [packet["cursor_frame"]])
            doc["time_mapping"] = {"mapping_id": frame_times["mapping_id"],
                                   "frame_seconds": {str(i): frame_times["frame_seconds"][str(i)]
                                                     for i in sorted(needed)}}
        store.put_inspection(session, event["event_id"], doc)
    receipts = list(replay(store, session, events))
    return {"events": events, "receipts": receipts, "decisions": store.decisions(session)}


def _read_packets(path):
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    return value if isinstance(value, list) else [value]


def _export(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        name = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    name.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packets", type=Path, help="Full inspection JSON, list, or JSONL; not label files")
    parser.add_argument("--session", required=True)
    parser.add_argument("--backend", choices=("local", "atlas"), default="local")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--db", default="work/memory.sqlite")
    parser.add_argument("--frame-times", type=Path, help="Verified native-frame-to-seconds map")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--events-out", type=Path, required=True)
    parser.add_argument("--decisions-out", type=Path, required=True)
    args = parser.parse_args()
    # Avoid clobbering an input, the database or another output during export.
    paths = [args.packets, args.events_out, args.decisions_out, Path(args.db)]
    paths += [p for p in (args.frame_times, args.env_file) if p]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Input, database, credentials and output paths must be distinct")
    if args.env_file:
        load_env_file(args.env_file)
    mapping = json.loads(args.frame_times.read_text()) if args.frame_times else None
    packets = _read_packets(args.packets)
    # Validate before connecting to a live backend.
    for packet in packets:
        to_event(packet, frame_times=mapping, allow_mock=args.allow_mock)
    store = AtlasStore() if args.backend == "atlas" else SQLiteStore(args.db)
    try:
        result = ingest(store, args.session, packets, frame_times=mapping, allow_mock=args.allow_mock)
        _export(args.events_out, result["events"])
        _export(args.decisions_out, result["decisions"])
        print(json.dumps({"backend": args.backend, "session": args.session,
                          "inspections_submitted": len(packets),
                          "durable_decisions": len(result["decisions"]),
                          "classification": "not_performed; neutral inspections only"}))
    finally:
        store.close()


if __name__ == "__main__":
    main()
