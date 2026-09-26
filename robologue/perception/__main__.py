import argparse
import json
import sys
from pathlib import Path

from .inspect import inspect_video, observation_event
from .video import load_manifest, register_video, sample_video
from .video import write_json


def main():
    parser = argparse.ArgumentParser(description="Person 1: local video registration, sampling, and cached VLM inspection")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("video")
    register.add_argument("--recording", required=True)
    register.add_argument("--manifest", default="work/recordings.json")
    register.add_argument("--source", default="captaincook4d")
    register.add_argument("--replace", action="store_true")
    listing = commands.add_parser("list")
    listing.add_argument("--manifest", default="work/recordings.json")
    for name in ("sample", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", default="work/recordings.json")
        command.add_argument("--recording", required=True)
        command.add_argument("--start", type=float, required=True)
        command.add_argument("--end", type=float, required=True)
        command.add_argument("--observed-until", type=float, required=True)
        command.add_argument("--frames", type=int, default=3)
        command.add_argument("--cache", default="work/perception")
        if name == "inspect":
            command.add_argument("--model", help="Explicit OpenRouter model ID; otherwise OPENROUTER_MODEL")
            command.add_argument("--packet-output", type=Path, help="New full inspection JSON for evidence_replay ingestion")
            command.add_argument("--env-file", type=Path, help="Explicit ignored credentials file (Person 2 loader)")
            command.add_argument("--event-output", type=Path, help="Create a NEW replay-compatible single-event JSONL file")
    args = parser.parse_args()
    try:
        if getattr(args, "env_file", None):
            from ..storage_common import load_env_file
            load_env_file(args.env_file)
        if args.command == "register":
            result = register_video(args.manifest, args.recording, args.video, source=args.source, replace=args.replace)
        elif args.command == "list":
            result = load_manifest(args.manifest)
        else:
            if args.command == "inspect" and args.event_output and args.event_output.exists():
                raise ValueError("Event output already exists; choose a new file.")
            if args.command == "inspect" and args.packet_output and args.packet_output.exists():
                raise ValueError("Packet output already exists; choose a new file.")
            kwargs = {"cache_dir": args.cache}
            function = sample_video
            if args.command == "inspect":
                function = inspect_video
                kwargs["model"] = args.model
            result = function(args.manifest, args.recording, args.start, args.end,
                              args.frames, args.observed_until, **kwargs)
            if args.command == "inspect" and args.packet_output:
                write_json(args.packet_output, result)
            if args.command == "inspect" and args.event_output:
                args.event_output.parent.mkdir(parents=True, exist_ok=True)
                with args.event_output.open("x") as target:
                    target.write(json.dumps(observation_event(result)) + "\n")
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
