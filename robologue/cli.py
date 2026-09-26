import argparse
import json
from pathlib import Path

from .captaincook import evaluation_records
from .harness import replay
from .store import AtlasStore, SQLiteStore


def read_events(path):
    with open(path) as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Robologue starter: synthetic replay or CaptainCook4D label preparation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay")
    run.add_argument("events", type=Path)
    run.add_argument("--session", required=True)
    run.add_argument("--backend", choices=["local", "atlas"], default="local")
    run.add_argument("--db", default="work/memory.sqlite")
    run.add_argument("--mode", choices=["memory", "stateless"], default="memory")
    run.add_argument("--limit", type=int, help="Stop after this many NEW events; rerun to resume")
    labels = commands.add_parser("prepare-labels")
    labels.add_argument("annotations", type=Path, help="Official error_annotations.json")
    labels.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-labels":
        records = list(evaluation_records(json.loads(args.annotations.read_text())))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            for record in records:
                target.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} evaluation-only records; no observations generated.")
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    store = AtlasStore() if args.backend == "atlas" else SQLiteStore(args.db)
    try:
        for decision in replay(store, args.session, read_events(args.events), args.mode, args.limit):
            print(json.dumps(decision))
    finally:
        store.close()


if __name__ == "__main__":
    main()
