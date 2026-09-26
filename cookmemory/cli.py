import argparse
import json
import os
import tempfile
from pathlib import Path

from .captaincook import evaluation_records
from .evaluate import compare, markdown_table
from .harness import replay
from .policy import HISTORY_PATH, POLICY_PATH, load_policy, promote, propose_change, validate_change
from .render import render
from .store import AtlasStore, SQLiteStore
from db import load_env_file


def read_events(path):
    with open(path) as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def cmd_replay(args):
    if args.env_file:
        load_env_file(args.env_file)
    store = AtlasStore() if args.backend == "atlas" else SQLiteStore(args.db)
    policy = load_policy(args.policy) if args.policy else None
    try:
        decisions = replay(store, args.session, read_events(args.events), args.mode, args.limit, policy,
                           max_events=args.max_events, max_seconds=args.max_seconds)
        try:
            for decision in decisions:
                line = json.dumps(decision)
                print(line)
        finally:
            if args.decisions_out:
                # Export durable decisions, including earlier passes. Skip
                # receipts are console progress, not UI decision records.
                args.decisions_out.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", dir=args.decisions_out.parent,
                                                      delete=False) as out:
                        temporary = out.name
                        for decision in store.decisions(args.session):
                            out.write(json.dumps(decision) + "\n")
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(temporary, args.decisions_out)
                finally:
                    if temporary and Path(temporary).exists():
                        Path(temporary).unlink()
    finally:
        store.close()


def cmd_compare(args):
    events = read_jsonl(args.events)
    labels = read_jsonl(args.labels) if args.labels else []
    policy = load_policy(args.policy) if args.policy else None
    results = compare(events, labels, policy)
    memory, stateless = results["memory"], results["stateless"]
    print(markdown_table("memory", memory["metrics"], "stateless", stateless["metrics"]))
    if args.json_out:
        out_dir = Path(args.json_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = args.tag or "compare"
        for mode in ("memory", "stateless"):
            path = out_dir / f"{tag}-{mode}.json"
            path.write_text(json.dumps(results[mode]["metrics"], indent=2) + "\n")
            print(f"Wrote {path}")
    if args.decisions_out:
        out_dir = Path(args.decisions_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode in ("memory", "stateless"):
            path = out_dir / f"{mode}.jsonl"
            with path.open("w") as target:
                for decision in results[mode]["decisions"]:
                    target.write(json.dumps(decision) + "\n")
            print(f"Wrote {path}")


def cmd_propose_policy(args):
    current = load_policy(args.current) if args.current else load_policy()
    requires = json.loads(args.requires)
    dev_metrics = {"order_missed": args.order_missed}
    new_policy, message = propose_change(dev_metrics, current, requires)
    print(message)
    if new_policy is None:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(new_policy, indent=2) + "\n")
    print(f"Proposed policy v{new_policy['version']} written to {out} (not yet promoted)")


def cmd_validate_policy(args):
    def load(p):
        return json.loads(Path(p).read_text())

    policy = load(args.policy)
    ok, checks = validate_change(load(args.dev_baseline), load(args.dev_candidate),
                                 load(args.heldout_baseline), load(args.heldout_candidate))
    print(json.dumps(checks, indent=2))
    if not ok:
        print("VALIDATION FAILED: policy not promoted")
        return
    promote(policy, checks, POLICY_PATH, HISTORY_PATH)
    print(f"VALIDATION PASSED: policy v{policy['version']} promoted to {POLICY_PATH}")


def cmd_render_ui(args):
    out = render(args.events, args.memory, args.stateless, args.out)
    print(f"Wrote {out} (open it in a browser)")


def main():
    parser = argparse.ArgumentParser(description="CookMemory: persistent procedural-memory harness")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("replay")
    run.add_argument("events", type=Path)
    run.add_argument("--session", required=True)
    run.add_argument("--backend", choices=["local", "atlas"], default="local")
    run.add_argument("--db", default="work/memory.sqlite")
    run.add_argument("--env-file", type=Path, help="Explicit ignored KEY=VALUE credentials file")
    run.add_argument("--max-events", type=int, default=10000, help="Total unique-event budget per session")
    run.add_argument("--max-seconds", type=float, default=300, help="Time budget for this invocation")
    run.add_argument("--mode", choices=["memory", "stateless"], default="memory")
    run.add_argument("--limit", type=int, help="Stop after this many NEW events; rerun to resume")
    run.add_argument("--decisions-out", type=Path, help="Write decisions as JSONL for the replay UI")
    run.add_argument("--policy", type=Path, help="Policy JSON (from propose-policy) applied to replay")
    run.set_defaults(func=cmd_replay)

    labels = commands.add_parser("prepare-labels")
    labels.add_argument("annotations", type=Path, help="Official error_annotations.json")
    labels.add_argument("--output", type=Path, required=True)
    labels.set_defaults(func=_prepare_labels)

    cmp = commands.add_parser("compare")
    cmp.add_argument("--events", type=Path, required=True)
    cmp.add_argument("--labels", type=Path, help="Evaluator-only labels JSONL")
    cmp.add_argument("--policy", type=Path, help="Apply a proposed policy to both modes")
    cmp.add_argument("--tag", default="compare", help="Tag for --json-out filenames")
    cmp.add_argument("--json-out", type=Path, help="Write metrics JSON per mode")
    cmp.add_argument("--decisions-out", type=Path, help="Write decisions JSONL per mode (for render-ui)")
    cmp.set_defaults(func=cmd_compare)

    propose = commands.add_parser("propose-policy")
    propose.add_argument("--order-missed", type=int, required=True,
                         help="Missed order-tagged errors measured on dev recordings")
    propose.add_argument("--requires", required=True,
                         help='Task-graph prerequisites as JSON, e.g. \'{"step-b": ["step-a"]}\'')
    propose.add_argument("--current", type=Path, help="Current policy JSON (defaults to work/policy.json)")
    propose.add_argument("--out", type=Path, required=True, help="Where to write the proposed (unpromoted) policy")
    propose.set_defaults(func=cmd_propose_policy)

    validate = commands.add_parser("validate-policy")
    validate.add_argument("--policy", type=Path, required=True, help="Proposed policy JSON to validate and promote")
    validate.add_argument("--dev-baseline", type=Path, required=True)
    validate.add_argument("--dev-candidate", type=Path, required=True)
    validate.add_argument("--heldout-baseline", type=Path, required=True)
    validate.add_argument("--heldout-candidate", type=Path, required=True)
    validate.set_defaults(func=cmd_validate_policy)

    ui = commands.add_parser("render-ui")
    ui.add_argument("--events", type=Path, required=True)
    ui.add_argument("--memory", type=Path, required=True, help="Decisions JSONL from memory mode")
    ui.add_argument("--stateless", type=Path, help="Decisions JSONL from stateless mode (enables toggle)")
    ui.add_argument("--out", type=Path, default=Path("demo.html"))
    ui.set_defaults(func=cmd_render_ui)

    args = parser.parse_args()
    if args.command == "replay" and args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    args.func(args)


def _prepare_labels(args):
    records = list(evaluation_records(json.loads(args.annotations.read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} evaluation-only records; no observations generated.")


if __name__ == "__main__":
    main()
