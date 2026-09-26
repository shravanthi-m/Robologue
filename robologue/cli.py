import argparse
import json
from pathlib import Path

from .captaincook import evaluation_records
from .evaluate import score
from .harness import replay
from .industreal_labels import reference_records
from .policies import (
    decide_promotion,
    finalize,
    propose_from_false_corrects,
    validate_policy,
)
from .store import AtlasStore, SQLiteStore


def read_events(path):
    with open(path) as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def read_verdicts(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def _print_report(title, report):
    print(
        f"{title}: scorable={report['scorable']} judged={report['judged']} "
        f"excluded={len(report['excluded_checkpoint_ids'])}"
    )
    acc = report["state_accuracy"]
    print(f"  state_accuracy={acc:.3f}" if acc is not None else "  state_accuracy=n/a")
    print(
        f"  false_corrects={report['false_correct_count']} "
        f"incorrect_recall={report['incorrect_recall']} "
        f"abstentions={report['abstention_count']} "
        f"model_calls={report['model_calls']}"
    )


def run_evaluate(args):
    """Score one verdict run, propose a rule, optionally decide promotion."""
    references = list(reference_records(args.recording, args.labels_csv))
    components = sorted({r["component_index"] for r in references})
    frames = sorted({r["frame"] for r in references})
    print(
        f"References: {len(references)} "
        f"(recording {args.recording}, {len(components)} components, {len(frames)} frames)"
    )

    verdicts = read_verdicts(args.verdicts)
    baseline = score(verdicts, references)
    _print_report("Baseline", baseline)

    candidate = propose_from_false_corrects(
        verdicts, references, args.policy_id, args.parent_policy_id
    )
    if candidate is None:
        print("Proposal: no eligible false-correct; no rule manufactured.")
    else:
        print(f"Proposal ({candidate['template_id']}):")
        print(f"  {candidate['checklist'][0]}")
        print(f"  supporting cases: {', '.join(candidate['supporting_dev_case_ids'])}")

    result = {
        "recording_id": args.recording,
        "n_references": len(references),
        "baseline": baseline,
        "proposed_policy": candidate,
    }

    if args.candidate_verdicts:
        cand_report = score(read_verdicts(args.candidate_verdicts), references)
        _print_report("Candidate", cand_report)
        decision = decide_promotion(baseline, cand_report, args.budget)
        print(f"Promotion decision: {decision['decision']}")
        for reason in decision["reasons"]:
            print(f"  - {reason}")
        policy_record = None
        if candidate is not None and decision["decision"] in ("accept", "reject"):
            policy_record = finalize(candidate, decision["decision"])
            print(
                f"Policy {policy_record['policy_id']} frozen as {policy_record['status']}."
            )
        result["candidate"] = cand_report
        result["promotion"] = decision
        result["frozen_policy"] = policy_record

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as target:
            json.dump(result, target, indent=2)
        print(f"Wrote JSON report to {args.output}")


def load_policy(path):
    """Load the immutable policy a run starts with. Accepts a bare policy
    record or an evaluate --output report holding a frozen policy. Only an
    accepted policy may run."""
    doc = json.loads(Path(path).read_text())
    record = doc.get("frozen_policy", doc)
    if not isinstance(record, dict):
        raise ValueError(f"{path}: no frozen policy found")
    validate_policy(record)
    if record["status"] != "accepted":
        raise ValueError(
            f"{path}: only an accepted policy may run "
            f"(status is {record['status']!r})"
        )
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Robologue: durable RGB inspection, memory, and hidden-label evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay")
    run.add_argument("events", type=Path)
    run.add_argument("--session", required=True)
    run.add_argument("--backend", choices=["local", "atlas"], default="local")
    run.add_argument("--db", default="work/memory.sqlite")
    run.add_argument("--mode", choices=["memory", "stateless"], default="memory")
    run.add_argument(
        "--limit", type=int, help="Stop after this many NEW events; rerun to resume"
    )
    run.add_argument(
        "--policy",
        type=Path,
        help="Frozen policy record, or an evaluate --output report "
        "containing one; its checklist rides on verification requests",
    )
    labels = commands.add_parser("prepare-labels")
    labels.add_argument(
        "annotations", type=Path, help="Official error_annotations.json"
    )
    labels.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate",
        description="Score verdicts against IndustReal-format labels, propose a "
        "checklist rule from false-corrects, and optionally decide "
        "promotion against a candidate run.",
    )
    evaluate.add_argument("labels_csv", type=Path, help="PSR_labels_raw.csv")
    evaluate.add_argument("verdicts", type=Path, help="JSONL verdicts (Interface B)")
    evaluate.add_argument("--recording", required=True)
    evaluate.add_argument(
        "--candidate-verdicts",
        type=Path,
        help="JSONL verdicts from the candidate policy run",
    )
    evaluate.add_argument("--policy-id", default="checklist-v2")
    evaluate.add_argument("--parent-policy-id", default="baseline-v1")
    evaluate.add_argument(
        "--budget",
        type=int,
        default=24,
        help="Max model calls allowed for the candidate",
    )
    evaluate.add_argument("--output", type=Path, help="Write the full JSON report here")
    dataset = commands.add_parser(
        "prepare-dataset", help="Extract and audit the supplied IndustReal archives"
    )
    dataset.add_argument("--labels-zip", type=Path, required=True)
    dataset.add_argument("--native-zip", type=Path, required=True)
    dataset.add_argument("--videos-zip", type=Path, required=True)
    dataset.add_argument("--procedure-info", type=Path, required=True)
    dataset.add_argument("--output", type=Path, default=Path("data/industreal"))
    media = commands.add_parser(
        "audit-media", help="Fully decode MP4s and verify native RGB alignment"
    )
    media.add_argument("root", type=Path)
    media.add_argument("--output", type=Path, default=Path("work/evaluation/media"))
    benchmark = commands.add_parser(
        "run-dataset", help="Run budgeted real RGB inference and paired evaluation"
    )
    benchmark.add_argument("root", type=Path)
    benchmark.add_argument("--output", type=Path, default=Path("work/evaluation"))
    benchmark.add_argument("--env-file", type=Path)
    benchmark.add_argument("--backend", choices=["local", "atlas"], default="local")
    benchmark.add_argument("--db", default="work/evaluation/runtime.sqlite")
    benchmark.add_argument("--run-id", default="native-eval-v1")
    benchmark.add_argument("--max-usd", type=float, default=10)
    benchmark.add_argument("--max-calls", type=int, default=250)
    benchmark.add_argument(
        "--budget-db",
        type=Path,
        help="Share one spending ledger across exploratory and final runs",
    )
    benchmark.add_argument(
        "--runtime-dir", type=Path, help="Share content-addressed neutral RGB cache"
    )
    benchmark.add_argument(
        "--components-file", type=Path, help="Public component geometry catalog"
    )
    benchmark.add_argument(
        "--reference-image",
        type=Path,
        help="Public CAD component key; never a labelled recording",
    )
    benchmark.add_argument(
        "--development-only",
        action="store_true",
        help="Pause after development; validation/heldout inference remain untouched",
    )
    args = parser.parse_args()
    if args.command == "prepare-dataset":
        from .dataset import prepare

        print(
            json.dumps(
                prepare(
                    args.labels_zip,
                    args.native_zip,
                    args.videos_zip,
                    args.output,
                    args.procedure_info,
                ),
                indent=2,
            )
        )
        return
    if args.command == "audit-media":
        from .media_audit import audit_media

        report = audit_media(args.root, args.output)
        print(
            f"Fully decoded {report['decode_ok']}/{report['total']} videos; verified {len(report['native_alignment'])} native timelines"
        )
        return
    if args.command == "run-dataset":
        import os
        from .storage_common import load_env_file
        from .model_client import CallBudget, OpenRouterClient
        from .benchmark import run_benchmark

        if args.env_file:
            load_env_file(args.env_file)
        if not os.getenv("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY required; no model request dispatched")
        budget = CallBudget(
            args.budget_db or args.output / "model-budget.sqlite",
            max_usd=args.max_usd,
            max_calls=args.max_calls,
        )
        store = AtlasStore() if args.backend == "atlas" else SQLiteStore(args.db)
        try:
            report = run_benchmark(
                args.root,
                args.output,
                store,
                OpenRouterClient(budget),
                run_id=args.run_id,
                runtime_dir=args.runtime_dir,
                components_file=args.components_file,
                reference_image=args.reference_image,
                development_only=args.development_only,
            )
            if report["status"] == "development_complete":
                _print_report("Development", report["runs"]["development"])
            else:
                _print_report("Memory baseline", report["runs"]["memory_baseline"])
                _print_report("Stateless", report["runs"]["stateless"])
            print(json.dumps(report["budget"]))
        finally:
            store.close()
            budget.close()
        return
    if args.command == "prepare-labels":
        records = list(evaluation_records(json.loads(args.annotations.read_text())))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            for record in records:
                target.write(json.dumps(record) + "\n")
        print(
            f"Wrote {len(records)} evaluation-only records; no observations generated."
        )
        return
    if args.command == "evaluate":
        run_evaluate(args)
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    policy = load_policy(args.policy) if args.policy else None
    if policy:
        print(
            f"Loaded accepted policy {policy['policy_id']} "
            f"({len(policy['checklist'])} checklist rule(s))."
        )
    store = AtlasStore() if args.backend == "atlas" else SQLiteStore(args.db)
    try:
        for decision in replay(
            store, args.session, read_events(args.events), args.mode, args.limit, policy
        ):
            print(json.dumps(decision))
    finally:
        store.close()


if __name__ == "__main__":
    main()
