import argparse
import json
from pathlib import Path

from .captaincook import evaluation_records
from .evaluate import score
from .harness import replay
from .industreal_labels import reference_records
from .policies import decide_promotion, finalize, propose_from_false_corrects
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
    print(f"{title}: scorable={report['scorable']} judged={report['judged']} "
          f"excluded={len(report['excluded_checkpoint_ids'])}")
    acc = report["state_accuracy"]
    print(f"  state_accuracy={acc:.3f}" if acc is not None else "  state_accuracy=n/a")
    print(f"  false_corrects={report['false_correct_count']} "
          f"incorrect_recall={report['incorrect_recall']} "
          f"abstentions={report['abstention_count']} "
          f"model_calls={report['model_calls']}")


def run_evaluate(args):
    """Score one verdict run, propose a rule, optionally decide promotion."""
    references = list(reference_records(args.recording, args.labels_csv))
    components = sorted({r["component_index"] for r in references})
    frames = sorted({r["frame"] for r in references})
    print(f"References: {len(references)} "
          f"(recording {args.recording}, {len(components)} components, {len(frames)} frames)")

    verdicts = read_verdicts(args.verdicts)
    baseline = score(verdicts, references)
    _print_report("Baseline", baseline)

    candidate = propose_from_false_corrects(
        verdicts, references, args.policy_id, args.parent_policy_id)
    if candidate is None:
        print("Proposal: no eligible false-correct; no rule manufactured.")
    else:
        print(f"Proposal ({candidate['template_id']}):")
        print(f"  {candidate['checklist'][0]}")
        print(f"  supporting cases: {', '.join(candidate['supporting_dev_case_ids'])}")

    result = {"recording_id": args.recording,
              "n_references": len(references),
              "baseline": baseline,
              "proposed_policy": candidate}

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
            print(f"Policy {policy_record['policy_id']} frozen as {policy_record['status']}.")
        result["candidate"] = cand_report
        result["promotion"] = decision
        result["frozen_policy"] = policy_record

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as target:
            json.dump(result, target, indent=2)
        print(f"Wrote JSON report to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Robologue: synthetic replay, label preparation, and evaluation")
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
    evaluate = commands.add_parser(
        "evaluate",
        description="Score verdicts against IndustReal-format labels, propose a "
                    "checklist rule from false-corrects, and optionally decide "
                    "promotion against a candidate run.")
    evaluate.add_argument("labels_csv", type=Path, help="PSR_labels_raw.csv")
    evaluate.add_argument("verdicts", type=Path, help="JSONL verdicts (Interface B)")
    evaluate.add_argument("--recording", required=True)
    evaluate.add_argument("--candidate-verdicts", type=Path,
                          help="JSONL verdicts from the candidate policy run")
    evaluate.add_argument("--policy-id", default="checklist-v2")
    evaluate.add_argument("--parent-policy-id", default="baseline-v1")
    evaluate.add_argument("--budget", type=int, default=24,
                          help="Max model calls allowed for the candidate")
    evaluate.add_argument("--output", type=Path, help="Write the full JSON report here")
    args = parser.parse_args()
    if args.command == "prepare-labels":
        records = list(evaluation_records(json.loads(args.annotations.read_text())))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            for record in records:
                target.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} evaluation-only records; no observations generated.")
        return
    if args.command == "evaluate":
        run_evaluate(args)
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
