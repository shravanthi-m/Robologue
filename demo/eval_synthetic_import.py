"""Reshape robologue's own eval-synthetic fixture + a real `evaluate --output` report into contract
files. The verdicts and the promotion decision are real robologue.cli output; only the two fields the
fixture never captured (a per-checkpoint reason and evidence references) are filled with an honest
placeholder that says exactly that, rather than invented rationale."""
import argparse
import csv
import json
from pathlib import Path

STATE = {"1": "correct", "0": "not_completed", "-1": "incorrect"}

NO_REASON = "No rationale was saved for this checkpoint by the synthetic evaluation fixture."


def _verdict(v):
    return {**v, "reason": NO_REASON, "evidence_ids": [], "open_issue_ids": []}


def export(labels_csv, baseline_path, candidate_path, report_path, out_dir):
    baseline = [json.loads(l) for l in baseline_path.read_text().splitlines() if l.strip()]
    candidate = [json.loads(l) for l in candidate_path.read_text().splitlines() if l.strip()]
    report = json.loads(report_path.read_text())
    recording_id = baseline[0]["recording_id"]
    max_frame = max(v["cursor_frame"] for v in baseline)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("verdicts_baseline.jsonl", baseline), ("verdicts_candidate.jsonl", candidate)):
        (out_dir / name).write_text("".join(json.dumps(_verdict(v)) + "\n" for v in rows), encoding="utf-8")
    for name in ("evidence.jsonl", "issue_events.jsonl"):
        (out_dir / name).write_text("", encoding="utf-8")

    # This fixture has no separate development recording distinct from the one being replayed, so a
    # genuinely recallable past-failure case can't be shown honestly. Record the real miss for
    # reference, but don't feed it into supporting_dev_case_ids below (that would recall-and-refuse it
    # at every checkpoint, which is just noise for something the fixture can't actually demonstrate).
    by_checkpoint = {v["checkpoint_id"]: v for v in baseline}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        label_rows = list(csv.reader(f))
    cases = []
    for case_id in report["proposed_policy"]["supporting_dev_case_ids"]:
        v = by_checkpoint.get(case_id)
        if v is None:
            continue
        frame = v["cursor_frame"]
        col = 1 + int(v["component_id"].split("-")[-1])
        revealed = STATE[label_rows[frame][col]]
        cases.append({"schema_version": 1, "source_kind": "synthetic-demo", "case_id": case_id,
                      "split": "development", "recording_id": recording_id, "component_id": v["component_id"],
                      "frame_start": frame, "frame_end": frame, "clip_path": f"media/{case_id}.mp4",
                      "verdict_then": v["verdict"], "revealed_outcome": revealed,
                      "summary": f"Development miss the evaluator scored: baseline said {v['verdict']}, "
                                 f"the official label says {revealed}. Not recallable in this fixture: "
                                 f"there is no development recording separate from the one replayed here."})
    (out_dir / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")

    timing = {"schema_version": 1, "source_kind": "synthetic-demo",
             "recordings": {recording_id: {"video_path": None, "fps": 1, "frame_offset": 0,
                                            "duration_s": max_frame + 1}}}
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")

    policy_baseline = {"schema_version": 1, "source_kind": "synthetic-demo", "policy_id": "baseline-v1",
                       "parent_policy_id": None, "status": "accepted", "checklist": [],
                       "supporting_dev_case_ids": []}
    (out_dir / "policy_baseline.json").write_text(json.dumps(policy_baseline, indent=2) + "\n", encoding="utf-8")
    # supporting_dev_case_ids is cleared here (see the cases.jsonl comment above); the real ids are
    # preserved in each case record's case_id instead, so nothing is lost, just not force-recalled.
    frozen = {**report["frozen_policy"], "schema_version": 1, "source_kind": "synthetic-demo",
             "supporting_dev_case_ids": []}
    (out_dir / "policy_candidate.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    def metrics(block):
        return {"state_accuracy": block["state_accuracy"], "false_correct": block["false_correct_count"],
                "incorrect_recall": block["incorrect_recall"], "abstentions": block["abstention_count"],
                "model_calls": block["model_calls"], "latency_ms": block["latency_ms"]}

    promo = report["promotion"]
    decision = {"accept": "accepted", "reject": "rejected", "inconclusive": "inconclusive",
               "no_proposal": "no_proposal"}[promo["decision"]]
    evaluation = {"schema_version": 1, "source_kind": "synthetic-demo", "recording_ids": [recording_id],
                 "baseline_policy_id": "baseline-v1", "candidate_policy_id": frozen["policy_id"],
                 "decision": decision, "reason": " ".join(promo["reasons"]),
                 "scored": report["baseline"]["n_references"], "excluded": 0,
                 "baseline": metrics(report["baseline"]), "candidate": metrics(report["candidate"])}
    (out_dir / f"evaluation_{decision}.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(baseline)} baseline + {len(candidate)} candidate verdicts for {recording_id}; "
          f"promotion decision: {decision}")


def main():
    p = argparse.ArgumentParser(description="Import robologue's eval-synthetic fixture + report into contracts")
    p.add_argument("--labels-csv", type=Path, default=Path("examples/eval-synthetic/PSR_labels_raw.csv"))
    p.add_argument("--baseline", type=Path, default=Path("examples/eval-synthetic/baseline-verdicts.jsonl"))
    p.add_argument("--candidate", type=Path, default=Path("examples/eval-synthetic/candidate-verdicts.jsonl"))
    p.add_argument("--report", type=Path, required=True, help="Output of `robologue.cli evaluate --output`")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    export(args.labels_csv, args.baseline, args.candidate, args.report, args.out)


if __name__ == "__main__":
    main()
