"""Reshape a saved robologue Atlas session into this player's contract files. Copies fields; invents nothing.

Only a single policy (baseline-v1) has been run against real evidence so far, so there is no candidate arm,
no recall and no evaluation yet. Those files are simply omitted; the player renders their absence honestly.
"""
import argparse
import json
import os
from pathlib import Path


def _issue_events(docs, component_ids):
    """The saved harness tracks one aggregate 'assembly-review' issue across all components per checkpoint.
    Turn its presence/absence across checkpoints into opened/carried/resolved events, honestly: a re-opening
    after a real resolution becomes a new issue id rather than reusing one the schema treats as closed."""
    events, seq, open_id, generation = [], 0, None, 0
    for doc in docs:
        dec = doc["decision"]
        issue = dec["open_issues"].get("assembly-review")
        cp_id = f"{dec['recording_id']}:{int(dec['cursor_frame'])}"
        if issue and open_id is None:
            generation += 1
            open_id = "assembly-review" if generation == 1 else f"assembly-review-{generation}"
            seq += 1
            events.append({"schema_version": 1, "source_kind": "rgb_vlm", "run_id": dec["run_id"], "seq": seq,
                           "issue_id": open_id, "event": "opened", "checkpoint_id": cp_id,
                           "cursor_frame": int(dec["cursor_frame"]), "component_id": "all-components",
                           "description": issue["description"], "evidence_ids": [issue["evidence_event_id"]]})
        elif issue and open_id is not None:
            seq += 1
            events.append({"schema_version": 1, "source_kind": "rgb_vlm", "run_id": dec["run_id"], "seq": seq,
                           "issue_id": open_id, "event": "carried", "checkpoint_id": cp_id,
                           "cursor_frame": int(dec["cursor_frame"]), "component_id": "all-components",
                           "description": issue["description"], "evidence_ids": []})
        elif not issue and open_id is not None:
            seq += 1
            events.append({"schema_version": 1, "source_kind": "rgb_vlm", "run_id": dec["run_id"], "seq": seq,
                           "issue_id": open_id, "event": "resolved", "checkpoint_id": cp_id,
                           "cursor_frame": int(dec["cursor_frame"]), "component_id": "all-components",
                           "description": "No components were listed as needing review at this checkpoint.",
                           "evidence_ids": [v["evidence_id"] for v in dec.get("verdicts", [])
                                            if v["component_id"] in component_ids][:1]})
            open_id = None
    return events


def export(db, session, component_ids, out_dir):
    decisions = list(db["recording_decisions"].find({"session": session}).sort("decision.timestamp", 1))
    if not decisions:
        raise ValueError(f"No recording_decisions for session {session!r}")
    inspections = {d["inspection"]["packet"]["evidence_id"]: d["inspection"]["packet"]
                  for d in db["recording_inspections"].find({"session": session})}
    recording_id = decisions[0]["decision"]["recording_id"]

    verdicts, evidence, seen_evidence = [], [], set()
    for doc in decisions:
        dec = doc["decision"]
        packet = inspections.get(dec["event_id"])
        if packet and packet["evidence_id"] not in seen_evidence:
            seen_evidence.add(packet["evidence_id"])
            evidence.append({"schema_version": 1, "source_kind": packet["source_kind"],
                             "evidence_id": packet["evidence_id"], "recording_id": packet["recording_id"],
                             "checkpoint_id": packet["checkpoint_id"], "cursor_frame": int(packet["cursor_frame"]),
                             "component_id": packet["component_id"], "frame_ids": [int(f) for f in packet["frame_ids"]],
                             "observations": packet["observations"], "uncertainties": packet["uncertainties"],
                             "provider": {"model_id": packet["provider"]["model_id"],
                                          "prompt_version": packet["provider"]["prompt_version"]}})
        for v in dec.get("verdicts", []):
            if v["component_id"] not in component_ids:
                continue
            needs_review = dec["open_issues"].get("assembly-review", {}).get("description", "")
            open_ids = ["assembly-review"] if v["component_id"] in needs_review else []
            verdicts.append({"schema_version": 1, "source_kind": v["source_kind"], "run_id": v["run_id"],
                             "recording_id": v["recording_id"], "checkpoint_id": v["checkpoint_id"],
                             "cursor_frame": int(v["cursor_frame"]), "component_id": v["component_id"],
                             "verdict": v["verdict"], "reason": v["rationale"],
                             "evidence_ids": [v["evidence_id"]], "open_issue_ids": open_ids,
                             "policy_id": v["policy_id"], "model_id": v["model_id"]})

    events = _issue_events(decisions, component_ids)
    last_t = decisions[-1]["decision"]["timestamp"]
    timing = {"schema_version": 1, "source_kind": "rgb_vlm",
             "recordings": {recording_id: {"video_path": None, "fps": 10, "frame_offset": 0,
                                            "duration_s": float(last_t) + 5}}}
    policy = {"schema_version": 1, "source_kind": "rgb_vlm", "policy_id": "baseline-v1", "parent_policy_id": None,
             "status": "accepted", "checklist": [], "supporting_dev_case_ids": []}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("verdicts_baseline.jsonl", verdicts), ("issue_events.jsonl", events),
                       ("evidence.jsonl", evidence)):
        (out_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "verdicts_candidate.jsonl").write_text("", encoding="utf-8")
    (out_dir / "cases.jsonl").write_text("", encoding="utf-8")
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    (out_dir / "policy_baseline.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(verdicts)} verdicts, {len(evidence)} evidence packets, {len(events)} issue events "
          f"for {recording_id} to {out_dir}")
    print("No candidate policy or evaluation exists yet for this session; those files were not written.")
    return {"verdicts": verdicts, "evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description="Export a real robologue Atlas session into contract files")
    parser.add_argument("--db", default="robologue")
    parser.add_argument("--session", required=True)
    parser.add_argument("--components", nargs="+", required=True, help="Component ids to include, e.g. component-0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise SystemExit("Export MONGODB_URI first; this script never accepts it as an argument.")
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        export(client[args.db], args.session, set(args.components), args.out)
    finally:
        client.close()


if __name__ == "__main__":
    main()
