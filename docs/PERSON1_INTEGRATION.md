# Person 1 + Person 2 recording integration

Based on Person 2 commit d1f5eac. This branch preserves its runtime/store contracts.
The latest bridge was syntax-checked; its integration test suite was not run at the
user's request. Earlier standalone Person 1 tests passed before these changes.
No new live VLM or Atlas run is claimed.

## Video -> VLM -> persistent recording replay

Use the existing video registration CLI for the downloaded RGB videos. Replace
paths, recording IDs, intervals and model configuration with verified values.

```bash
python3 -m robologue.perception register /path/to/recording.mp4 --recording RECORDING_ID --source industreal
python3 -m robologue.perception inspect --recording RECORDING_ID --start 0 --end 10 --observed-until 10 --env-file .env --packet-output work/inspection-01.json
python3 -m robologue.evidence_replay work/inspection-01.json --session recording-demo --backend atlas --env-file .env --events-out work/events.jsonl --decisions-out work/decisions.jsonl
python3 -m robologue.cli render-ui --events work/events.jsonl --memory work/decisions.jsonl --out work/replay.html
```

The inspection command reads OPENROUTER_MODEL and OPENROUTER_API_KEY from the explicit
ignored environment file. The replay command uses Person 2's required Atlas mode.
For offline development use `--backend local --db work/memory.sqlite` instead.
No paid model request is made by evidence_replay: it ingests already-generated packets.

Full neutral packets are inserted immutably into `recording_inspections`; the same
session also uses Person 2's `sessions` and `recording_decisions` collections.
Re-ingesting the same packet safely resumes/projections without duplicating decisions.
Cache-hit and new-call flags are excluded from immutable identity comparison.
Changed evidence or time mapping under an existing inspection identity is rejected.
Observation events contain recording/time/frame evidence references and deliberately
omit completed_steps, issue and resolves: the model inspection is not a failure verdict.

## Native JPEG frames require explicit time mapping

`robologue.industreal_data` preserves numeric source frame IDs. Person 2's replay
requires seconds. Supply a verified mapping; this bridge does not assume 10 FPS.

Example mapping shape only (values must be checked against actual source alignment):

```json
{
  "schema_version": 1,
  "recording_id": "RECORDING_ID",
  "mapping_id": "reviewed-source-map-v1",
  "verified": true,
  "frame_seconds": {"10": 0.0, "20": 1.0, "40": 3.0}
}
```

Include every selected frame and the observation cursor. The mapping must belong to
the same recording and be strictly chronological. A mapping to annotation seconds
must account for offsets/clipping; setting verified=true is a human assertion, not
an automated alignment check.

```bash
python3 -m robologue.evidence_replay work/evidence-01.json --frame-times work/frame-times.json --session recording-demo --backend atlas --env-file .env --events-out work/events.jsonl --decisions-out work/decisions.jsonl
```

Mock packets require explicit `--allow-mock` and retain a MOCK marker in the event.
Use local mode for mock development. Ground-truth action/PSR annotations are never
read by the bridge. Save labels for the evaluator.

## Export/resume behavior

An input can be one packet JSON, a JSON list, or JSONL. Supply the full chronological
packet list when exporting the full report: events-out represents this input batch,
whereas decisions-out contains all durable decisions for the session. The bridge
preflights recording/order compatibility before inserting evidence. A crash after
inspection insertion may leave an unprocessed inspection; rerunning is safe.
There is one worker per session. Use a new session for a different recording/policy.

Remaining work: a verifier to infer evidence-backed component failures/completion,
PSR-label evaluation, real-data alignment and a live VLM-to-Atlas smoke run.
