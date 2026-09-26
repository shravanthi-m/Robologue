# Synthetic evaluation fixtures

Hand-authored demo data in the official IndustReal `PSR_labels_raw.csv`
shape (no header; rows are `frame_name,state_0,state_1,...`). These are
NOT real IndustReal recordings. They exist so the `evaluate` CLI command
runs end to end without the licensed dataset.

- `PSR_labels_raw.csv`: 5 frames x 2 components.
- `baseline-verdicts.jsonl`: 10 verdicts with two false approvals
  (declaring `correct` on incorrect components).
- `candidate-verdicts.jsonl`: the same run with the candidate policy
  applied, turning those two approvals into `insufficient_evidence`.

Run:

```bash
python -m robologue.cli evaluate examples/eval-synthetic/PSR_labels_raw.csv \
  examples/eval-synthetic/baseline-verdicts.jsonl --recording rec-SYN \
  --candidate-verdicts examples/eval-synthetic/candidate-verdicts.jsonl \
  --policy-id checklist-v2 --parent-policy-id baseline-v1 --budget 24
```
