# Evaluation audit results

Date: September 26, 2026. Branch: `codex/integrated-video-evaluation`.

## Result status

**136 offline tests passed. All 86 MP4s fully decoded. All nine native recordings
passed sampled RGB/video alignment. Live Atlas restart checks passed for all 62
checkpoint packets.**

**Real visual evaluation is in progress.** No completed benchmark accuracy is
claimed yet. The ignored environment now contains the supplied vision credential.
All exploratory/final attempts share a persistent $10 ledger. Mocked observations
remain contract fixtures only. Current spending snapshot is recorded in the
machine-readable audit; failed and uncertain-charge attempts are included.

Machine-readable evidence: [audit-results.json](audit-results.json).
Implementation, run commands and frozen protocol: [INTEGRATION.md](INTEGRATION.md).

## Inputs inspected

| Recording | Native RGB frames | PSR checkpoints | Component-state cases |
|---|---:|---:|---:|
| 03_assy_0_1 | 2754 | 6 | 66 |
| 03_assy_1_3 | 2630 | 8 | 88 |
| 03_main_0_1 | 1348 | 6 | 66 |
| 08_assy_0_1 | 5276 | 8 | 88 |
| 08_assy_2_4 | 3443 | 6 | 66 |
| 08_main_0_1 | 2595 | 8 | 88 |
| 09_assy_0_1 | 3501 | 7 | 77 |
| 09_assy_3_1 | 3824 | 7 | 77 |
| 09_main_0_1 | 2554 | 6 | 66 |

Totals: 27,925 RGB images, 62 checkpoint frames, 682 component states.
Reference distribution: 380 correct, 286 not completed, 16 incorrect.
There are no missing checkpoint JPEGs or gaps in native frame numbering.

The action label archive contains 9,273 rows across 36 train / 16 validation /
32 test recordings, with no recording overlap between splits and no class-ID/name
conflicts. All 84 labelled recording IDs have MP4s. Two extra videos have no action
labels: `15_main_0_1` and `21_assy_1_1`. Each native AR file exactly matches its
recording's rows in the global test CSV.

### Boundary discrepancy

One of 9,273 action intervals ends at frame 2572 for `11_assy_0_1`
(`tighten_nut`, starts at 2552). The video decodes to 2,572 frames, indexed
0 through 2571. The official AR loader samples endpoints inclusively, so the
last annotated endpoint has no corresponding decoded frame. The source label
is retained and flagged. It is outside the nine native PSR evaluation recordings.

## Media checks

All 86 MP4 streams completed ffprobe frame decoding without reported errors.
For each of the nine native recordings, decoded video frame count equals native
RGB image count. Five pixel-alignment spot checks per recording passed at
first, quarter, middle, three-quarter and last frames.

Runtime timestamps come from actual decoded MP4 presentation timestamps,
normalized to the first frame. They describe the rendered timeline; physical
sensor capture-time alignment is not claimed.

## Persistence and recovery checks

Using live Atlas with explicit `source_kind=mock` observations:

- 62 RGB checkpoint packets produced 62 durable decisions.
- Atlas outputs matched SQLite outputs.
- Closing/reopening Atlas and replaying each recording produced zero new decisions.
- A synthetic unresolved issue survived connection close and restart.
- Atlas rejected an attempted stale revision update.

Offline tests additionally cover crash after checkpoint commit but before decision
projection, policy/config changes on resume, neutral/verdict cache reuse,
stateless memory separation, missing/future evidence, current-frame citations,
invalid provider JSON, failed transport without retries, persistent unknown charges,
provider pricing lockout, ZIP traversal and same-size corruption, duplicate verdicts,
paired-cohort promotion, and freezing selection before heldout inference.

## Actual model inputs and outputs when inference is enabled

Input: at most three past/current RGB images, the public CAD component key and
bounded geometry descriptions, numeric frame
IDs, neutral observations, and up to three earlier model-generated summaries.
No hidden state values or trial/file names are sent to the model.

Output: 11 component verdicts per checkpoint, each with one of four outcomes,
a rationale and cited frame IDs; durable memory and a verification/observation
decision; JSONL, JSON and HTML evaluation reports. Accuracy includes abstentions
as misses. The ledger records every dispatched attempt, including failures.

The predeclared comparison uses a fixed model with and without memory. One
development-derived checklist can be validated and frozen before final subject
inference. These native recordings are from official test_p1, so the internal
subject split is exploratory and must not be called the official benchmark.

## Branch integration

Main `c0b489c` already contains Person 3 scoring/promotion and the new package name.
Person 1 `a7f900d` supplies the numeric RGB adapter and VLM inspection.
Person 2 `d1f5eac` supplies the durable store and receipt/outbox runtime.
This integration ports those recording modules into `robologue`, restores promoted
checklist behavior, and adds a budgeted verifier, dataset coordinator, report
generation and stronger evaluation safeguards.
