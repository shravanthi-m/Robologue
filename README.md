# Robologue

A hackathon starter for a persistent procedural-memory harness, evaluated on
CaptainCook4D egocentric recordings. The goal is to remember unresolved mistakes
through a long task and improve which evidence an agent checks before advancing.

**Status:** runnable memory plumbing, a synthetic observation replay, optional
MongoDB Atlas persistence, and a CaptainCook4D evaluation-label converter.
There is no video model, trained error detector, adaptive policy optimizer, or
real-data benchmark result yet. The bundled decisions are deterministic.

## Why CaptainCook4D?

It provides real kitchen recordings, procedural annotations, error categories,
and recipe task graphs. Some errors were deliberately induced during collection;
real footage does not imply every mistake occurred naturally. This project tests
procedural memory, not robot motor control or demonstrated sim-to-real transfer.

- [Dataset and video downloader](https://captaincook4d.github.io/captain-cook/)
- [Official annotations](https://github.com/CaptainCook4D/annotations)
- [Annotation schema](https://github.com/CaptainCook4D/annotations/blob/main/ANNOTATIONS.md)
- [Task graphs](https://github.com/CaptainCook4D/annotations/tree/main/task_graphs)

### Sensor data and dataset decision

CaptainCook4D is not video-only. The [official downloader](https://github.com/CaptainCook4D/downloader)
documents a `spatial` stream containing head and hand pose, plus camera pose,
depth, audio, and accelerometer/gyroscope/magnetometer streams. The paper's data
composition table lists left and right wrist poses. These are human/device
measurements, not robot joint commands. Verify an actual spatial file before
promising full finger-joint positions, joint angles, or a specific coordinate layout.

Coverage is incomplete: the downloader notes that some recordings have only
GoPro footage and others lack spatial/IMU data. Synchronized streams are described
as aligned with GoPro; raw streams retain device timestamps. Inspect timestamps,
coordinate frames, missing values, and tracking validity in the selected files.
Device IMU motion is not a direct measure of hand or object motion.

**Recommendation: keep CaptainCook4D for the current procedural-memory MVP.**
Select one recipe with a few complete recordings and check RGB + spatial coverage.
Use RGB plus a small motion summary if the spatial stream is usable. A lack of
tracking is unknown evidence, not an error or proof of no motion. Pose should help
choose when/what to inspect; it cannot independently prove successful execution.

[IndustReal](https://github.com/TimSchoonbeek/IndustReal) is the alternative if
explicit hand-joint tracking and assembly are central. Its README documents
`hands.csv`, gaze and head tracking at 10 FPS, RGB/depth, and assembly error labels.
This is still human tracking, not robot actuation. Prefer the switch only if a
small usable subset can be opened promptly; do not spend the build comparing
entire datasets. The existing checkpoint/replay core is dataset-independent,
while the label converter currently supports CaptainCook4D only.

## Proposed pipeline

```text
video chunks + procedure instructions + optional synchronized pose summary
  -> perception adapter (to build; observations and uncertainty only)
  -> timestamped events
  -> persistent task memory in Atlas
  -> verification decision + evidence references
  -> separate evaluator using hidden error annotations
  -> proposed policy change -> held-out validation -> versioned promotion (to build)
```

Agent memory holds completed steps and unresolved issues. An issue persists until
an observation explicitly resolves it. A normal-looking later frame must not
silently clear the history. Absence of a detected issue is not proof of success.

## Run the synthetic starter

Python 3.10+; local mode needs no dependencies or API keys. Run from this directory:

```bash
python3 -m robologue.cli replay examples/observations.jsonl --session demo-memory --limit 2
# Start a new process and resume; already applied events are skipped.
python3 -m robologue.cli replay examples/observations.jsonl --session demo-memory
# Same observations, but no semantic memory between events.
python3 -m robologue.cli replay examples/observations.jsonl --session demo-stateless --mode stateless
python3 -m unittest discover -s tests -v
```

Expected: memory mode keeps the utensil issue open at events 2 and 3, then clears
it at event 4. Stateless mode forgets the earlier issue at event 2. This is an
explicitly authored plumbing fixture, **not evidence of model improvement**.
The `issue`, `resolves`, and `completed_steps` fields are perception outputs in the
future system; in this fixture they are hand-authored. Resolution claims are
trusted by the starter and need evidence validation in the real system.

Local checkpoints live in `work/memory.sqlite`. To rerun from scratch, choose a
new session name. Use a different session for every recording, mode, and experiment.

## Use the hackathon Atlas Sandbox

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[atlas]'
# Export MONGODB_URI using your provided Sandbox credentials.
export MONGODB_DATABASE=robologue
python -m robologue.cli replay examples/observations.jsonl --session atlas-demo --backend atlas
```

`.env.example` documents variables; this code does not load `.env` automatically.
Never commit credentials. Local mode is only for development; the submitted
project must use the provided hackathon Atlas Sandbox.

The `sessions` collection stores one atomic checkpoint document per session.
One worker per session is supported. Concurrent writers need optimistic locking.
Only five recent observations enter the retained context, but receipt IDs and task
state still grow. Long deployments need separate event storage, compaction, and
MongoDB document-size handling. This is not a billion-token implementation.
No database provisioning, credentials, or live Atlas connection is bundled.

## Prepare real evaluation labels

Download `annotation_json/error_annotations.json` from the official annotation
repository into `data/`, then run:

```bash
python3 -m robologue.cli prepare-labels data/error_annotations.json --output work/evaluation-labels.jsonl
```

The converter follows the published recording/step schema. It preserves repeated
step IDs using unique annotation IDs. Missing steps can have `-1` timestamps;
these become untimed evaluator records, never observations at the start of a video.
The output must not already exist, to avoid accidental overwriting.

**Do not generate observed actions from annotation descriptions.** These can state
the expected instruction rather than what actually happened. In particular,
`errors`, `has_errors`, `is_error`, and `modified_description` are hidden labels.
The replay schema rejects unexpected fields, but cannot detect answers copied into
free-text observations. Keep evaluator files outside the agent's retrieval/tools.

Use fixed chronological video windows initially. Produce observations from video,
then replay the generated JSONL using the same command as above. Do not expose
future frames, error-marked boundaries, or test annotations to the agent. Missing
steps may be scoreable only after a prerequisite deadline or the recording ends.

## Observation contract

One JSON object per line:

```json
{
  "event_id": "recording-window-001",
  "recording_id": "recording-id",
  "timestamp": 10.0,
  "observation": "Description of evidence visible up to this timestamp",
  "completed_steps": [],
  "issue": {"id": "stable-issue-id", "description": "Unresolved concern grounded in observations"},
  "resolves": []
}
```

`issue`, `completed_steps`, and `resolves` are optional. Times must be finite,
nonnegative, and chronological. An event ID cannot be reused with changed content.
Task graphs describe dependencies; do not assume every recipe has a strict linear
order. The starter does not yet parse or enforce the official graphs.

## Four-person plan: five to six hours

Keep one recipe/task family and one end-to-end demo. Four people let us separate
sensor work, harness work, evaluation, and the demo rather than expand the scope.

| Owner | Workstream | Concrete handoff |
| --- | --- | --- |
| Person 1 | Data + perception | Select/download a tiny recording subset; inspect spatial schema and alignment; produce chronological observation JSONL from RGB and optional motion summaries |
| Person 2 | Harness + Atlas | Persist task memory in the provided Sandbox; add evidence-based verification decisions and bounded retrieval; prove restart recovery |
| Person 3 | Evaluation + adaptation | Keep labels isolated; define whole-recording splits; compare fixed-model baselines; propose and validate one bounded policy change if time permits |
| Person 4 | Replay UI + integration + submission | Connect the event/decision stream to a minimal recording replay; show evidence and unresolved issues; own integration checks, README, video, and submission |

### Shared interfaces

- Person 1 produces the existing observation JSONL contract. Summarize optional
  sensor evidence in `observation` initially; do not add unknown schema fields.
- Person 2 consumes observations and returns decisions containing `event_id`,
  `timestamp`, `action`, `open_issues`, and `completed_steps`.
- Person 3 alone handles evaluation labels and produces aggregate metrics. No
  current test label or error description enters perception or agent context.
- Person 4 displays predictions as predictions and clearly labels synthetic
  fixtures. Use sample events while the real-data adapter is being built.

### Schedule and scope gates

| Elapsed | Shared milestone |
| --- | --- |
| 0–30 min | Freeze interfaces and one task; Person 1 opens a real recording and matching sensor/label files; others use the synthetic fixture |
| 30–120 min | Work independently: observation adapter, Atlas harness, evaluator, replay UI |
| 2–3 hours | Integrate one real recording end to end and restart midway |
| 3–4 hours | Evaluate separate recordings with identical perception outputs and model settings |
| 4–5 hours | Add one tested verification-policy change only if the core works; otherwise fix failure cases |
| Final 30–60 min | Freeze features, record the one-minute demo, verify public repo/demo access, and submit |

At 30 minutes, if spatial data cannot be opened/aligned, proceed RGB-first on
CaptainCook4D; do not silently present generated tracking as captured sensor data.
If detailed hand joints are essential and an IndustReal subset is already usable,
make one dataset switch then and freeze it. No model training, robot control,
full-dataset downloads, or second task family in this time budget.

For recursive harnessing, propose one check such as requiring explicit prerequisite
evidence after repeated order-related errors. Evaluate on development recordings,
then freeze the policy before final testing. Never weaken the evaluator to improve
the score. Keep the underlying model fixed when measuring harness improvements.

## Evaluation and honest claims

First compare identical perception outputs with and without memory. If pose is
added, separately compare RGB-only and RGB-plus-pose with the same harness, so
sensor improvements are not attributed to memory. Split by complete
recording (preferably by participant/environment), not frames. Ground-truth labels
can inform development feedback, but final evaluation labels never enter memory.

Report missed errors, false alarms, decision latency, model calls, and whether
issues survive restart. Include normal steps; predicting an error everywhere is
not useful. Evaluate error detection separately from intervention/recovery:
recordings cannot establish what an unexecuted correction would have caused.

The core demo should show persistent task state and changing verification behavior.
An image classifier or basic retrieval UI alone would not satisfy the intended
harness contribution and risks the event's prohibited-project categories.

## Code map

- `robologue/harness.py`: validated replay, idempotency, issue memory, stateless baseline.
- `robologue/store.py`: local SQLite or Atlas single-document checkpoints.
- `robologue/captaincook.py`: evaluator-only annotation conversion.
- `robologue/cli.py`: CLI entry points.
- `examples/observations.jsonl`: original synthetic fixture, not CaptainCook4D data.
- `tests/test_core.py`: restart, leakage-boundary, ordering, and missing-step checks.

## Attribution and original work

CaptainCook4D recordings/annotations are external research inputs; follow their
published terms and cite Peddi et al., *CaptainCook4D: A Dataset for Understanding
Errors in Procedural Activities*, NeurIPS 2024. The project site states Apache 2.0
for the dataset; check terms accompanying each downloaded asset.

All starter application code and the synthetic fixture were created for this
hackathon. No third-party application implementation or dataset is vendored here.
Clearly distinguish external models, libraries, and data from the harness your
team builds during the event. Do not present existing research results as ours.
