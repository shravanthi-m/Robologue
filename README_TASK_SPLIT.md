# Robologue: Four-Person Task Split

**Build window:** 5-6 hours. **Dataset:** CaptainCook4D. **Start with:** one recipe,
RGB recordings, three verification tools, persistent task memory, and one evaluated
policy revision. Pose summaries are optional until the files are validated.

> We are building an agent that learns what evidence to inspect before declaring a
> physical procedure step successful. It remembers unresolved issues, checks earlier
> evidence, and improves its verification policy while the underlying models stay fixed.

Read [the architecture](README.md#target-architecture-budgeted-verification-with-persistent-memory)
for the design and [the observation contract](README.md#observation-contract) for the evidence format.

## Ownership: who does what?

| Person | Owner | Core question | Main deliverable |
| --- | --- | --- | --- |
| **1** | **Shrav - data and perception** | What evidence is available, and what does it show? | Working video inspection tool with timestamps and uncertainty |
| **2** | **Agent and Atlas memory** | What should the agent inspect, remember, and conclude? | Budgeted verification agent with restart-safe memory |
| **3** | **Evaluation and policy improvement** | Was the decision right, and did the policy improve? | Hidden-label evaluator and tested policy revision |
| **4** | **Product, integration and submission** | Can we run and clearly show the complete system? | Replay interface, integrated run, and one-minute demo |

Assign names to Persons 2-4 at kickoff. No one needs to wait for the real video tool:
use explicit synthetic fixtures until integration. Person 4 coordinates integration;
each owner remains responsible for their module.

## Current starting point

Already implemented:

- Local/Atlas checkpoint stores, deterministic replay, and a stateless fixture baseline.
- CaptainCook4D annotation conversion and an IndustReal PSR label adapter, both evaluator-only.
- Hidden-label four-outcome scorer and a tested single-rule policy promotion gate.
- Tests covering persistence, restart recovery, event ordering, label-field separation,
  and the evaluator/policy gates. Person 1's video inspection tooling is still to be
  built; use the synthetic observation fixture until it lands.

Still required:

- Download and validate real recordings; verify one live VLM inspection.
- Model-driven tool orchestration, run-level budgets, evidence-based verdicts and memory updates.
- Experience retrieval, scoring, automatic rule proposal/selection, and the interface.

The existing deterministic fixture is not a trained model or a real-data result.

## Phase 0 - 0:00-0:30: shared kickoff and contracts

Everyone agrees on these decisions before expanding implementation:

- [ ] One task family, with both normal and error examples.
- [ ] Person 3 assigns whole recordings to development, validation, and final test.
- [ ] One integration recording from development; no test-label debugging.
- [ ] Fixed VLM and verifier model IDs; explicit prompt/policy versions.
- [ ] Pre-segmented step verification for the MVP, with boundaries supplied without error labels.
- [ ] Three tools: `inspect_video`, `check_prerequisite`, `search_experience`.
- [ ] Initial budget, for example two video inspections and five total calls per decision.
- [ ] Error categories supported by the selected data; do not promise a category before checking examples.
- [ ] Freeze the shared request/result/verdict contracts below.

**Done when:** each person can point to an input fixture, expected output shape, and
module they own. If there are too few usable recordings for separate splits, reduce
the evaluation claim rather than split adjacent frames across development and test.

## Person 1 - Shrav: data and perception

### First independent tasks - 0:30-1:30

- [ ] Obtain one selected recording; retain its actual recording ID and camera/view.
- [ ] Register it with the existing perception CLI; inspect duration and metadata.
- [ ] Sample three intervals and check the frames against video playback.
- [ ] Verify time origin and any offset relative to the procedure timeline.
- [ ] Configure `OPENROUTER_API_KEY` privately and an explicit image-capable JSON-output model.
- [ ] Run one real `inspect_video` request and review its observations and uncertainty.
- [ ] Record the manifest path, inspection ID, prompt/model versions, and sample evidence paths.

### Integration - 1:30-2:30

- [ ] Give Person 2 the Python callable and a real result; replace their fake inspection tool.
- [ ] Give Person 4 frame paths and timestamps for the evidence display.
- [ ] Tell Person 3 the precise time origin and recording identity for annotation matching.
- [ ] Handle unreadable frames and empty intervals explicitly; do not fill gaps with invented evidence.

### Baseline and improvement runs - 2:30-4:30

- [ ] Make the selected recordings available and keep perception configuration fixed.
- [ ] Reuse cached results for identical inspection requests; new intervals still require inspection.
- [ ] If time permits, inspect one spatial file for schema, units, synchronization and tracking validity.
- [ ] Add only a justified motion feature; otherwise keep motion marked unavailable.
- [ ] Do not change perception prompts during a baseline-versus-policy comparison.

### Final handoff

- [ ] Media and evidence are available for the demo; timestamps and attribution are checked.
- [ ] Person 2 can inspect a permitted interval without asking you to operate the tool.
- [ ] No error labels or annotation-derived answers enter the perception path.

**Own:** `examples/observations.jsonl` and the observation JSONL contract; perception
adapters (`robologue/perception/`, `tests/test_perception.py`, `docs/PERSON1.md`) when built.
**Do not own:** final verdict logic, evaluation labels, learned rules, or the UI.
**Acceptance:** real video interval -> evidence-backed observations with uncertainty,
no future frames, provenance, and a reproducible cache.

## Person 2 - verification agent and Atlas memory

### First independent tasks - 0:30-1:30

- [ ] Connect to the provided hackathon Atlas Sandbox using private environment variables.
- [ ] Build a tool-calling verifier using fake inspection/experience results first.
- [ ] Assemble compact step context: instruction, prerequisites, known state, open issues,
  active policy and remaining budget.
- [ ] Implement the verdict schema below and validate model outputs before saving.
- [ ] Enforce budgets in code, not only in the prompt; bound retries and charge attempted calls.
- [ ] Keep step instructions and prerequisite structure separate from observed completion.

### Integration - 1:30-2:30

- [ ] Replace fake video inspection with Person 1's callable.
- [ ] Supply `observed_until` from the trusted run cursor, not model-selected arguments.
- [ ] Implement `check_prerequisite` with verified/unverified/contradicted status and evidence IDs.
- [ ] Store observations, verdicts, outstanding issues and checkpoint cursor in Atlas.
- [ ] Emit run events to Person 4 and prediction records to Person 3.

### Baseline and improvement runs - 2:30-4:30

- [ ] Implement `search_experience` over eligible development cases.
- [ ] Filter out current-recording, validation and test cases before semantic retrieval.
- [ ] If Vector Search setup blocks integration, use explicit filtered lookup as a labeled fallback.
- [ ] Resume after a process restart without clearing issues or reapplying completed events.
- [ ] Pin a policy version per run and apply Person 3's accepted revision only between runs.
- [ ] Validate evidence IDs; do not resolve issues merely because a new frame appears normal.

### Final handoff

- [ ] One reproducible command/function starts a real run with explicit recording, session and policy.
- [ ] Logs include model versions, tool calls, costs, evidence and decisions.
- [ ] Budget exhaustion supports `insufficient_evidence` rather than forced guessing.

**Own:** `robologue/harness.py`, `robologue/store.py`, proposed `robologue/agent.py`
and `robologue/tools.py`, plus corresponding tests. Coordinate shared CLI changes.
**Acceptance:** real tool-use loop with Atlas persistence, bounded calls, structured
verdicts and restart recovery. The existing deterministic replay is only scaffolding.

## Person 3 - evaluation and policy improvement

### First independent tasks - 0:30-1:30

- [ ] Inspect official annotations and choose usable recordings with Person 1.
- [ ] Assign whole recordings to development/validation/final test; record provenance.
- [ ] Define stable decision IDs and how step predictions map to annotation intervals.
- [ ] Define treatment of repeated steps, missing steps with `-1` timestamps, and abstentions.
- [ ] Implement scoring against dummy predictions before the agent exists.
- [ ] Predeclare metrics and promotion criteria: missed errors, false alarms, abstentions,
  prediction coverage and logical tool cost. Report counts as well as rates.

### Integration - 1:30-2:30

- [ ] Consume Person 2's verdicts without exposing labels to their tools or context.
- [ ] Score one development recording and confirm annotation/time matching with Person 1.
- [ ] Give Person 4 an evaluation report shape for its separate results view.
- [ ] Distinguish perception failures, tool-selection failures, memory failures and ambiguous labels.

### Baseline and improvement runs - 2:30-4:30

- [ ] Run the fixed-policy baseline under frozen models/budgets.
- [ ] Use development misses to propose one scoped rule automatically.
- [ ] Save a candidate policy with ID, trigger, action, rationale and supporting development cases.
- [ ] Compare current and candidate policies on validation recordings with equal budgets
  and fixed eligible experience; video queries may differ, but the evidence tools stay fixed.
- [ ] Promote only if predeclared criteria pass; otherwise retain the baseline and record rejection.
- [ ] Freeze the selected policy and experience eligibility before final test.

### Final handoff

- [ ] Run untouched final-test recordings and report actual results, including failures to improve.
- [ ] Keep labels and evaluation outcomes inaccessible to agent tools.
- [ ] Explain sample-size limitations and that recorded footage cannot prove physical recovery.

**Own:** `robologue/captaincook.py`, `robologue/industreal_labels.py`,
`robologue/evaluate.py`, `robologue/policies.py`, and evaluator tests
(`tests/test_evaluation.py`). Coordinate CLI changes.
**Acceptance:** independent scoring, no answer leakage, and an auditable candidate
promotion/rejection. A manually written rule must be labeled manual.

## Person 4 - product, integration and submission

### First independent tasks - 0:30-1:30

- [ ] Build a minimal replay view using synthetic fixture events.
- [ ] Define the run-start interface with Person 2 and a JSONL event stream/file to consume.
- [ ] Show recording/time, current step, tool calls, evidence, memory, verdict and active policy.
- [ ] Keep secrets and provider calls on the backend, not in browser code.
- [ ] Make the frontend useful even if it initially reads a completed run rather than streams live.

### Integration - 1:30-2:30

- [ ] Connect one real run; display Person 1's evidence and Person 2's memory/decisions.
- [ ] Confirm one command/button can reproduce the integration run.
- [ ] Add Person 3's metrics in a separate evaluation view after prediction.
- [ ] Clearly label live, cached, recorded and synthetic content.

### Baseline and improvement runs - 2:30-4:30

- [ ] Display baseline and candidate runs with model/policy/budget metadata.
- [ ] Show the changed rule, which tool choice changed, and supporting evidence.
- [ ] Surface failed calls and insufficient evidence instead of hiding them.
- [ ] Coordinate one restart demonstration with Person 2.
- [ ] Own integration smoke checks and keep all four owners informed of broken contracts.

### Final handoff

- [ ] Record the one-minute demo with working audio/video.
- [ ] Verify public repo and demo accessibility; include all team members in submission.
- [ ] State original contributions and external data/model/library attribution.
- [ ] Include reproducible run instructions and actual evaluation results.

**Own:** proposed `web/`, `scripts/run_demo.*`, integration smoke checks, and submission
materials. Coordinate architecture README edits; avoid editing teammates' core modules.
**Acceptance:** a judge can follow the complete behavior and identify what we built.
The replay supports the agent demo; a dashboard is not the core project.

## Shared contracts: freeze these before implementation

### A. Person 1 -> Person 2: video inspection

Existing callable:

```python
from robologue.perception import inspect_video

result = inspect_video(
    manifest_path="work/recordings.json",
    recording_id="RECORDING_ID",
    t_start=0,
    t_end=10,
    n_frames=3,
    observed_until=10,  # supplied by trusted orchestrator
    model="provider/model-id",
)
```

Result includes `inspection_id`, `recording_id`, `window`, `observations`,
`uncertainties`, `evidence`, `model_version`, `prompt_version`, `provenance`,
`call`, `cache_hit`, and explicit motion availability. See Person 1's guide for
full details. The tool extracts evidence; it does not issue correctness verdicts.

### B. Person 2 -> Persons 3 and 4: target verdict

Agree on a new versioned schema; do not confuse this with the old replay decision:

```json
{
  "schema_version": 1,
  "run_id": "dev-run-v1",
  "decision_id": "recording-A:step-instance-03",
  "recording_id": "recording-A",
  "step_id": "step-03",
  "window": [412, 440],
  "verdict": "insufficient_evidence",
  "error_type": null,
  "reason": "Tool identity is occluded; completion is not verified.",
  "evidence_ids": ["inspection-001"],
  "policy_version": "verification-v1",
  "model_version": "configured-verifier",
  "tool_calls": 2,
  "video_inspections": 1
}
```

Example only. Verdicts: `correct`, `incorrect`, `insufficient_evidence`. Normalize
error types against the selected dataset taxonomy. Use a unique step-instance ID
because the same step can repeat. No hidden label is included in this payload.

### C. Person 3 -> Person 2: target policy

```json
{
  "policy_id": "verification-v2",
  "parent_policy": "verification-v1",
  "status": "candidate",
  "rules": [{
    "id": "verify-prerequisite-v1",
    "scope": "selected-task-and-step",
    "trigger": "Required predecessor has no verified completion",
    "action": "Call check_prerequisite before accepting this step"
  }],
  "source_split": "development",
  "supporting_case_ids": ["dev-case-01"]
}
```

Person 3 owns promotion status and validation results. Person 2 loads only the
chosen version at run start; validation and test labels cannot be retrieved as cases.

### D. Person 2 -> Person 4: run events

Use an agreed JSONL event envelope: `run_id`, `sequence`, `type`, `timestamp`,
`payload`. Event types can be `step_started`, `tool_requested`, `tool_result`,
`memory_updated`, `verdict`, `run_finished`, `run_failed`. Sequence numbers make
replay/order explicit. This is a target integration contract, not already implemented.

## Shared phase gates

| Time | Gate | If it fails |
| --- | --- | --- |
| 0:30 | One recording opens; contracts and splits are agreed | Drop pose work; use fixtures while fixing access |
| 1:30 | Each module works independently with fixtures | Fix interfaces before adding features |
| 2:30 | One real step completes video -> agent -> Atlas -> evaluator -> UI | Everyone prioritizes the broken integration boundary |
| 3:30 | Repeatable baseline and restart demo work | Stop feature expansion; finish the baseline |
| 4:30 | One candidate is validated and adopted or rejected | Report the result honestly; don't tune on final test |
| Final 30-90 min | Freeze, final evaluation, record, verify access, submit | Cut optional UI polish and extra recordings |

For a five-hour deadline, begin the freeze by hour four. Four people should reduce
integration risk, not expand scope to extra datasets, tasks, or model training.

## Coordination rules

- Work directly on `main`. Commit small usable changes. Person 4 coordinates merges
  and an integration check.
- Prefer separate clones/worktrees if sharing a computer. Do not switch the branch
  underneath another person's running work in the same checkout.
- Commit small usable changes. Person 4 coordinates merges and an integration check.
- Announce schema changes before implementation; keep fixtures backward-compatible
  or version the schema and update consumers together.
- Never commit credentials, raw recordings, caches or private annotation exports.
- Keep the provided Atlas Sandbox in the final runtime.
- Freeze the models and evidence configuration when comparing policies.

## Final demo story

1. Show a fixed-policy miss or unresolved prerequisite on real recorded evidence.
2. Show what memory/evidence was missing and the rule proposed from development feedback.
3. Show validation accepting or rejecting the change and the held-out results.
4. Restart the agent and show that outstanding task state survives.
5. Identify the code, data and functionality built during the hackathon.

We claim a persistent, budgeted verification harness evaluated on real recordings.
We do not claim physical robot recovery, proven sim-to-real transfer, superiority
to RL, or measured improvement unless our actual evaluation supports it.
