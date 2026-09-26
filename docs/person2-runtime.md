# Person 2: agent runtime and persistent memory

This contribution extends the actual `recursive-harness` branch at
`4f2afb482bf8881f7bd9412c725d732796db2570`. It supports both existing tracks:
the grid-world recursive harness and CookMemory's recording replay. They
retain separate storage formats and APIs. No JEV, JEPA or Hermes package is
required by this contribution. The runnable proposer remains deterministic.

## Install and choose storage

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[atlas]'
cp .env.example .env
# Fill MONGODB_URI using the hackathon Sandbox credentials.
```

A shell export in a separate terminal does not propagate to an already
running Codex process; an explicit `--env-file .env` works. The loader accepts
KEY=VALUE with optional quotes/export, performs no interpolation and preserves
exported variables. Never put keys in the command line or commit .env.
The team's `MongoDB_Connection_String` key is accepted as an alias when
`MONGODB_URI` is empty.

- `--backend atlas` requires a URI and a successful ping.
- `--backend auto` chooses Atlas when a URI is set; connection failures raise
  a sanitized error. A configured Atlas run never silently becomes SQLite.
- `--backend local` explicitly uses SQLite even if a URI is set.
- PyMongo connections have finite connect, selection and socket timeouts,
  retryable writes and majority write acknowledgement.
- No frontier-model key is required for the deterministic runtime.

## Recursive harness: durable runs

```bash
# Commit two episodes and exit.
.venv/bin/python -m loop --backend local --run-id team-demo \
  --tasks pick-and-deliver,multi-room-deliver --seeds 7,8 \
  --generations 3 --stop-after-episodes 2

# A new process resumes to a TOTAL of three generations.
.venv/bin/python demo.py --backend local --run-id team-demo --resume

# Atlas uses the same runtime.
.venv/bin/python demo.py --backend atlas --env-file .env --run-id atlas-demo

# Read the architecture and reports from that run.
.venv/bin/python -m eval.generalization --backend local --run-id team-demo
.venv/bin/python -m viz.evolution --backend local --run-id team-demo \
  --out work/evolution.html
```

Use the same backend, database, tasks, seeds, provider, limits and deployed
code when resuming. A new run ID starts a new namespace without clearing
other people's data. Include `--env-file .env` and `--backend atlas` on each
Atlas command, including generalization and visualization.

`runtime.execute_run(backend, tasks, seeds, generations, run_id=...)`
is the Python integration entry point. It accepts a `RunBackend` or raw backend,
an optional provider and `RuntimeLimits`. `loop.run_episode` and legacy
`loop.run_generations` remain available for the original benchmark code;
the latter does not offer managed restart recovery. Use the managed runtime
for the shared demo.

Product orchestration can call `runtime.get_run_status(backend, run_id)` or
`python -m loop --backend atlas --env-file .env --run-id atlas-demo --status`.
It reads completion/progress, pending-write state, record counts and quotas
without advancing the run. While a worker is active, projection counts can
temporarily lag the authoritative receipt count.

### Storage and recovery

Every managed storage `_id` begins with `run:<run_id>:`. Application-facing
IDs remain logical IDs, preserving the existing frozen Trace/Architecture
contracts. `RunBackend` adds this scope to reads, writes and clears.
Spatial memory is keyed by architecture version and environment; only
blocked-cell corrections persist. Visited cells start fresh each episode.

A checkpoint at `policies/_id=runtime` contains a revision, episode cursor,
active version, receipts, spatial memory, generation history, quota ledger
and a pending write. It is authoritative:

1. Compute an episode without development database writes.
2. Atomically commit its receipt, new memory, next cursor and pending trace.
3. Project the pending documents with fixed identities.
4. Derive skill counters from the committed receipts and clear pending.

A crash after step 2 replays the projections without recomputing the episode
or increasing its counters twice. A crash before step 2 can recompute the
current episode. The same sequence publishes accepted architecture versions,
lessons, mutation audit and generation reports. Model reservations are
committed before dispatch; an uncertain interrupted request stays charged.

One worker per run/session is supported. Compare-and-swap detects stale
checkpoint writers; there is no distributed worker lease. Never run two
workers for one ID. The projection set is eventually consistent while
pending writes exist; resume flushes it before proceeding. This is not a
multi-document transaction.

Immutable architecture insertion rejects an existing version with different
behavior. Replaying an identical publication retains the original timestamp.
Resume checks the frozen run configuration and can extend the target total
generation count, up to the overall receipt bound.

### Runtime limits

Defaults: 600 steps/episode, 10 proposal attempts, 10 model calls, 100,000
reserved model tokens, $1 reserved cost and 300 seconds per invocation.
A run supports at most 100 generations and 1,000 development receipts.
Checkpoint size is capped at 8 MB. CLI flags expose the major limits.

Per-call defaults: 16 KB input context, 1,024 output tokens, $0.10 worst-case
cost. These are conservative reservations, not measured billing. Costs round
up to microdollars, remaining total cost rounds down. Quotas survive restart.
An exhausted run quota requires a new bounded run; invocation wall time
resets on resume. Time checks happen between bounded episodes and stages.
A network provider must enforce its supplied request timeout itself.

### Person 3 model interface

Implement:

```python
class CandidateProvider:
    name = "chosen-model:prompt-v1"  # include config version; never a key
    is_model = True

    def propose(self, context, permit):
        # Honor timeout_seconds, max_output_tokens, max_input_tokens,
        # and max_cost_usd. Include all prompt overhead in your bound.
        return ProposalResult(mutation=data, metadata={
            "model": "chosen-model", "prompt_version": "v1",
            "input_tokens": actual_in, "output_tokens": actual_out,
            "cost_usd": actual_cost,
        })
```

Pass the instance to `execute_run(..., provider=CandidateProvider())`.
There is no model CLI/plugin loader yet. The adapter must account for its
system prompt and pricing inside the permit; never dispatch if the total
request cannot fit. The runtime reserves context bytes as a conservative
input-token bound. It cannot independently establish a third party's pricing
or forcibly cancel an arbitrary Python provider.

Context includes development trace summaries, the current architecture,
registered skill data and validated lesson references. It excludes evaluator
labels, world truth, credentials and database handles. Lesson text is advice
for the proposer. The robot's executable behavior comes from accepted modules,
verifiers and policies; it does not follow free-text lesson instructions.

Model exceptions fall back to the deterministic provider with the reservation
still consumed. Malformed/unsupported proposals are rejected and audited.
Every candidate cites available development trace IDs and passes a bounded
mutation gate and held-out parent/child validation. Supported changes:
collision verifier, spatial-memory module (including careful-navigation),
removing spatial memory, max_steps 1–600, retry attempts 0–10.
Other frozen mutation names remain reserved until executable hooks exist.
Generated code is never evaluated.

Validation and generalization start with fresh per-environment memory.
Validation traces are audit records; these runs cannot update development
lessons, skill counters or saved spatial memory. Fixed validation seeds are
already present in the branch; they are not a secret final evaluation split.

## CookMemory: recording observations and decisions

The existing recording replay now has revision checks, a recoverable pending
decision and idempotent stored decision projections. Existing schema-1
observations still work. Optional `schema_version: 1` and `evidence_refs`
provide a Person 1 handoff:

```json
{
  "schema_version": 1,
  "event_id": "recording-A:sample-12",
  "recording_id": "recording-A",
  "timestamp": 12.0,
  "observation": "Bowl moved; utensil cleaning was not visible.",
  "completed_steps": ["move-bowl"],
  "issue": {"id": "utensil-check", "description": "Need cleaning evidence"},
  "evidence_refs": [{
    "recording_id": "recording-A",
    "start_seconds": 10.0,
    "end_seconds": 12.0,
    "frame_ids": ["recording-A:frame-360"]
  }]
}
```

Evidence intervals must belong to the same recording and end by the event
timestamp; future evidence is rejected. `frame_ids` is optional. References
remain in unresolved issues, recent memory and emitted decisions. Observation
objects reject unknown fields, including evaluator labels. Maximum event size:
64 KB; at most 64 references/event and 32 frame IDs/reference.

```bash
.venv/bin/python -m cookmemory.cli replay examples/observations.jsonl \
  --session recording-demo --limit 2 --decisions-out work/decisions.jsonl
.venv/bin/python -m cookmemory.cli replay examples/observations.jsonl \
  --session recording-demo --decisions-out work/decisions.jsonl
```

`--limit` counts new events; replay emits console skip receipts for previously
processed inputs. `--decisions-out` exports all durable decisions for the
session, including earlier passes, ready for the existing replay renderer.
Atlas adds `--backend atlas --env-file .env`. Separate sessions are required
per recording, memory/stateless mode and policy configuration. Policy content
is hashed so changing rules under the same version cannot silently change a
resumed session. Limits default to 10,000 unique events/session, 300 seconds
per invocation and an 8 MB checkpoint.

Legacy sessions remain readable; decisions committed before this contribution
were not separately stored and cannot be reconstructed from their hashes.
Start a new session for a complete decision export. For a new interrupted
session, replay first flushes any pending decision before inspecting inputs.
Event processing and receipt are atomic; projection and JSONL export follow.
Atlas decision records live in `recording_decisions`; replay checkpoints
continue to live in `sessions`.

## Four-person handoff

| Owner | Integration |
| --- | --- |
| 1 — Data/perception | Produce label-free schema-1 observations with recording/interval/frame references. Keep evaluation annotations in a separate file. |
| 2 — Agent/memory | Own runtime, backend selection, scoped memory, checkpoints, budgets and decision export. |
| 3 — Evaluation/adaptation | Implement the bounded provider above, stronger validation splits, real perception evaluations and proposed policies. |
| 4 — Product/integration | Consume full durable decision JSONL, render the existing replay UI, pass run/session IDs through orchestration and show provenance. |

The recording track and grid-world track are independently runnable. Connecting
real video tools to a frontier-model inspection loop is further integration
work with Persons 1 and 3; this contribution supplies memory and execution
boundaries without claiming the simulation operates on recordings.

## Demonstration evidence and limits

Executed locally in separate processes: pause the simulation after two episodes,
resume to three generations, replay the recording fixture across two processes,
export full decisions, run frozen generalization and generate the timeline.
Observed simulation success: v0 50%, v1 100%, v2 100%; maze generalization 100%.
The fixture is synthetic and these figures describe the simulation only.
No automated test suite was added or run for this contribution.

Live Atlas execution completed on September 26, 2026, using the user's Sandbox.
Run `person2-atlas-20260926` paused after two episodes and resumed in another
process to three generations: 12 committed receipts, 12 development traces,
12 experiences, architectures v0/v1/v2, two mutations, navigation counters
10 successes/two failures and no pending writes. No model calls were made.
Recording session `person2-recording-atlas-20260926` committed two events,
then reconnected, skipped those two and exported all four durable decisions.
These records remain in their own run/session namespaces for the team demo.

MongoDB's [single-document atomicity guidance](https://www.mongodb.com/docs/manual/core/write-operations-atomicity/)
supports filtering an update by the expected current value; the runtime uses
revision filters for checkpoint compare-and-swap. Multiple projected records
use durable intents and idempotent identities.
