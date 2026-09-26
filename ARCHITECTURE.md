# ARCHITECTURE.md

A robot that rewrites its own harness. This document explains the loop, the
constraints that keep it honest, what each database collection stores, how to
reproduce the results, and what is deliberately not claimed.

## The idea in one paragraph

Most agent harnesses are fixed: the robot fails, a human reads the logs and
edits the code. Here the harness is data. Every episode produces a trace; a
deterministic classifier labels failures; a pattern detector finds failures
that repeat; a proposer suggests a bounded architecture change; a validator
tests the change on held-out seeds; only then is a new architecture version
promoted. MongoDB (Atlas in production, SQLite locally) is the robot's
developmental memory: experiences, failures, lessons, skills, mutations, and
architecture versions all persist there, and every reported number traces back
to a stored record.

## The loop

Managed demo/loop entry points use the durable orchestration in
[Person 2 runtime](docs/person2-runtime.md). It scopes every stored ID by run,
checkpoints episodes before projecting records, reserves model quotas before
dispatch, and recovers pending writes on resume. The low-level episode and
legacy generation functions below remain the simulation's execution engine.

Lesson text is available to proposal providers. The current robot executes
accepted architecture modules and policies; free-text lessons do not directly
drive its actions. Validation and generalization use separate in-memory state
and store audit traces without changing development memory or counters.

```
episode -> trace -> classify -> patterns -> propose -> validate -> promote
   ^                                                              |
   |______________________ next generation _______________________|
```

1. **Episode** (`loop.run_episode`): the robot (`sim/robot.py`) plans with BFS
   on a stale assumed map inside a deterministic grid world (`sim/world.py`).
   The true world contains obstacles the map does not show, so failures emerge
   from missing capabilities, not from scripted failure flags.
2. **Classify** (`evolver/classifier.py`): each failed trace gets exactly one
   of nine failure categories, by deterministic rules over the trace.
3. **Patterns** (`evolver/patterns.py`): finds failures that repeat
   (same category twice or more) and knowledge the harness re-learns across
   episodes instead of remembering.
4. **Propose** (`evolver/proposer.py`): maps a pattern to one bounded mutation.
   Mutations reference registry skills by name only. No code generation.
5. **Validate** (`evolver/validator.py`): parent vs child architecture on fixed
   validation seeds. The child must strictly improve success rate, or match it
   with strictly fewer steps. Otherwise the mutation is recorded as rejected.
6. **Promote** (`architectures.py`): accepted mutations become new immutable
   architecture versions (v0 -> v1 -> v2 ...). Versions are never edited.

A parallel track runs **counterfactual learning** (`evolver/counterfactual.py`):
each failure produces a lesson (what went wrong, what would have worked, the
inferred lesson, qualitative confidence). A lesson is `proposed` until the
architecture actually adopts its recommendation, then `validated`. Validated
lessons are available to future proposal providers and cited as supporting
evidence. The current robot executes the promoted architecture's behavior.

## Frozen contracts

These never change at runtime; the evolver cannot invent new ones:

- **Collections** (`contracts.py`): experiences, trajectories, tasks, skills,
  failures, lessons, policies, architectures, mutations, evaluations,
  environments.
- **Mutation types**: ADD_MODULE, REMOVE_MODULE, MODIFY_POLICY,
  MODIFY_CONTEXT_STRATEGY, ADD_TOOL, REMOVE_TOOL, ADD_VERIFIER,
  MODIFY_PLANNER, MODIFY_MEMORY_POLICY, MODIFY_RETRY_POLICY.
- **Failure categories**: perception_error, localization_error,
  planning_error, memory_error, context_error, tool_selection_error,
  execution_error, verification_error, task_decomposition_error.
- **Skills** (`skills/registry.py`): nine named skills with descriptions,
  prerequisites, versions, and real success/failure counts. Skills compose
  into named strategies (data, not code); applying a strategy activates each
  underlying skill in its natural architecture list.

## What each collection stores

- `architectures`: immutable harness versions (modules, verifiers, tools,
  policies, planner, context strategy), each with its parent mutation.
- `mutations`: every proposed change, accepted or rejected, with supporting
  trace ids and the full validation result.
- `trajectories` / `experiences` / `failures`: per-episode records; failures
  carry the classifier's category.
- `lessons`: counterfactual lessons with status proposed/validated.
- `skills`: the registry plus measured success/failure counts from real
  episodes.
- `evaluations`: per-generation summaries, benchmark reports, and
  generalization reports. Every number cites trace or mutation ids.
- `tasks`, `environments`, `policies`: task specs, world configs, policy docs.

## The observed evolution

On the fixed benchmark (tasks pick-and-deliver and multi-room-deliver,
seeds 7 and 8):

- **v0**: 2/4 success, avg 90.8 steps. Repeated `verification_error`:
  collisions with no verification.
- **v1** (ADD_VERIFIER +collision_check): 4/4 success, avg 34.5 steps.
  Validation: parent 0.33 success / 322 steps vs child 1.00 / 98 steps.
  Two counterfactual lessons validated.
- **v2** (ADD_MODULE +spatial_memory): 4/4 success, avg 33.8 steps with isolated validation.
  The harness re-learned the same blocked cells every episode, so it
  started persisting them.
- **Generation 2**: no recurring patterns; the architecture stops changing
  on its own.

On the held-out task `maze-deliver` (a maze layout never used to trigger any
mutation), frozen v2 scores 3/3 with the previously learned skills
transferring (collision replans, persistent spatial memory).

## Reproducing the results

```bash
python3 -m unittest discover -s tests        # 58 tests
python3 demo.py --backend local --run-id demo-1 # narrated v0 -> v1 -> v2
python3 -m eval.benchmark --trials 3         # reproducible benchmark report
python3 -m eval.generalization --backend local --run-id demo-1
python3 -m viz.evolution --backend local --run-id demo-1
```

Set `MONGODB_URI` (and optionally `MONGODB_DATABASE`) to run against the
Atlas Hackathon Sandbox instead of local SQLite. Benchmark trials run on
scratch databases so they stay independent; the report lands in the
configured database.

## What is not claimed

- This is a deterministic grid simulation, not a physical robot and not
  CaptainCook4D. No real-world robotics results are claimed.
- The Atlas backend completed live Sandbox restart demos for the simulation
  and recording tracks; details and stored run IDs are in the Person 2 guide.
- The classifier covers six of nine failure categories with real rules;
  localization_error, context_error, and task_decomposition_error are
  reserved with documented trigger conditions, because the current sim gives
  the robot perfect localization by design.
- Lesson confidence is qualitative (low/medium/high), never a fake
  probability.
- There is no LLM in the loop yet. The "brain" is deterministic rules over
  traces. That is intentional: the claim is about the recursive mechanism,
  and a deterministic mechanism is auditable.
- The held-out maze was designed by the author, not sampled from a task
  distribution. It tests transfer to one unseen environment, not general
  generalization.
