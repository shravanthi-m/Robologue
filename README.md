# Recursive Harness: a robot that rewrites itself from experience

A robot starts with a stale map, no collision checking, and no memory. It
fails. A deterministic evolver diagnoses the failures, proposes bounded
changes to the robot's own architecture, validates them on held-out seeds,
and promotes only what the numbers justify. Three generations later the
robot routes around obstacles it cannot see and remembers what it learned.
MongoDB is its developmental memory: every experience, failure, lesson,
skill, mutation, and architecture version persists there, and every reported
number traces back to a stored record.

## Person 2 runtime and memory

The demo and loop now use isolated run IDs, durable checkpoints, explicit
Atlas/local selection and bounded proposal providers. CookMemory recording
replay retains evidence references and exports durable decisions across
restarts. See [the implementation and team handoff](docs/person2-runtime.md)
for the storage protocol, budgets, observation contract and model adapter API.

## The demo story

```
v0: 2/4 success, avg 90.8 steps   -> repeated verification_error
v1: 4/4 success, avg 34.5 steps   -> ADD_VERIFIER +collision_check (validated)
v2: 4/4 success, avg 33.8 steps   -> ADD_MODULE +spatial_memory (validated)
then: stable. No patterns, no changes. The harness only mutates on evidence.
```

Frozen v2 then solves `maze-deliver`, a maze it never trained on, 3/3,
using the skills it learned: collision replans and persistent spatial memory.

## Run it

Python 3.10+. No dependencies for local mode (SQLite). For Atlas:

```bash
pip install -e '.[atlas]'    # only if you want the Atlas backend
export MONGODB_URI='...'     # hackathon Sandbox URI
export MONGODB_DATABASE=cookmemory
```

```bash
# The three-minute narrated demo (deterministic, isolated run)
python3 demo.py --backend local --run-id demo-1

# Atlas demo: explicit credentials file and isolated run
python3 demo.py --backend atlas --env-file .env --run-id atlas-demo-1

# The managed evolution loop (use a new ID for each independent run)
python3 -m loop --backend local --run-id run-1 --generations 3 --tasks pick-and-deliver,multi-room-deliver --seeds 7,8

# Reproducible benchmark: 3 independent trials, report saved to evaluations
python3 -m eval.benchmark --trials 3

# Held-out generalization: frozen best architecture on the unseen maze
python3 -m eval.generalization --backend local --run-id run-1 --held-out-task maze-deliver --seeds 21,22,23

# Evolution timeline (static HTML, no server)
python3 -m viz.evolution --backend local --run-id run-1
# open evolution.html

# Tests
python3 -m unittest discover -s tests
```

See `ARCHITECTURE.md` for the loop, the frozen contracts, what each
collection stores, and what is deliberately not claimed.

## How it works (short)

- `sim/` — deterministic grid world with a stale-map premise: the planner's
  map is missing obstacles the true world has. Failures emerge from missing
  capabilities.
- `evolver/` — classifier (failed trace -> one of 9 categories), pattern
  detector, mutation proposer (10 bounded types, registry skills only),
  validator (parent vs child on fixed seeds), and counterfactual lessons
  that graduate from proposed to validated when adopted.
- `skills/` — skill registry with prerequisites, versions, composition into
  strategies, and real success/failure counts from episodes.
- `architectures.py` — immutable versioned harness definitions; mutations are
  data, never generated code.
- `eval/` — reproducible benchmark and frozen-architecture generalization test.
- `db.py` — explicit Atlas/local selection and scoped storage; an Atlas
  connection failure raises an error without silently changing backends.
- `runtime.py`, `budget.py`, `providers.py` — durable runs, quota reservations,
  bounded mutation proposals and the Person 3 model integration boundary.

## What was built when

This repo is a comparison build. The starting point was the
[shravanthi-m/MongoHack](https://github.com/shravanthi-m/MongoHack) CookMemory
starter (public, built during the MongoDB Harness Engineering & Model
Wrangling Hackathon on 2026-09-26); the CookMemory evaluation patch
(`cookmemory/`, committed earlier) is credited work on top of that starter.

The recursive harness (`sim/`, `evolver/`, `skills/`, `architectures.py`,
`contracts.py`, `db.py`, `loop.py`, `eval/`, `viz/`, `demo.py`,
`ARCHITECTURE.md`, and the Prompt 1-3 test suites) was built during the
hackathon as the entry itself: an embodied agent that recursively evolves
its own harness from experience, with MongoDB Atlas as its persistent
developmental memory.

## Contributors

- RosarioM123 - recursive harness design and implementation (sim, evolver,
  skills, evaluation, demo, docs)
- shravanthi-m - author of the original CookMemory starter
  ([shravanthi-m/MongoHack](https://github.com/shravanthi-m/MongoHack)) this
  comparison build starts from
- Varun Sahni ([varunsahni18](https://github.com/varunsahni18)) - original
  CookMemory starter team
- basith-md ([basith-md](https://github.com/basith-md)) - original CookMemory
  starter team

## Honest claims

Deterministic simulation, not a physical robot. The Atlas backend is
implemented and both runtime tracks completed live Sandbox restart demos. The
classifier's three perception-adjacent categories are reserved (documented,
not faked) because the sim localizes perfectly by design. Lesson confidence
is qualitative. No LLM is in the default loop; the mechanism is deterministic rules
over traces, which is what makes it auditable.
