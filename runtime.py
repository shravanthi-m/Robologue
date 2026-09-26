"""Durable run orchestration with idempotent projections and bounded providers.

The checkpoint in policies is authoritative. An episode/mutation intent is
saved there before its derived documents are written; replaying an outbox
after a crash never reruns the completed episode or increments counters twice.
One worker per run is supported. CAS detects stale checkpoint writers.
"""
from __future__ import annotations
import copy
import json
import time
import uuid

import architectures
import db as storage
from budget import BudgetExceeded, RuntimeLimits, empty_ledger
from providers import DeterministicProvider, ProposalResult, proposal_context
from skills import ensure_seed_skills, expand_component

CHECKPOINT_ID = "runtime"
MAX_CHECKPOINT_BYTES = 8_000_000


def _commit(db, state):
    previous = state["revision"]
    state = copy.deepcopy(state)
    state["revision"] += 1
    if len(json.dumps(state).encode()) > MAX_CHECKPOINT_BYTES:
        raise BudgetExceeded("Checkpoint capacity reached; start another bounded run")
    db.compare_and_swap("policies", state, previous)
    return state


def _memory(value=None):
    value = value or {}
    return {"blocked": {tuple(c) for c in value.get("blocked", [])},
            "visited": set()}


def _counts(receipts, name):
    used = []
    for r in receipts:
        active = (name == "navigation"
                  or name in ("pickup", "delivery") and name in r["selected_tools"]
                  or name == "collision_check" and r["replans"] > 0
                  or name == "spatial_memory" and r["spatial_memory"])
        if active:
            used.append(r["success"])
    return sum(used), len(used) - sum(used)


def _project_pending(db, state):
    """Idempotent projection. Every identity/counter is fixed by checkpoint."""
    import loop
    pending = state.get("pending")
    if not pending:
        return
    if pending["kind"] == "episode":
        trace = pending["trace"]
        db.save("trajectories", trace)
        db.save("experiences", {
            "_id": trace["trace_id"], "trace_id": trace["trace_id"],
            "task_id": trace["task_id"], "environment_id": trace["environment_id"],
            "architecture_version": trace["harness_version"],
            "success": trace["success"], "failure_type": trace["failure_category"],
            "steps": trace["metrics"]["steps"], "created_at": trace["created_at"]})
        if not trace["success"]:
            db.save("failures", {"_id": trace["trace_id"], "trace_id": trace["trace_id"],
                    "failure_category": trace["failure_category"], "task_id": trace["task_id"],
                    "created_at": trace["created_at"]})
        if pending["lesson"]:
            db.save("lessons", pending["lesson"])
        if pending["spatial_memory"]:
            arch = architectures.get_version(db, trace["harness_version"])
            loop.save_memory(db, arch, trace["environment_id"], _memory(pending["memory"]))
        ensure_seed_skills(db)
        for name in ("navigation", "pickup", "delivery", "collision_check", "spatial_memory"):
            doc = db.find_one("skills", {"name": name, "type": "skill"})
            doc["success_count"], doc["failure_count"] = _counts(state["receipts"], name)
            db.save("skills", doc)
    elif pending["kind"] == "generation":
        mutation = pending["mutation"]
        if mutation:
            if mutation["accepted"]:
                architectures.create_version(db, pending["parent"], mutation)
                for doc in pending["lesson_updates"]:
                    db.save("lessons", doc)
                for doc in pending["trace_updates"]:
                    db.save("trajectories", doc)
            db.save("mutations", mutation)
        db.save("evaluations", dict(pending["summary"],
                                     _id=f"generation:{pending['summary']['generation']}"))
    elif pending["kind"] != "reservation":
        raise RuntimeError("Unknown checkpoint projection")


def _flush(db, state):
    if state.get("pending"):
        _project_pending(db, state)
        state = dict(state, pending=None)
        state = _commit(db, state)
    return state


def _deadline(started, limits):
    remaining = limits.max_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise BudgetExceeded("Invocation wall-time budget exhausted; resume the run")
    return remaining


def _runner(limits, started):
    import loop
    from sim import get_task
    def run(db, arch, task, seed, **kwargs):
        _deadline(started, limits)
        bounded = copy.deepcopy(arch)
        requested = bounded["policies"].get("max_steps") or get_task(task).max_steps
        bounded["policies"]["max_steps"] = min(requested, limits.max_episode_steps)
        return loop.run_episode(db, bounded, task, seed, **kwargs)
    return run


def _propose(db, state, arch, traces, provider, limits, started):
    from evolver.patterns import find_patterns
    for pattern in find_patterns(traces):
        _deadline(started, limits)
        ledger = copy.deepcopy(state["ledger"])
        if ledger["proposals"] >= limits.max_proposals:
            raise BudgetExceeded("Proposal budget exhausted")
        ledger["proposals"] += 1
        task_ids = {t["task_id"] for t in traces}
        lessons = [d for d in db.find("lessons", {"status": "validated"}, limit=None)
                   if d["task_id"] in task_ids][-10:]
        context = proposal_context(pattern, arch, traces, lessons)
        permit = {"timeout_seconds": _deadline(started, limits)}
        if provider.is_model:
            permit = limits.reserve_model(ledger, len(json.dumps(context).encode()), permit["timeout_seconds"])
        state = _commit(db, dict(state, ledger=ledger, pending={"kind": "reservation"}))
        state = _flush(db, state)
        call_started = time.monotonic()
        fallback = False
        try:
            result = provider.propose(copy.deepcopy(context), dict(permit))
            if not isinstance(result, ProposalResult):
                raise ValueError("Provider must return ProposalResult")
        except Exception:
            if not provider.is_model:
                raise
            # The reservation remains consumed even on timeout/unknown charge.
            fallback = True
            result = DeterministicProvider().propose(context, permit)
        _deadline(started, limits)
        if result.mutation is None:
            continue
        # Malformed model output must become an audited rejection, never
        # an exception before the mutation gate or an arbitrary DB document.
        malformed = not isinstance(result.mutation, dict) or not isinstance(result.metadata, dict)
        proposed = result.mutation if isinstance(result.mutation, dict) else {}
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        evidence = {t["trace_id"] for t in traces}
        supporting = proposed.get("supporting_trace_ids", pattern.get("trace_ids", []))
        meta = {"name": provider.name, "fallback": "deterministic" if fallback else None,
                "latency_ms": round((time.monotonic()-call_started)*1000, 2),
                **{k: metadata[k] for k in ("model", "prompt_version", "input_tokens",
                                                  "output_tokens", "cost_usd")
                   if k in metadata}}
        mutation = {
            "mutation_id": uuid.uuid5(uuid.NAMESPACE_URL,
                f"{db.run_id}:mutation:{state['generation']}:{ledger['proposals']}").hex,
            "mutation_type": proposed.get("mutation_type"), "target": proposed.get("target", ""),
            "reason": proposed.get("reason", ""), "expected_effect": proposed.get("expected_effect", ""),
            "new_component": proposed.get("new_component"), "parent_version": arch["version_id"],
            "resulting_version": "", "supporting_trace_ids": supporting,
            "validation": {"provider": meta}, "accepted": False,
            "created_at": __import__("loop").utcnow()}
        if "changes" in proposed:
            mutation["changes"] = proposed["changes"]
        try:
            if malformed:
                raise ValueError("Provider returned malformed proposal data")
            if len(json.dumps(storage.jsonable(proposed)).encode()) > limits.max_input_bytes:
                raise ValueError("Proposal document exceeds schema limits")
            if not isinstance(supporting, list) or not supporting or not set(supporting) <= evidence:
                raise ValueError("Proposal references unavailable development traces")
            for key in ("reason", "target", "expected_effect"):
                if not isinstance(mutation[key], str) or len(mutation[key]) > 2048:
                    raise ValueError("Proposal text exceeds schema limits")
            if provider.is_model:
                for key in ("model", "prompt_version"):
                    if key in metadata and (not isinstance(metadata[key], str) or len(metadata[key]) > 128):
                        raise ValueError("Provider attribution exceeds schema limits")
                for key, maximum in (("input_tokens", permit["max_input_tokens"]),
                                     ("output_tokens", permit["max_output_tokens"]),
                                     ("cost_usd", permit["max_cost_usd"])):
                    usage = metadata.get(key, 0)
                    if isinstance(usage, bool) or not isinstance(usage, (int, float)) or not 0 <= usage <= maximum:
                        raise ValueError("Provider usage exceeds its reservation")
            architectures.validate_mutation(arch, mutation)
        except (ValueError, TypeError):
            # Keep a bounded audit record even for unstoreable model output.
            for key in ("mutation_type", "reason", "target", "expected_effect"):
                value = mutation[key]
                mutation[key] = value[:2048] if isinstance(value, str) else "<invalid>"
            component = mutation["new_component"]
            mutation["new_component"] = component[:128] if isinstance(component, str) else None
            mutation.pop("changes", None)
            mutation["supporting_trace_ids"] = [s for s in supporting
                if isinstance(s, str) and s in evidence][:1000] if isinstance(supporting, list) else []
            mutation["validation"]["provider"] = {
                "name": provider.name, "fallback": "deterministic" if fallback else None,
                "latency_ms": meta["latency_ms"]}
            mutation["validation"]["rejection"] = "invalid_or_unsupported_proposal"
            return state, mutation
        from evolver.validator import validate
        child = architectures.apply_mutation(arch, mutation)
        checked = validate(db, arch, child, _runner(limits, started))
        mutation["validation"].update(checked)
        mutation["accepted"] = checked["passed"]
        if mutation["accepted"]:
            mutation["resulting_version"] = child["version_id"]
        return state, mutation
    return state, None


def _lesson_updates(db, mutation):
    if not mutation or not mutation["accepted"] or not mutation["new_component"]:
        return []
    expanded = expand_component(mutation["new_component"])
    components = set(expanded["verifiers"] + expanded["modules"] + expanded["tools"])
    docs = []
    for tid in mutation["supporting_trace_ids"]:
        for lesson in db.find("lessons", {"trace_id": tid, "status": "proposed"}):
            if lesson.get("recommended_component") in components:
                docs.append(dict(lesson, status="validated", validated_by=mutation["mutation_id"]))
    return docs


def execute_run(backend, tasks, seeds, generations=3, *, run_id=None,
                resume=False, limits=None, provider=None, stop_after_episodes=None):
    """Run to a TOTAL generation count, resuming completed episode receipts.

    A crash during the uncommitted computation may rerun that episode. Once
    the checkpoint contains its pending receipt it is projected without any
    computation on resume. Model reservations also survive uncertain calls.
    """
    import loop
    from evolver.counterfactual import analyze_failure
    from sim import get_task
    limits, provider = limits or RuntimeLimits(), provider or DeterministicProvider()
    tasks, seeds = list(tasks), list(seeds)
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError("Supply unique, nonempty tasks and seeds")
    if isinstance(generations, bool) or not isinstance(generations, int) or not 1 <= generations <= 100:
        raise ValueError("generations must be 1-100")
    if generations * len(tasks) * len(seeds) > 1000:
        raise ValueError("Run exceeds 1000 episode receipts; use separate runs")
    if any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("Task IDs must be nonempty strings")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Seeds must be integers")
    if not isinstance(provider.name, str) or not 1 <= len(provider.name) <= 128 or not isinstance(provider.is_model, bool):
        raise ValueError("Provider requires a bounded name and boolean is_model")
    if stop_after_episodes is not None and (isinstance(stop_after_episodes, bool)
            or not isinstance(stop_after_episodes, int) or stop_after_episodes < 1):
        raise ValueError("stop_after_episodes must be positive")
    for task in tasks:
        get_task(task)
    if resume and not run_id and not isinstance(backend, storage.RunBackend):
        raise ValueError("Resume requires an explicit run_id")
    db = backend if isinstance(backend, storage.RunBackend) else storage.RunBackend(
        backend, run_id or uuid.uuid4().hex)
    if run_id is not None and run_id != db.run_id:
        raise ValueError("Backend and requested run IDs differ")
    state = db.find_one("policies", {"_id": CHECKPOINT_ID})
    if resume:
        if state is None:
            raise ValueError("No checkpoint for this run")
        if (state["tasks"], state["seeds"], state["limits"], state["provider"]) != (
                tasks, seeds, limits.to_dict(), provider.name):
            raise ValueError("Resume must preserve tasks, seeds, limits and provider")
        if generations < state["generation"]:
            raise ValueError("Cannot resume to a generation count already exceeded")
    else:
        if state is not None:
            raise storage.CheckpointConflict("Run already exists; use --resume or another --run-id")
        state = {"_id": CHECKPOINT_ID, "revision": 0, "tasks": tasks, "seeds": seeds,
                 "target_generations": generations,
                 "limits": limits.to_dict(), "provider": provider.name,
                 "version_id": "v0", "generation": 0, "cursor": 0, "phase": "episodes",
                 "history": [], "receipts": [], "memory": {}, "ledger": empty_ledger(),
                 "pending": None}
        db.compare_and_swap("policies", state, None)
    started, completed = time.monotonic(), 0
    # Always finish existing projections before reading an active architecture.
    state = _flush(db, state)
    if state.get("target_generations") != generations:
        state = _commit(db, dict(state, target_generations=generations))
    architectures.ensure_v0(db)
    ensure_seed_skills(db)
    runner = _runner(limits, started)
    slots = [(task, seed) for task in tasks for seed in seeds]
    print(f"[runtime] backend={db.backend_name} phase={state['phase']} "
          f"generation={state['generation']} version={state['version_id']}")
    while state["generation"] < generations:
        _deadline(started, limits)
        arch = architectures.get_version(db, state["version_id"])
        if arch is None:
            raise RuntimeError("Checkpoint references a missing architecture")
        if state["phase"] == "episodes" and state["cursor"] < len(slots):
            task, seed = slots[state["cursor"]]
            spec = get_task(task)
            key = f"{arch['version_id']}:{spec.environment_id}"
            memory = _memory(state["memory"].get(key)) if "spatial_memory" in arch["active_modules"] else _memory()
            previous = [db.find_one("trajectories", {"trace_id": r["trace_id"]})
                        for r in state["receipts"] if r["task_id"] == task]
            trace, memory = runner(db, arch, task, seed, memory=memory,
                                   persist=False, tag="development", history=previous)
            tid = uuid.uuid5(uuid.NAMESPACE_URL,
                f"{db.run_id}:episode:{state['generation']}:{state['cursor']}").hex
            trace["trace_id"] = tid
            lessons = db.find("lessons", {"task_id": task, "status": "validated"}, limit=10)
            trace["context"].update(run_id=db.run_id,
                lessons_available=[d["inferred_lesson"] for d in lessons],
                lesson_usage="proposal_context_only",
                lesson_ids=[d["lesson_id"] for d in lessons])
            lesson = analyze_failure(trace, arch)
            if lesson:
                lesson["lesson_id"] = uuid.uuid5(uuid.NAMESPACE_URL, f"{tid}:lesson").hex
            # Visited cells belong to this episode; map corrections survive.
            saved_memory = storage.jsonable({"blocked": memory["blocked"]})
            receipt = {"trace_id": tid, "generation": state["generation"],
                       "task_id": task, "success": trace["success"],
                       "selected_tools": trace["selected_tools"],
                       "replans": trace["metrics"]["replans"],
                       "spatial_memory": "spatial_memory" in arch["active_modules"]}
            memories = dict(state["memory"])
            if receipt["spatial_memory"]:
                memories[key] = saved_memory
            state = _commit(db, dict(state, cursor=state["cursor"]+1,
                receipts=state["receipts"]+[receipt], memory=memories,
                pending={"kind": "episode", "trace": trace, "lesson": lesson,
                         "spatial_memory": receipt["spatial_memory"], "memory": saved_memory}))
            state = _flush(db, state)
            completed += 1
            if stop_after_episodes is not None and completed >= stop_after_episodes:
                print(f"[runtime] paused after {completed} committed episodes; resume run={db.run_id}")
                return state["history"]
            continue
        if state["phase"] == "episodes":
            state = _commit(db, dict(state, phase="evolve"))
        traces = [db.find_one("trajectories", {"trace_id": r["trace_id"]})
                  for r in state["receipts"] if r["generation"] == state["generation"]]
        state, mutation = _propose(db, state, arch, traces, provider, limits, started)
        lessons = _lesson_updates(db, mutation)
        updates = []
        if mutation and mutation["accepted"]:
            mutation["lessons_validated"] = len(lessons)
            for trace in traces:
                if trace["trace_id"] in mutation["supporting_trace_ids"]:
                    trace = copy.deepcopy(trace)
                    trace.update(adaptation_proposed={
                        "mutation_id": mutation["mutation_id"], "mutation_type": mutation["mutation_type"]},
                        adaptation_accepted=True, resulting_harness_version=mutation["resulting_version"])
                    updates.append(trace)
        summary = dict(loop.summarize(traces), generation=state["generation"],
            version=arch["version_id"], trace_ids=[t["trace_id"] for t in traces],
            collisions=sum(t["metrics"].get("bumps", 0) for t in traces),
            replans=sum(t["metrics"].get("replans", 0) for t in traces),
            mutation=loop._mutation_summary(mutation))
        next_version = mutation["resulting_version"] if mutation and mutation["accepted"] else arch["version_id"]
        state = _commit(db, dict(state, generation=state["generation"]+1,
            cursor=0, phase="episodes", version_id=next_version,
            history=state["history"]+[summary],
            pending={"kind": "generation", "mutation": mutation, "parent": arch,
                     "lesson_updates": lessons, "trace_updates": updates, "summary": summary}))
        state = _flush(db, state)
        print(f"[runtime] gen={summary['generation']} {summary['version']} "
              f"success={summary['success_rate']:.2f} avg_steps={summary['avg_steps']} next={next_version}")
    db.save("evaluations", {"_id": "run-summary", "kind": "run", "eval_id": db.run_id,
        "tasks": tasks, "seeds": seeds, "generations": len(state["history"]), "trials": 1,
        "trial_reports": [{"trial": 0, "generations": state["history"]}],
        "budget": state["ledger"], "created_at": loop.utcnow()})
    return state["history"]


def add_arguments(parser):
    parser.add_argument("--backend", choices=("auto", "atlas", "local"), default="auto")
    parser.add_argument("--env-file", help="Explicit ignored KEY=VALUE file; does not override exported variables")
    parser.add_argument("--run-id", help="Storage namespace; generated for new runs")
    parser.add_argument("--resume", action="store_true", help="Continue this run to the TOTAL generation count")
    parser.add_argument("--status", action="store_true", help="Read the checkpoint and stored counts; requires --run-id")
    parser.add_argument("--stop-after-episodes", type=int, help="Commit this many new episodes, then stop")
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-proposals", type=int, default=10)
    parser.add_argument("--max-model-calls", type=int, default=10)
    parser.add_argument("--max-model-tokens", type=int, default=100000)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)


def limits_from_args(args):
    return RuntimeLimits(max_episode_steps=args.max_episode_steps, max_seconds=args.max_seconds,
        max_proposals=args.max_proposals, max_model_calls=args.max_model_calls,
        max_model_tokens=args.max_model_tokens, max_cost_usd=args.max_cost_usd)


def get_run_status(backend, run_id):
    """Read progress and projection counts for product orchestration."""
    db = backend if isinstance(backend, storage.RunBackend) else storage.RunBackend(backend, run_id)
    if db.run_id != run_id:
        raise ValueError("Backend and requested run IDs differ")
    state = db.find_one("policies", {"_id": CHECKPOINT_ID})
    if state is None:
        raise ValueError("No checkpoint for this run")
    return {"run_id": db.run_id, "backend": db.backend_name,
        "revision": state["revision"], "generation": state["generation"],
        "target_generations": state.get("target_generations"),
        "complete": (state["generation"] >= state["target_generations"] and not state.get("pending"))
                    if "target_generations" in state else None,
        "version_id": state["version_id"], "phase": state["phase"], "cursor": state["cursor"],
        "pending": bool(state.get("pending")), "committed_episodes": len(state["receipts"]),
        "development_traces": sum(t.get("context", {}).get("tag") == "development"
            for t in db.find("trajectories", limit=None)),
        "experiences": len(db.find("experiences", limit=None)),
        "architectures": sorted(d["version_id"] for d in db.find("architectures", limit=None)),
        "mutations": len(db.find("mutations", limit=None)),
        "navigation_counts": {k: (db.find_one("skills", {"name": "navigation"}) or {}).get(k, 0)
                              for k in ("success_count", "failure_count")},
        "budget": state["ledger"]}
