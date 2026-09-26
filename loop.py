"""The explicit recursive improvement loop.

run_episode -> evaluate -> classify -> find patterns -> propose mutation ->
validate -> promote (or reject) -> persist everything.

Usage:
    python -m loop --generations 3 --tasks pick-and-deliver --seeds 7,8
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import architectures
import db as db_module
from contracts import Trace
from evolver import (
    analyze_failure,
    classify,
    find_patterns,
    get_validated_lessons,
    mark_lessons_validated,
    propose_mutation,
    validate,
)
from sim import Robot, World, get_task
from skills import ensure_seed_skills, record_result


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------- #
# Cross-episode memory (only active when the spatial_memory module is on).
def _memory_key(version_id: str, environment_id: str) -> str:
    return f"memory:{version_id}:{environment_id}"


def load_memory(db, arch: Dict, environment_id: str) -> Optional[Dict]:
    if "spatial_memory" not in arch.get("active_modules", []):
        return None
    doc = db.find_one("policies", {"_id": _memory_key(arch["version_id"], environment_id)})
    if not doc:
        return None
    return {
        "blocked": {tuple(c) for c in doc.get("blocked", [])},
        "visited": set(),
    }


def save_memory(db, arch: Dict, environment_id: str, memory: Dict) -> None:
    db.save("policies", {
        "_id": _memory_key(arch["version_id"], environment_id),
        "version_id": arch["version_id"],
        "environment_id": environment_id,
        "blocked": sorted(memory.get("blocked", set())),
        "updated_at": utcnow(),
    })


# ---------------------------------------------------------------------- #
def run_episode(db, arch: Dict, task_id: str, seed: int,
                memory: Optional[Dict] = None, tag: str = "episode",
                persist: bool = True,
                history: Optional[List[Dict]] = None) -> Tuple[Dict, Dict]:
    spec = get_task(task_id)
    world = World(spec, seed)
    if memory is None:
        memory = load_memory(db, arch, spec.environment_id)
    robot = Robot(world, arch, memory=memory)
    blocked_before = set(robot.memory["blocked"])

    trace = Trace(
        trace_id=uuid.uuid4().hex,
        task_id=task_id,
        environment_id=spec.environment_id,
        harness_version=arch["version_id"],
        seed=seed,
        resulting_harness_version=arch["version_id"],
        created_at=utcnow(),
    )
    trace.context = {
        "tag": tag,
        "assumed_map": "stale",
        "active_modules": list(arch.get("active_modules", [])),
        "verification_strategy": list(arch.get("verification_strategy", [])),
        "context_strategy": arch.get("context_strategy"),
        # Validated lessons from earlier failures on this task are consulted
        # before acting.
        "lessons_consulted": get_validated_lessons(db, task_id) if persist else [],
    }
    max_steps = arch.get("policies", {}).get("max_steps") or spec.max_steps
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 600:
        raise ValueError("Episode step budget must be an integer between 1 and 600")
    trace.metrics["max_steps"] = max_steps
    tools = {"navigation"}

    for step in range(max_steps):
        obs = world.observe()
        trace.observations.append({
            "pos": list(obs["pos"]), "dir": obs["dir"],
            "bump": obs["bump"], "carrying": obs["carrying"],
        })
        action = robot.decide(obs)
        trace.actions.append(action)
        event = world.step(action)
        if event["bump"]:
            trace.metrics["bumps"] = trace.metrics.get("bumps", 0) + 1
        if event.get("failed_pick"):
            trace.metrics["failed_picks"] = trace.metrics.get("failed_picks", 0) + 1
        if event["picked"]:
            tools.add("pickup")
            if event.get("picked_kind") != spec.target_kind:
                trace.metrics["picked_wrong_kind"] = True
        if event["dropped"]:
            tools.add("delivery")
        if world.task_complete():
            trace.success = True
            break

    trace.selected_tools = sorted(tools)
    trace.metrics["steps"] = len(trace.actions)
    trace.metrics["replans"] = robot.replans
    trace.metrics["visited_cells"] = len(robot.memory["visited"])
    trace.metrics["learned_blocked"] = sorted(robot.memory["blocked"] - blocked_before)
    trace.metrics["known_blocked"] = len(robot.memory["blocked"])
    trace.outcome = {
        "delivered": list(world.delivered),
        "final_pos": list(world.robot_pos),
    }
    if not trace.success:
        trace.failure_category = classify(trace.to_dict(), arch, history)
        lesson = analyze_failure(trace.to_dict(), arch)
    else:
        lesson = None

    trace_dict = trace.to_dict()
    if persist:
        db.save("trajectories", trace_dict)
        db.save("experiences", {
            "_id": trace.trace_id, "trace_id": trace.trace_id,
            "task_id": task_id, "environment_id": spec.environment_id,
            "architecture_version": arch["version_id"],
            "success": trace.success,
            "failure_type": trace.failure_category,
            "steps": trace.metrics["steps"],
            "created_at": trace.created_at,
        })
        if not trace.success:
            db.save("failures", {
                "_id": trace.trace_id, "trace_id": trace.trace_id,
                "failure_category": trace.failure_category,
                "task_id": task_id,
                "created_at": trace.created_at,
            })
        if lesson:
            db.save("lessons", lesson)
        # Real skill success metrics from this episode.
        record_result(db, "navigation", trace.success)
        if "pickup" in tools:
            record_result(db, "pickup", trace.success)
        if "delivery" in tools:
            record_result(db, "delivery", trace.success)
        if trace.metrics.get("replans", 0) > 0:
            record_result(db, "collision_check", trace.success)
        if "spatial_memory" in arch.get("active_modules", []):
            record_result(db, "spatial_memory", trace.success)

    # Persist cross-episode memory when the module is active.
    if persist and "spatial_memory" in arch.get("active_modules", []):
        save_memory(db, arch, spec.environment_id, robot.memory)

    return trace_dict, robot.memory


# ---------------------------------------------------------------------- #
def maybe_evolve(db, arch: Dict, recent_traces: List[Dict]) -> Tuple[Dict, Optional[Dict]]:
    """Propose, validate, and possibly promote one architecture mutation."""
    patterns = find_patterns(recent_traces)
    for pattern in patterns:
        mutation = propose_mutation(pattern, arch, traces=recent_traces)
        if mutation is None:
            continue
        child = architectures.apply_mutation(arch, mutation)
        validation = validate(db, arch, child, run_episode)
        mutation["validation"] = validation
        if validation["passed"]:
            child = architectures.create_version(db, arch, mutation)
            mutation["accepted"] = True
            mutation["lessons_validated"] = mark_lessons_validated(db, mutation)
            db.save("mutations", mutation)
            # Stamp the traces that triggered this evolution.
            for t in recent_traces:
                if t["trace_id"] in mutation["supporting_trace_ids"]:
                    t["adaptation_proposed"] = {
                        "mutation_id": mutation["mutation_id"],
                        "mutation_type": mutation["mutation_type"],
                    }
                    t["adaptation_accepted"] = True
                    t["resulting_harness_version"] = child["version_id"]
                    db.save("trajectories", t)
            return child, mutation
        db.save("mutations", mutation)  # rejected: still recorded
        return arch, mutation
    return arch, None


# ---------------------------------------------------------------------- #
def summarize(traces: List[Dict]) -> Dict:
    n = len(traces)
    succ = sum(1 for t in traces if t.get("success"))
    steps = sum(t.get("metrics", {}).get("steps", 0) for t in traces)
    return {
        "episodes": n,
        "successes": succ,
        "success_rate": round(succ / n, 3) if n else 0.0,
        "avg_steps": round(steps / n, 1) if n else 0.0,
    }


def _mutation_summary(mutation: Optional[Dict]) -> Optional[Dict]:
    if mutation is None:
        return None
    v = mutation.get("validation", {})
    return {
        "mutation_id": mutation["mutation_id"],
        "accepted": mutation["accepted"],
        "mutation_type": mutation["mutation_type"],
        "new_component": mutation.get("new_component"),
        "resulting_version": mutation.get("resulting_version"),
        "lessons_validated": mutation.get("lessons_validated", 0),
        "validation_passed": v.get("passed"),
        "validation_parent": v.get("parent"),
        "validation_child": v.get("child"),
    }


def run_generations(db, task_ids: List[str], seeds: List[int],
                    n_generations: int) -> List[Dict]:
    ensure_seed_skills(db)
    arch = architectures.ensure_v0(db)
    print(f"[loop] db={getattr(db, 'backend_name', '?')} start={arch['version_id']}")
    history = []
    all_traces: List[Dict] = []
    for g in range(n_generations):
        traces = []
        for task_id in task_ids:
            for seed in seeds:
                trace, _memory = run_episode(
                    db, arch, task_id, seed,
                    history=[t for t in all_traces if t["task_id"] == task_id],
                )
                traces.append(trace)
        all_traces.extend(traces)
        summary = summarize(traces)
        summary.update({
            "generation": g,
            "version": arch["version_id"],
            "trace_ids": [t["trace_id"] for t in traces],
            "collisions": sum(t["metrics"].get("bumps", 0) for t in traces),
            "replans": sum(t["metrics"].get("replans", 0) for t in traces),
        })
        db.save("evaluations", summary)
        arch, mutation = maybe_evolve(db, arch, traces)
        summary["mutation"] = _mutation_summary(mutation)
        history.append(summary)
        print(f"[gen {g}] {summary['version']}: "
              f"{summary['successes']}/{summary['episodes']} success, "
              f"avg_steps={summary['avg_steps']}")
        if mutation is None:
            print(f"[gen {g}] no patterns found; architecture unchanged.")
        elif mutation["accepted"]:
            v = mutation["validation"]
            print(f"[gen {g}] MUTATION ACCEPTED: {mutation['mutation_type']} "
                  f"+{mutation['new_component']} -> {mutation['resulting_version']} "
                  f"(val parent {v['parent']['success_rate']:.2f}/{v['parent']['total_steps']} "
                  f"steps vs child {v['child']['success_rate']:.2f}/{v['child']['total_steps']} steps, "
                  f"{mutation.get('lessons_validated', 0)} lesson(s) validated)")
        else:
            print(f"[gen {g}] mutation {mutation['mutation_type']} rejected by validator.")
    return history


def main() -> None:
    import runtime
    parser = argparse.ArgumentParser(description="Recursive harness loop")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--tasks", default="pick-and-deliver",
                        help="comma-separated task ids")
    parser.add_argument("--seeds", default="7,8",
                        help="comma-separated seeds")
    runtime.add_arguments(parser)
    args = parser.parse_args()
    database = db_module.get_db(args.backend, env_file=args.env_file)
    try:
        if args.status:
            if not args.run_id:
                parser.error("--status requires --run-id")
            import json
            print(json.dumps(runtime.get_run_status(database, args.run_id), indent=2))
            return
        runtime.execute_run(
            database, [t.strip() for t in args.tasks.split(",")],
            [int(s) for s in args.seeds.split(",")], args.generations,
            run_id=args.run_id, resume=args.resume,
            limits=runtime.limits_from_args(args),
            stop_after_episodes=args.stop_after_episodes,
        )
    finally:
        database.close()


if __name__ == "__main__":
    main()
