"""Mutation validator: parent vs child on a validation task set.

Accept when the child strictly improves the success rate, or matches it with
strictly fewer steps. The validator runs episodes through a caller-supplied
runner so this module never imports the loop (no circular import).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
from sim import get_task

# (task_id, seed) pairs held fixed for validation.
VAL_TASKS: List[Tuple[str, int]] = [
    ("pick-and-deliver", 101),
    ("pick-and-deliver", 102),
    ("multi-room-deliver", 101),
]


def _summarize(traces: List[Dict]) -> Dict:
    n = len(traces)
    succ = sum(1 for t in traces if t.get("success"))
    steps = sum(t.get("metrics", {}).get("steps", 0) for t in traces)
    return {
        "episodes": n,
        "success_rate": succ / n if n else 0.0,
        "total_steps": steps,
        "avg_steps": steps / n if n else 0.0,
    }


def validate(db, parent: Dict, child: Dict,
             run_episode: Callable) -> Dict:
    """Cold-start each architecture, with sequential memory per environment.

    Evaluation traces are audited, but cannot update developmental lessons,
    skill counters or persistent spatial memory.
    """
    results = {}
    for arch in (parent, child):
        traces = []
        memories = {}
        for task_id, seed in VAL_TASKS:
            environment = get_task(task_id).environment_id
            memory = memories.get(environment, {"blocked": set(), "visited": set()})
            trace, memory_out = run_episode(
                db, arch, task_id, seed, memory=memory,
                tag="validation", persist=False,
            )
            # spatial_memory persists across validation episodes too.
            if "spatial_memory" in arch.get("active_modules", []):
                memories[environment] = {"blocked": set(memory_out["blocked"]), "visited": set()}
            db.save("trajectories", trace)
            traces.append(trace)
        results[arch["version_id"]] = dict(_summarize(traces),
                                           trace_ids=[t["trace_id"] for t in traces])

    p, c = results[parent["version_id"]], results[child["version_id"]]
    passed = bool(
        c["success_rate"] > p["success_rate"]
        or (c["success_rate"] == p["success_rate"] and c["total_steps"] < p["total_steps"])
    )
    return {"parent": p, "child": c, "passed": passed}
