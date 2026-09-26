"""Durable RGB -> evidence -> memory -> component-verdict orchestration.

The runner accepts public task definitions, RGB-only registry and numeric
checkpoints. It has no reference-label reader. Evaluation is in benchmark.py.
"""

import copy
import json
import re
from .budget import BudgetExceeded
from .evidence_replay import to_event
from .harness import advance
from .industreal_data import get_evidence, register_frames
from .model_client import DEFAULT_MODEL
from .perception.video import json_key
from .verifier import verify, memory_item

ENGINE_VERSION = "real-rgb-pipeline-v1"
MAX_STATE_BYTES = 8_000_000


def _commit(store, session, state, expected):
    state = copy.deepcopy(state)
    state["revision"] = (expected or 0) + 1
    if len(json.dumps(state).encode()) > MAX_STATE_BYTES:
        raise BudgetExceeded("Pipeline checkpoint full")
    store.compare_and_swap(session, state, expected)
    return state


def _flush(store, session, state):
    if state.get("pending"):
        store.put_decision(session, state["pending"])
        state = _commit(store, session, dict(state, pending=None), state["revision"])
    return state


def run_recording(
    store,
    run_id,
    recording_id,
    rgb_dir,
    checkpoints,
    components,
    time_map,
    client,
    work_dir,
    *,
    mode="memory",
    policy=None,
    allow_candidate=False,
    stop_after=None,
    inspect_fn=get_evidence,
    verify_fn=verify,
    model=DEFAULT_MODEL,
    reference_image=None,
):
    from pathlib import Path

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    if mode not in ("memory", "stateless"):
        raise ValueError("Unknown memory mode")
    if (
        not checkpoints
        or len(checkpoints) > 1000
        or any(type(f) is not int or f < 0 for f in checkpoints)
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError("Require ordered unique checkpoint frames")
    if (
        not isinstance(components, dict)
        or not components
        or len(components) > 11
        or any(not isinstance(k, str) or not k or len(k) > 80 for k in components)
        or any(
            not isinstance(v, str) or not v or len(v) > 300 for v in components.values()
        )
    ):
        raise ValueError("Require bounded public component definitions")
    if (
        time_map.get("recording_id") != recording_id
        or time_map.get("verified") is not True
    ):
        raise ValueError("Require verified native frame time map")
    if policy:
        from .policies import validate_policy

        validate_policy(policy)
        if policy["status"] != "accepted" and not (
            allow_candidate and policy["status"] == "candidate"
        ):
            raise ValueError("Only accepted policies run outside candidate validation")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be positive")
    root = Path(work_dir)
    manifest = root / "rgb_manifest.json"
    from .industreal_data import load_manifest

    if recording_id not in load_manifest(manifest)["recordings"]:
        register_frames(manifest, recording_id, rgb_dir)
    registration = load_manifest(manifest)["recordings"][recording_id]
    session = f"pipeline:{run_id}:{mode}:{recording_id}"
    config = {
        "engine_version": ENGINE_VERSION,
        "run_id": run_id,
        "recording_id": recording_id,
        "mode": mode,
        "model": model,
        "components_hash": json_key(components),
        "policy_hash": json_key(policy or {}),
        "checkpoints": checkpoints,
        "rgb_index_hash": registration["index_hash"],
        "mapping_id": time_map["mapping_id"],
        "time_map_hash": json_key(time_map),
    }
    from .verifier import PROMPT
    from .perception.inspect import PROMPT as INSPECTION_PROMPT
    from .perception.video import digest_file

    config.update(
        verifier_prompt_hash=json_key(PROMPT),
        inspection_prompt_hash=json_key(INSPECTION_PROMPT),
        reference_image_hash=digest_file(reference_image) if reference_image else None,
        client_version=getattr(client, "version", "bounded-json-v1"),
    )
    state = store.load(session)
    if state is None:
        state = {
            "revision": 0,
            "config": config,
            "cursor": 0,
            "recent": [],
            "receipts": [],
            "harness_state": None,
            "pending": None,
        }
        store.compare_and_swap(session, state, None)
    elif state["config"] != config:
        raise ValueError("Resume configuration changed; use another run ID")
    state = _flush(store, session, state)
    new = 0
    while state["cursor"] < len(checkpoints):
        frame = checkpoints[state["cursor"]]
        import time

        started = time.monotonic()
        before = client.budget.summary()["dispatched_calls"] if client.budget else 0
        packet = inspect_fn(
            manifest,
            recording_id,
            f"{recording_id}:{frame}",
            "all-components",
            max(0, frame - 50),
            frame,
            frame,
            model=model,
            cache_dir=root / "perception",
            client=client,
        )
        event = to_event(packet, frame_times=time_map)
        stable = {
            k: v
            for k, v in packet.items()
            if k not in ("cache_hit", "new_api_calls", "logical_tool_calls")
        }
        store.put_inspection(
            session,
            event["event_id"],
            {
                "kind": "neutral_rgb_inspection",
                "packet": stable,
                "event": event,
                "mapping_id": time_map["mapping_id"],
            },
        )
        memory = state["recent"][-3:] if mode == "memory" else []
        provider_failed = False
        try:
            answer, meta = verify_fn(
                packet,
                components,
                memory,
                client,
                root,
                model=model,
                policy=policy,
                reference_image=reference_image,
            )
        except ValueError:
            provider_failed = True
            answer = [
                {
                    "component_id": name,
                    "verdict": "insufficient_evidence",
                    "rationale": "Verifier failed or returned invalid evidence; no assembly conclusion.",
                    "evidence_frame_ids": [],
                }
                for name in sorted(components)
            ]
            meta = {
                "cache_hit": False,
                "call": {"latency_seconds": 0},
                "new_api_calls": 1,
            }
        after = client.budget.summary()["dispatched_calls"] if client.budget else before
        verdicts = []
        for index, item in enumerate(answer):
            verdicts.append(
                {
                    "schema_version": 1,
                    "source_kind": "rgb_vlm",
                    "run_id": run_id,
                    "recording_id": recording_id,
                    "checkpoint_id": f"{recording_id}:{frame}:{item['component_id']}",
                    "cursor_frame": frame,
                    "component_id": item["component_id"],
                    "verdict": item["verdict"],
                    "policy_id": (policy or {}).get("policy_id", "baseline-v1"),
                    "model_id": model,
                    "rationale": item["rationale"],
                    "evidence_frame_ids": item["evidence_frame_ids"],
                    "evidence_id": packet["evidence_id"],
                    "provider_failed": provider_failed,
                    "model_calls": after - before if index == 0 else 0,
                    "latency_ms": (
                        (time.monotonic() - started) * 1000 if index == 0 else 0
                    ),
                }
            )
        unresolved = [
            v["component_id"]
            for v in verdicts
            if v["verdict"] in ("incorrect", "insufficient_evidence")
        ]
        if unresolved:
            event["issue"] = {
                "id": "assembly-review",
                "description": "Needs current evidence: " + ", ".join(unresolved),
            }
        else:
            event["resolves"] = ["assembly-review"]
        harness_state, decision = advance(
            state["harness_state"], event, mode=mode, policy=policy
        )
        point = dict(
            decision,
            cursor_frame=frame,
            run_id=run_id,
            recording_id=recording_id,
            mode=mode,
            verdicts=verdicts,
            inspection_cache_hit=packet["cache_hit"],
            verifier_cache_hit=meta["cache_hit"],
            new_api_calls=after - before,
            memory_items=len(memory),
        )
        recent = (
            (state["recent"] + [memory_item(packet, answer)])[-3:]
            if mode == "memory"
            else []
        )
        state = _commit(
            store,
            session,
            dict(
                state,
                cursor=state["cursor"] + 1,
                recent=recent,
                receipts=state["receipts"] + [event["event_id"]],
                harness_state=harness_state,
                pending=point,
            ),
            state["revision"],
        )
        state = _flush(store, session, state)
        new += 1
        counts = {}
        for v in verdicts:
            counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        print(
            f"[pipeline] {run_id}/{mode} {recording_id} frame={frame} "
            + json.dumps(counts)
            + f" calls={after-before} memory={len(memory)}",
            flush=True,
        )
        if stop_after is not None and new >= stop_after:
            break
    decisions = store.decisions(session)
    return {
        "session": session,
        "complete": state["cursor"] == len(checkpoints),
        "committed_checkpoints": state["cursor"],
        "decisions": decisions,
        "verdicts": [v for d in decisions for v in d["verdicts"]],
        "pending": bool(state.get("pending")),
    }
