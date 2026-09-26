"""Deterministic memory plumbing. Replace perception upstream, not with labels."""

import hashlib
import json
import math
import time
from budget import BudgetExceeded

FIELDS = {"event_id", "recording_id", "timestamp", "observation", "completed_steps", "issue", "resolves",
          "schema_version", "evidence_refs"}


def validate(event):
    if not isinstance(event, dict):
        raise ValueError("Observation must be an object")
    if len(json.dumps(event).encode()) > 64000:
        raise ValueError("Observation exceeds 64KB; reference frames instead of embedding them")
    unknown = set(event) - FIELDS
    if unknown:
        raise ValueError(f"Unsupported observation fields (possible label leakage): {sorted(unknown)}")
    for key in ("event_id", "recording_id", "observation"):
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError(f"{key} must be a nonempty string")
    timestamp = event.get("timestamp")
    if isinstance(event.get("schema_version", 1), bool) or event.get("schema_version", 1) != 1:
        raise ValueError("Unsupported observation schema_version")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be a finite nonnegative number")
    for key in ("completed_steps", "resolves"):
        values = event.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"{key} must be a list of nonempty strings")
    issue = event.get("issue")
    if issue is not None:
        if not isinstance(issue, dict) or set(issue) != {"id", "description"}:
            raise ValueError("issue must contain exactly id and description")
        if any(not isinstance(v, str) or not v for v in issue.values()):
            raise ValueError("issue values must be nonempty strings")
    refs = event.get("evidence_refs", [])
    if not isinstance(refs, list) or len(refs) > 64:
        raise ValueError("evidence_refs must be a list of at most 64 references")
    for ref in refs:
        required = {"recording_id", "start_seconds", "end_seconds"}
        if not isinstance(ref, dict) or not required <= set(ref) or set(ref) - required - {"frame_ids"}:
            raise ValueError("Evidence reference requires recording_id/start_seconds/end_seconds")
        if ref["recording_id"] != event["recording_id"]:
            raise ValueError("Evidence must belong to this recording")
        start, end = ref["start_seconds"], ref["end_seconds"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (start, end)) or not 0 <= start <= end <= timestamp:
            raise ValueError("Evidence must be a finite interval available by this event")
        frames = ref.get("frame_ids", [])
        if not isinstance(frames, list) or len(frames) > 32 or any(not isinstance(f, str) or not f for f in frames):
            raise ValueError("frame_ids must contain at most 32 nonempty strings")


def _policy_issues(event, policy, prior_completed):
    """Bounded adaptive rules. Returns [(issue_id, description)] to open."""
    opened = []
    if not policy or not policy.get("strict_order"):
        return opened
    requires = policy.get("requires", {})
    for step in event.get("completed_steps", []):
        for prereq in requires.get(step, []):
            if prereq not in prior_completed:
                opened.append((
                    f"order:{step}",
                    f"Step '{step}' completed before prerequisite '{prereq}'; need explicit evidence.",
                ))
    return opened


def advance(state, event, mode="memory", policy=None):
    validate(event)
    if mode not in ("memory", "stateless"):
        raise ValueError("Unknown mode")
    policy_digest = hashlib.sha256(json.dumps(policy or {}, sort_keys=True).encode()).hexdigest()
    if state is None:
        state = {"recording_id": event["recording_id"], "mode": mode, "last_timestamp": -1,
                 "processed": {}, "completed_steps": [], "open_issues": {}, "recent": [],
                 "policy_version": (policy or {}).get("version", 0), "policy_digest": policy_digest}
    if state["recording_id"] != event["recording_id"] or state["mode"] != mode:
        raise ValueError("Use a different session for each recording and mode")
    if state.get("policy_version", 0) != (policy or {}).get("version", 0) or state.get("policy_digest", policy_digest) != policy_digest:
        raise ValueError("Policy changed; start a separate session for the new policy")
    digest = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
    previous = state["processed"].get(event["event_id"])
    if previous:
        if previous != digest:
            raise ValueError("An existing event_id was reused with different content")
        return state, {"event_id": event["event_id"], "status": "already_processed"}
    if event["timestamp"] < state["last_timestamp"]:
        raise ValueError("New events must arrive in chronological order")
    state = json.loads(json.dumps(state))
    state["policy_digest"] = policy_digest
    if mode == "stateless":
        state["completed_steps"], state["open_issues"], state["recent"] = [], {}, []
    for issue_id in event.get("resolves", []):
        state["open_issues"].pop(issue_id, None)
    issue = event.get("issue")
    if issue:
        state["open_issues"][issue["id"]] = {"description": issue["description"],
                                            "evidence_event_id": event["event_id"],
                                            "evidence_refs": event.get("evidence_refs", [])}
    prior_completed = set(state["completed_steps"])
    state["completed_steps"] = sorted(prior_completed | set(event.get("completed_steps", [])))
    for issue_id, description in _policy_issues(event, policy, prior_completed):
        state["open_issues"].setdefault(issue_id, {"description": description,
                                                  "evidence_event_id": event["event_id"],
                                                  "evidence_refs": event.get("evidence_refs", []),
                                                  "policy": True})
    state["recent"] = (state["recent"] + [{"event_id": event["event_id"], "observation": event["observation"],
        "evidence_refs": event.get("evidence_refs", [])}])[-5:]
    state["processed"][event["event_id"]] = digest
    state["last_timestamp"] = event["timestamp"]
    # No claim of success: absence of an issue is not proof a task was completed.
    decision = {"event_id": event["event_id"], "timestamp": event["timestamp"],
                "action": "request_verification" if state["open_issues"] else "continue_observing",
                "open_issues": state["open_issues"], "completed_steps": state["completed_steps"],
                "evidence_refs": event.get("evidence_refs", [])}
    return state, decision


def _commit(store, session, state, expected):
    state = dict(state, revision=(expected or 0) + 1)
    if len(json.dumps(state).encode()) > 8_000_000:
        raise BudgetExceeded("Replay checkpoint capacity reached; use another bounded recording session")
    store.compare_and_swap(session, state, expected)
    return state


def _flush(store, session, state):
    if state and state.get("pending_decision"):
        store.put_decision(session, state["pending_decision"])
        state = _commit(store, session, dict(state, pending_decision=None), state.get("revision", 0))
    return state


def replay(store, session, events, mode="memory", limit=None, policy=None, *, max_events=10000, max_seconds=300):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    state = store.load(session)
    if state and state.get("replay_limits", {"max_events": max_events}) != {"max_events": max_events}:
        raise ValueError("Resume must preserve the session's total event budget")
    started = time.monotonic()
    state = _flush(store, session, state)
    count = 0
    for event in events:
        if time.monotonic() - started >= max_seconds:
            raise BudgetExceeded("Replay invocation time budget exhausted; resume this session")
        if limit is not None and count >= limit:
            break
        expected = state.get("revision", 0) if state else None
        state, decision = advance(state, event, mode, policy)
        if decision.get("status") != "already_processed":
            if len(state["processed"]) > max_events:
                raise BudgetExceeded("Replay event budget exhausted")
            # Event receipt and its memory effects are persisted together.
            state = _commit(store, session, dict(state, pending_decision=decision,
                             replay_limits={"max_events": max_events}), expected)
            state = _flush(store, session, state)
            count += 1
        yield decision
