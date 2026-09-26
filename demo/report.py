"""Render the decision-point replay player from saved records only; it never calls a model or reads labels."""
import base64
import html
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
SOURCE_KINDS = ("mock", "rgb_vlm", "released_predictions")
VERDICTS = ("correct", "incorrect", "not_completed", "insufficient_evidence")
ARMS = ("baseline", "candidate")
DECISIONS = ("accepted", "rejected", "inconclusive", "no_proposal")
METRICS = (("state_accuracy", "State accuracy"), ("false_correct", "False-correct count"),
           ("incorrect_recall", "Incorrect-state recall"), ("abstentions", "Abstentions"),
           ("model_calls", "Model calls"), ("latency_ms", "Latency (ms)"))
MAX_CLIPS = 3
FONT_FILE = Path(__file__).with_name("assets") / "RedHatDisplay-latin.woff2"

STR, OPT_STR, INT, STRS, INTS, DICT, OPT_DICT, LIST = (
    "str", "str|null", "int", "list[str]", "list[int]", "dict", "dict|null", "list")

SPECS = {
    "evidence": {"evidence_id": STR, "recording_id": STR, "checkpoint_id": STR, "cursor_frame": INT,
                 "component_id": STR, "frame_ids": INTS, "observations": STRS, "uncertainties": STRS,
                 "provider": DICT},
    "verdict": {"run_id": STR, "recording_id": STR, "checkpoint_id": STR, "cursor_frame": INT,
                "component_id": STR, "verdict": VERDICTS, "reason": STR, "evidence_ids": STRS,
                "open_issue_ids": STRS, "policy_id": STR, "model_id": STR},
    "issue_event": {"run_id": STR, "seq": INT, "issue_id": STR, "event": ("opened", "carried", "resolved"),
                    "checkpoint_id": STR, "cursor_frame": INT, "component_id": STR, "description": STR,
                    "evidence_ids": STRS},
    "case": {"case_id": STR, "split": STR, "recording_id": STR, "component_id": STR, "frame_start": INT,
             "frame_end": INT, "clip_path": STR, "verdict_then": VERDICTS,
             "revealed_outcome": ("correct", "incorrect", "not_completed"), "summary": STR},
    "policy": {"policy_id": STR, "parent_policy_id": OPT_STR, "status": ("candidate", "accepted", "rejected"),
               "checklist": STRS, "supporting_dev_case_ids": STRS},
    "evaluation": {"recording_ids": STRS, "baseline_policy_id": STR, "candidate_policy_id": OPT_STR,
                   "decision": DECISIONS, "reason": STR, "scored": INT, "excluded": INT,
                   "baseline": DICT, "candidate": OPT_DICT},
    "snapshot": {"backend": ("atlas", "local"), "session_key": DICT, "last_committed_checkpoint": OPT_STR,
                 "open_issues": LIST},
    "timing": {"recordings": DICT},
}
OPTIONAL = {"verdict": {"recalled_case_ids": STRS}, "snapshot": {"restart": DICT}}


def _ok(value, kind):
    if isinstance(kind, tuple):
        return isinstance(value, str) and value in kind
    if kind == STR:
        return isinstance(value, str) and value != ""
    if kind == OPT_STR:
        return value is None or _ok(value, STR)
    if kind == INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == STRS:
        return isinstance(value, list) and all(_ok(v, STR) for v in value)
    if kind == INTS:
        return isinstance(value, list) and all(_ok(v, INT) for v in value)
    if kind == DICT:
        return isinstance(value, dict)
    if kind == OPT_DICT:
        return value is None or isinstance(value, dict)
    if kind == LIST:
        return isinstance(value, list)
    raise AssertionError(kind)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _map_points(entry):
    return sorted([int(frame), float(seconds)] for frame, seconds in entry["frame_map"].items())


def _timing_entry_problems(rid, entry):
    label = f"timing[{rid}]"
    if not isinstance(entry, dict):
        return [f"{label}: expected an object"]
    if ("fps" in entry) == ("frame_map" in entry):
        return [f"{label}: set exactly one of fps or frame_map"]
    problems = []
    if entry.get("video_path") is not None and not _ok(entry["video_path"], STR):
        problems.append(f"{label}: video_path must be a nonempty string or null")
    if "duration_s" in entry and not (_number(entry["duration_s"]) and entry["duration_s"] > 0):
        problems.append(f"{label}: duration_s must be a positive number")
    if "fps" in entry:
        if not (_number(entry["fps"]) and entry["fps"] > 0):
            problems.append(f"{label}: fps must be a positive number")
        if "frame_offset" in entry and not _ok(entry["frame_offset"], INT):
            problems.append(f"{label}: frame_offset must be an integer")
        return problems
    fmap = entry["frame_map"]
    if not isinstance(fmap, dict) or len(fmap) < 2:
        return problems + [f"{label}: frame_map needs at least two frame -> seconds points"]
    try:
        points = _map_points(entry)
    except (TypeError, ValueError):
        return problems + [f"{label}: frame_map keys must be integer frames and values numbers"]
    if any(not math.isfinite(t) or t < 0 for _, t in points):
        problems.append(f"{label}: frame_map seconds must be finite and nonnegative")
    elif any(b[1] <= a[1] for a, b in zip(points, points[1:])):
        problems.append(f"{label}: frame_map seconds must increase with frame number")
    return problems


def validate(kind, record, label):
    if not isinstance(record, dict):
        return [f"{label}: expected a JSON object"]
    problems = []
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"{label}: schema_version must be {SCHEMA_VERSION} (got {record.get('schema_version')!r})")
    if record.get("source_kind") not in SOURCE_KINDS:
        problems.append(f"{label}: source_kind must be one of {', '.join(SOURCE_KINDS)}")
    for field, kind_ in SPECS[kind].items():
        if field not in record:
            problems.append(f"{label}: missing {field}")
        elif not _ok(record[field], kind_):
            problems.append(f"{label}: invalid {field}")
    for field, kind_ in OPTIONAL.get(kind, {}).items():
        if field in record and not _ok(record[field], kind_):
            problems.append(f"{label}: invalid {field}")
    if problems:
        return problems
    if kind == "evidence" and any(f > record["cursor_frame"] for f in record["frame_ids"]):
        problems.append(f"{label}: frame_ids exceed cursor_frame (future frames are not allowed)")
    if kind == "policy" and len(record["checklist"]) > 1:
        problems.append(f"{label}: checklist may add at most one instruction")
    if kind == "case" and record["frame_end"] < record["frame_start"]:
        problems.append(f"{label}: frame_end is before frame_start")
    if kind == "evaluation":
        if record["scored"] < 0 or record["excluded"] < 0:
            problems.append(f"{label}: scored and excluded must be nonnegative")
        for arm in ARMS:
            metrics = record[arm]
            if metrics is None:
                if arm == "baseline" or record["decision"] != "no_proposal":
                    problems.append(f"{label}: {arm} metrics are required")
                continue
            for key, _ in METRICS:
                if key not in metrics:
                    problems.append(f"{label}: {arm}.{key} missing")
                elif not (_number(metrics[key]) or metrics[key] == "unavailable"):
                    problems.append(f"{label}: {arm}.{key} must be a number or \"unavailable\"")
    if kind == "timing":
        for rid, entry in record["recordings"].items():
            problems.extend(_timing_entry_problems(rid, entry))
    return problems


def frame_to_seconds(timing, recording_id, frame):
    entry = (timing or {}).get("recordings", {}).get(recording_id)
    if entry is None:
        raise ValueError(f"no timing entry for recording {recording_id}; no default FPS is assumed")
    problems = _timing_entry_problems(recording_id, entry)
    if problems:
        raise ValueError(problems[0])
    if "fps" in entry:
        seconds = (frame - entry.get("frame_offset", 0)) / entry["fps"]
        if seconds < 0:
            raise ValueError(f"frame {frame} is before frame_offset for {recording_id}")
        return seconds
    points = _map_points(entry)
    if not points[0][0] <= frame <= points[-1][0]:
        raise ValueError(f"frame {frame} is outside the frame_map range for {recording_id}")
    for (f0, t0), (f1, t1) in zip(points, points[1:]):
        if f0 <= frame <= f1:
            return t0 + (t1 - t0) * (frame - f0) / (f1 - f0)
    raise AssertionError("unreachable")


def _recall(verdict, policy, eligible, label, errors):
    if "recalled_case_ids" in verdict:
        source, ids = "verdict", verdict["recalled_case_ids"]
    elif policy is not None:
        source = "policy"
        ids = [c for c in policy["supporting_dev_case_ids"]
               if c not in eligible or eligible[c]["component_id"] == verdict["component_id"]]
    else:
        source, ids = "none", []
    shown, refused = [], []
    for case_id in ids:
        if case_id in eligible:
            shown.append(eligible[case_id])
        else:
            refused.append(case_id)
            errors.append(f"{label}: recalled case {case_id} is not an eligible development case and was not shown")
    return {"source": source, "cases": shown[:MAX_CLIPS], "overflow": max(0, len(shown) - MAX_CLIPS),
            "refused": refused}


def _lanes(pairs, seconds, errors):
    lanes, last_seq = {}, None
    for label, event in sorted(pairs, key=lambda pair: pair[1]["seq"]):
        if event["seq"] == last_seq:
            errors.append(f"{label}: duplicate seq {event['seq']} in run {event['run_id']}")
            continue
        last_seq = event["seq"]
        issue_id, kind = event["issue_id"], event["event"]
        lane = lanes.get(issue_id)
        t = seconds(event["cursor_frame"], label)
        if kind == "opened":
            if lane is not None:
                errors.append(f"{label}: issue {issue_id} was opened twice")
                continue
            lanes[issue_id] = {"issue_id": issue_id, "description": event["description"],
                               "since": event["checkpoint_id"], "start": t, "end": None, "resolved": False,
                               "resolved_at": None, "resolved_by": [], "carried": []}
        elif lane is None or lane["resolved"]:
            errors.append(f"{label}: {kind} event for issue {issue_id}, which is not open")
        elif kind == "carried":
            if t is not None:
                lane["carried"].append(t)
        elif not event["evidence_ids"]:
            errors.append(f"{label}: resolution of {issue_id} cites no evidence, so the issue stays open")
        else:
            lane.update(resolved=True, end=t, resolved_at=event["checkpoint_id"], resolved_by=event["evidence_ids"])
    return list(lanes.values())


def build(verdicts, evidence, evaluation, cases, policies, timing, events, snapshot, load_errors=()):
    """Return (html, errors, notices, is_mock)."""
    errors, notices, kinds = list(load_errors), [], set()

    def valid(kind, record, label):
        if isinstance(record, dict) and record.get("source_kind") in SOURCE_KINDS:
            kinds.add(record["source_kind"])
        problems = validate(kind, record, label)
        errors.extend(problems)
        return not problems

    evidence_by_id = {}
    for i, record in enumerate(evidence or [], 1):
        if valid("evidence", record, f"evidence record {i}"):
            known = evidence_by_id.setdefault(record["evidence_id"], record)
            if known != record:
                errors.append(f"evidence record {i}: evidence_id {record['evidence_id']} reused with different content")

    policy = {}
    for arm in ARMS:
        record = (policies or {}).get(arm)
        if record is None:
            errors.append(f"policy_{arm}: no policy record supplied")
        policy[arm] = record if record is not None and valid("policy", record, f"policy_{arm}") else None

    recording, rows_by_arm = None, {}
    for arm in ARMS:
        rows, seen, run_id = [], set(), None
        for i, record in enumerate((verdicts or {}).get(arm) or [], 1):
            label = f"verdicts_{arm} record {i}"
            if not valid("verdict", record, label):
                continue
            recording = recording or record["recording_id"]
            if record["recording_id"] != recording:
                errors.append(f"{label}: recording {record['recording_id']} differs from {recording}; one recording per replay")
                continue
            run_id = run_id or record["run_id"]
            if record["run_id"] != run_id:
                errors.append(f"{label}: run_id {record['run_id']} differs from {run_id} in the same arm")
                continue
            if record["checkpoint_id"] in seen:
                errors.append(f"{label}: duplicate verdict for {record['checkpoint_id']}")
                continue
            seen.add(record["checkpoint_id"])
            if policy[arm] and record["policy_id"] != policy[arm]["policy_id"]:
                errors.append(f"{label}: policy_id {record['policy_id']} does not match policy_{arm} ({policy[arm]['policy_id']})")
            rows.append(record)
        rows_by_arm[arm] = sorted(rows, key=lambda r: r["cursor_frame"])
        if not rows:
            notices.append(f"No valid {arm} verdicts, so the {arm} view is disabled.")

    eligible = {}
    for i, record in enumerate(cases or [], 1):
        if not valid("case", record, f"cases record {i}"):
            continue
        if record["split"] != "development":
            notices.append(f"Case {record['case_id']} excluded from recall: split is {record['split']!r}; only development cases may be recalled.")
        elif record["recording_id"] == recording:
            notices.append(f"Case {record['case_id']} excluded from recall: it comes from the recording being replayed.")
        else:
            eligible[record["case_id"]] = record

    timing_js, video_path, duration, timing_ready = None, None, None, False
    if timing is None:
        errors.append("timing: no timing config supplied, so checkpoints cannot be placed on the video")
    elif valid("timing", timing, "timing") and recording:
        entry = timing["recordings"].get(recording)
        if entry is None:
            errors.append(f"timing: no entry for recording {recording}; no default FPS is assumed")
        else:
            timing_ready = True
            video_path, duration = entry.get("video_path"), entry.get("duration_s")
            timing_js = ({"mode": "fps", "fps": entry["fps"], "offset": entry.get("frame_offset", 0)}
                         if "fps" in entry else {"mode": "map", "points": _map_points(entry)})

    def seconds(frame, label):
        if not timing_ready:
            return None
        try:
            return frame_to_seconds(timing, recording, frame)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            return None

    events_by_run = {}
    for i, record in enumerate(events or [], 1):
        label = f"issue_events record {i}"
        if valid("issue_event", record, label):
            events_by_run.setdefault(record["run_id"], []).append((label, record))

    arms, models = {}, set()
    for arm in ARMS:
        rows = rows_by_arm[arm]
        if not rows:
            arms[arm] = None
            continue
        run_id = rows[0]["run_id"]
        lanes = _lanes(events_by_run.pop(run_id, []), seconds, errors)
        info = {lane["issue_id"]: lane for lane in lanes}
        checkpoints = []
        for record in rows:
            label = f"{arm} {record['checkpoint_id']}"
            models.add(f"verifier {record['model_id']}")
            seen_evidence = []
            for evidence_id in record["evidence_ids"]:
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    errors.append(f"{label}: evidence {evidence_id} not found in saved evidence")
                    seen_evidence.append({"evidence_id": evidence_id, "missing": True})
                    continue
                if (item["recording_id"], item["cursor_frame"]) != (record["recording_id"], record["cursor_frame"]):
                    errors.append(f"{label}: evidence {evidence_id} belongs to {item['recording_id']} frame {item['cursor_frame']}, not this checkpoint's frame {record['cursor_frame']}")
                provider = item["provider"]
                models.add(f"evidence {provider.get('model_id', '?')}/{provider.get('prompt_version', '?')}")
                seen_evidence.append({"evidence_id": evidence_id, "missing": False, "frame_ids": item["frame_ids"],
                                      "observations": item["observations"], "uncertainties": item["uncertainties"]})
            open_issues = [{"issue_id": issue_id, "since": info.get(issue_id, {}).get("since"),
                            "description": info.get(issue_id, {}).get("description")}
                           for issue_id in record["open_issue_ids"]]
            checkpoints.append({
                "checkpoint_id": record["checkpoint_id"], "cursor_frame": record["cursor_frame"],
                "t": seconds(record["cursor_frame"], label), "component_id": record["component_id"],
                "verdict": record["verdict"], "reason": record["reason"], "evidence": seen_evidence,
                "open_issues": open_issues, "recalled": _recall(record, policy[arm], eligible, label, errors)})
        p = policy[arm]
        arms[arm] = {"run_id": run_id, "policy_id": rows[0]["policy_id"], "checkpoints": checkpoints, "issues": lanes,
                     "policy": None if p is None else {k: p[k] for k in (
                         "policy_id", "parent_policy_id", "status", "checklist", "supporting_dev_case_ids")}}
    for run_id in events_by_run:
        notices.append(f"Issue events for run {run_id} match no replayed arm and were ignored.")

    evaluation = evaluation if evaluation is not None and valid("evaluation", evaluation, "evaluation") else None
    if evaluation is not None:
        for arm, key in (("baseline", "baseline_policy_id"), ("candidate", "candidate_policy_id")):
            if evaluation[key] and arms.get(arm) and evaluation[key] != arms[arm]["policy_id"]:
                notices.append(f"Evaluation {key} {evaluation[key]} differs from the replayed {arm} policy {arms[arm]['policy_id']}.")
    snapshot = snapshot if snapshot is not None and valid("snapshot", snapshot, "session_snapshot") else None

    times = [cp["t"] for arm in arms.values() if arm for cp in arm["checkpoints"] if cp["t"] is not None]
    if duration is None:
        duration = (max(times) + 5) if times else 10
    payload = {"recording_id": recording, "video_path": video_path, "duration": duration, "timing": timing_js,
               "backend": snapshot["backend"] if snapshot else None, "is_mock": "mock" in kinds, "arms": arms}
    page = _page(payload, arms, sorted(models), kinds, evaluation, snapshot, errors, notices)
    return page, errors, notices, "mock" in kinds


def render_report(verdicts, evidence, evaluation, cases, policies, timing, events, snapshot, load_errors=()):
    return build(verdicts, evidence, evaluation, cases, policies, timing, events, snapshot, load_errors)[0]


def _e(value):
    return html.escape("—" if value is None else str(value))


def _txt(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "none"
    return value


def _script_json(value):
    return (json.dumps(value).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e"))


def _metric(value):
    if value == "unavailable":
        return '<span class="muted">unavailable</span>'
    return _e(value)


HEADLINES = {
    "accepted": ("Candidate accepted", "Per the saved evaluation, the candidate improved the false-correct count without lowering state accuracy."),
    "rejected": ("Candidate rejected", "The baseline policy is retained."),
    "inconclusive": ("Inconclusive", "Too little reference data to decide; no policy change."),
    "no_proposal": ("No proposal", "No eligible development miss produced a candidate rule."),
}


def _evaluation_bar(ev):
    if ev is None:
        return '<div class="evalbar"><span class="decision none">NO EVALUATION</span><span class="muted">No valid evaluation record was saved.</span></div>'
    title, _ = HEADLINES[ev["decision"]]
    return (f'<div class="evalbar"><span class="decision {ev["decision"]}">{_e(title)}</span>'
            f'<span>{_e(ev["reason"])}</span><span class="muted mono">scored {_e(ev["scored"])} · '
            f'excluded {_e(ev["excluded"])} · same for both arms</span></div>')


def _evaluation_html(ev):
    if ev is None:
        return '<p class="empty">No valid evaluation record was saved, so there is nothing to report.</p>'
    title, subtitle = HEADLINES[ev["decision"]]
    rows = []
    for key, name in METRICS:
        base = _metric(ev["baseline"][key])
        cand = _metric(ev["candidate"][key]) if ev["candidate"] is not None else '<span class="muted">—</span>'
        rows.append(f'<tr><th>{_e(name)}</th><td class="mono">{base}</td><td class="mono">{cand}</td></tr>')
    return (f'<div class="evalhead"><span class="decision {ev["decision"]}">{_e(title)}</span>'
            f'<p>{_e(subtitle)}</p></div>'
            f'<h3>Saved reason</h3><p class="reason">{_e(ev["reason"])}</p>'
            f'<table class="m"><thead><tr><th></th><th>Baseline<br><code>{_e(ev["baseline_policy_id"])}</code></th>'
            f'<th>Candidate<br><code>{_e(ev["candidate_policy_id"])}</code></th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            f'<dl class="kv"><dt>Scored</dt><dd class="mono">{_e(ev["scored"])} (identical for both arms)</dd>'
            f'<dt>Excluded</dt><dd class="mono">{_e(ev["excluded"])}</dd>'
            f'<dt>Recordings</dt><dd class="mono">{_e(_txt(ev["recording_ids"]))}</dd></dl>')


def _memory_html(snap):
    if snap is None:
        return '<p class="empty">No valid Atlas session snapshot was saved.</p>'
    key = snap["session_key"]
    issues = "".join(
        f'<li><code>{_e(item.get("issue_id"))}</code> open since <code>{_e(item.get("since_checkpoint"))}</code>'
        f'<br><span class="muted">{_e(item.get("description"))}</span></li>' if isinstance(item, dict)
        else f'<li>{_e(item)}</li>' for item in snap["open_issues"]) or '<li class="muted">No open issues</li>'
    out = (f'<h3>Session document</h3><dl class="kv">'
           f'<dt>Backend</dt><dd><span class="dot {_e(snap["backend"])}"></span>{"Atlas" if snap["backend"] == "atlas" else "Local SQLite"}</dd>'
           f'<dt>Run</dt><dd class="mono">{_e(key.get("run_id"))}</dd>'
           f'<dt>Recording</dt><dd class="mono">{_e(key.get("recording_id"))}</dd>'
           f'<dt>Policy</dt><dd class="mono">{_e(key.get("policy_id"))}</dd>'
           f'<dt>Last committed</dt><dd class="mono">{_e(snap["last_committed_checkpoint"])}</dd></dl>'
           f'<h3>Open issues</h3><ul class="issues">{issues}</ul>')
    restart = snap.get("restart")
    before = restart.get("before") if isinstance(restart, dict) else None
    after = restart.get("after") if isinstance(restart, dict) else None
    if not (isinstance(before, dict) and isinstance(after, dict)):
        return out + '<h3>Restart</h3><p class="empty">No restart record was saved.</p>'
    fields = (("process", "Process"), ("last_committed_checkpoint", "Last committed"),
              ("open_issue_ids", "Open issues"), ("verdict_count", "Verdicts saved"))
    rows = "".join(f'<tr><th>{name}</th><td class="mono">{_e(_txt(before.get(k)))}</td>'
                   f'<td class="mono">{_e(_txt(after.get(k)))}</td></tr>' for k, name in fields)
    same_issues = sorted(map(str, before.get("open_issue_ids") or [])) == sorted(map(str, after.get("open_issue_ids") or []))
    checks = ((same_issues, "Same open issues after restart"),
              (before.get("last_committed_checkpoint") == after.get("last_committed_checkpoint"),
               "Resumed from the same committed checkpoint"),
              (before.get("verdict_count") == after.get("verdict_count"), "No duplicate verdicts (count unchanged)"))
    items = "".join(f'<li class="{"ok" if good else "bad"}">{"✓" if good else "✗"} {text}</li>' for good, text in checks)
    return (out + f'<h3>Restart · before / after</h3><table class="m"><thead><tr><th></th><th>Before</th>'
            f'<th>After</th></tr></thead><tbody>{rows}</tbody></table><ul class="checks">{items}</ul>')


def _problems_html(errors, notices):
    if not errors and not notices:
        return ""
    parts = ['<section class="problems" aria-label="Validation problems">']
    if errors:
        parts.append(f'<div class="errs"><h2>{len(errors)} error{"s" if len(errors) != 1 else ""} in saved records</h2><ul>'
                     + "".join(f"<li>{_e(e)}</li>" for e in errors) + "</ul></div>")
    if notices:
        parts.append('<div class="notes"><h2>Notices</h2><ul>' + "".join(f"<li>{_e(n)}</li>" for n in notices) + "</ul></div>")
    return "".join(parts) + "</section>"


def _font_face():
    if not FONT_FILE.exists():
        return ""
    data = base64.b64encode(FONT_FILE.read_bytes()).decode("ascii")
    return ('@font-face{font-family:"Red Hat Display";font-style:normal;font-weight:400 700;font-display:swap;'
            f'src:url(data:font/woff2;base64,{data}) format("woff2")}}')


def _page(payload, arms, models, kinds, ev, snap, errors, notices):
    if "mock" in kinds:
        badge = '<span class="badge mock" title="At least one record has source_kind mock">MOCK DATA</span>'
    elif kinds:
        badge = f'<span class="badge real" title="No mock records loaded">REAL · {_e(", ".join(sorted(kinds)))}</span>'
    else:
        badge = '<span class="badge none">NO DATA</span>'
    backend = snap["backend"] if snap else None
    backend_html = (f'<span class="dot {backend}"></span>{"Atlas" if backend == "atlas" else "Local SQLite"}'
                    if backend else '<span class="dot"></span>unknown')
    buttons = "".join(
        f'<button type="button" data-arm="{arm}"{"" if arms.get(arm) else " disabled"}>{arm.title()}'
        f'<small>{_e(arms[arm]["policy_id"]) if arms.get(arm) else "no verdicts"}</small></button>' for arm in ARMS)
    body = BODY
    for key, value in (("{{RECORDING}}", _e(payload["recording_id"])), ("{{MODELS}}", _e(" · ".join(models) or None)),
                       ("{{BACKEND}}", backend_html), ("{{ARM_BUTTONS}}", buttons), ("{{BADGE}}", badge),
                       ("{{PROBLEMS}}", _problems_html(errors, notices)), ("{{EVAL_BAR}}", _evaluation_bar(ev)),
                       ("{{EVAL}}", _evaluation_html(ev)), ("{{MEMORY}}", _memory_html(snap)),
                       ("{{PAYLOAD}}", _script_json(payload))):
        body = body.replace(key, value)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Decision Replay</title><style>' + _font_face() + CSS + '</style></head><body>' + body
            + '<script>' + JS + '</script></body></html>')


CSS = """
:root{--bg:#FFFFFF;--surface:#ECEBE8;--surface-grey:#E4E4E0;--surface-dp:#F8F8F8;--white:#FFFFFF;--stroke:#F5F5FA;
--dark:#2D2D2D;--dark-2:#434343;--dark-3:#2C2C2C;--divider-dark:#5A5A5A;
--text:#29303D;--text-muted:rgba(0,0,0,.4);--white-50:rgba(255,255,255,.5);
--accent:#E6FB2D;--accent-dark:#D9EE1C;--accent-green:#E3EF7A;
--v-correct:#4E8F5A;--v-correct-fill:#E1EEDD;--v-incorrect:#C05A5A;--v-incorrect-fill:#F6E1E1;
--v-notdone:#7D7D78;--v-notdone-fill:#E4E4E0;--v-insuff:#B7791F;--v-insuff-fill:#F6EBCF;
--mock:#6B4BC4;--mock-fill:#ECE6FA;--error:#C05A5A;
--fz-xs:.75rem;--fz-sm:.875rem;--fz-md:1rem;--fz-lg:1.5rem;--fw-400:400;--fw-500:500;--fw-600:600;
--btn-radius:1.875rem;--card-radius:1.5rem;--pad:2.5rem;
--sans:"Red Hat Display",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:var(--fz-md);line-height:1.5;-webkit-font-smoothing:antialiased}
.mono,code{font-family:var(--mono);font-size:.8rem}.muted{color:var(--text-muted)}
button{font:inherit;color:inherit}:focus-visible{outline:none;box-shadow:0 0 0 2px var(--accent-dark)}
.top{display:flex;align-items:center;flex-wrap:wrap;gap:12px 28px;padding:1.25rem var(--pad);background:var(--white);position:sticky;top:0;z-index:5;border-bottom:1px solid var(--stroke)}
.brand{font-weight:700;font-size:1.2rem;display:flex;align-items:center;gap:10px;white-space:nowrap;letter-spacing:-.01em;text-transform:uppercase}
.brand::before{content:"";width:10px;height:10px;background:var(--dark);border-radius:50%;box-shadow:0 0 0 3px var(--accent)}
.meta{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:var(--fz-sm);min-width:0}
.meta b{font-weight:var(--fw-400);color:var(--text-muted);margin-right:6px}
.meta code{color:var(--text)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--v-notdone);vertical-align:0}
.dot.atlas{background:var(--v-correct)}.dot.local{background:var(--v-insuff)}
.spacer{flex:1}
.versions{display:inline-flex;gap:8px;margin-left:auto}
.versions button{background:var(--surface-dp);border:1px solid transparent;border-radius:var(--btn-radius);padding:.55rem 1.1rem;cursor:pointer;color:var(--text-muted);display:flex;align-items:baseline;gap:8px;line-height:100%;font-size:var(--fz-sm);font-weight:var(--fw-600);transition:background-color .2s,color .2s,border-color .2s}
.versions button small{font-family:var(--mono);font-size:10.5px;font-weight:400;opacity:.8}
.versions button:hover{border-color:var(--accent-green)}
.versions button[aria-pressed="true"]{background:var(--dark);color:var(--white)}
.versions button:disabled{opacity:.4;cursor:not-allowed}
.toggle{display:flex;align-items:center;gap:6px;font-size:var(--fz-sm);color:var(--text-muted);cursor:pointer;white-space:nowrap}
.toggle input{accent-color:var(--dark)}
.badge{font:600 11px/1 var(--mono);letter-spacing:.5px;padding:.6rem .9rem;border-radius:var(--btn-radius);white-space:nowrap}
.badge.mock{color:var(--mock);background:var(--mock-fill)}
.badge.real{color:var(--v-correct);background:var(--v-correct-fill)}
.badge.none{color:var(--error);background:var(--v-incorrect-fill)}
.problems{margin:1rem var(--pad) 0;padding:1rem 1.5rem;background:var(--surface-dp);border-radius:1.25rem;font-size:var(--fz-sm)}
.problems h2{font-size:var(--fz-sm);font-weight:var(--fw-600);margin:0 0 4px}
.problems ul{margin:0 0 4px;padding-left:18px}
.problems .errs{color:var(--error);background:var(--v-incorrect-fill);border-radius:1rem;padding:.6rem 1rem;margin-bottom:6px}
.problems .notes{color:var(--text-muted)}
.main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:1.25rem;padding:1.25rem var(--pad) 2.5rem;align-items:start}
.stage{display:flex;flex-direction:column;gap:12px;min-width:0;background:var(--surface);border-radius:var(--card-radius);padding:1.25rem}
.screen-wrap{position:relative}
.screen{position:relative;aspect-ratio:16/9;background:var(--dark-3);border-radius:1.25rem;overflow:hidden}
.screen video,.screen canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
.rec-label{position:absolute;top:14px;right:14px;font:600 10.5px/1 var(--mono);letter-spacing:.5px;text-transform:uppercase;padding:.45rem .75rem;border-radius:var(--btn-radius);background:var(--white);color:var(--text);z-index:1}
.rec-label::before{content:"\\25CF";color:var(--v-incorrect);margin-right:6px}
.rec-label.mock{color:var(--mock);background:var(--mock-fill)}.rec-label.mock::before{color:var(--mock)}
.card{position:absolute;left:16px;bottom:16px;width:min(560px,calc(100% - 32px));max-height:calc(100% - 32px);overflow:auto;background:var(--white);border-radius:var(--card-radius);padding:1.1rem 1.25rem;box-shadow:0 16px 40px rgba(0,0,0,.28);opacity:0;transform:translateY(16px);transition:opacity .2s,transform .2s;pointer-events:none;z-index:2}
.card.show{opacity:1;transform:none;pointer-events:auto}
.card-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:8px}
.card-title{font-size:var(--fz-md);font-weight:var(--fw-600);line-height:120%}.card-sub{font-size:var(--fz-xs);color:var(--text-muted);margin-top:2px}
.arm-tag{font:600 10.5px/1 var(--mono);padding:.4rem .7rem;border-radius:var(--btn-radius);background:var(--accent);color:var(--dark);white-space:nowrap}
.crow{display:grid;grid-template-columns:100px 1fr;gap:12px;padding:.45rem 0;border-top:1px solid var(--surface-grey);font-size:var(--fz-sm);line-height:150%}
.clabel{font-size:var(--fz-xs);font-weight:var(--fw-500);color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;padding-top:2px}
.cval ul{margin:0;padding-left:16px}.cval li{margin:1px 0}.cval .unc li{color:var(--v-insuff)}
.src{font-size:var(--fz-xs);color:var(--text-muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:2px}
.chip{background:var(--surface-dp);border:1px solid var(--surface-grey);border-radius:var(--btn-radius);padding:.15rem .6rem;font-family:var(--mono);font-size:11px;color:var(--text)}
.err{color:var(--error)}
.card-foot{display:flex;justify-content:space-between;align-items:center;margin-top:.5rem;gap:10px;font-size:var(--fz-sm);position:sticky;bottom:-1.1rem;background:var(--white);padding:.6rem 0 .1rem;border-top:1px solid var(--surface-grey)}
.btn{background:transparent;border:1px solid var(--dark-2);border-radius:var(--btn-radius);padding:.6rem 1rem;cursor:pointer;color:var(--text);font-weight:var(--fw-600);font-size:var(--fz-sm);line-height:100%;transition:border-color .2s,background-color .2s,color .2s}
.btn:hover{border-color:#E6FF00}
.btn:active{border-color:var(--accent-green)}
.btn.primary{background:var(--dark);border-color:var(--dark);color:var(--white);padding:.81rem 1.5rem}
.btn.primary:hover{background:var(--accent);border-color:var(--accent);color:var(--text)}
.btn.primary:active{background:var(--accent-dark)}
.btn.icon{width:40px;height:40px;padding:0;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}
#play.btn.icon{background:var(--dark);border-color:var(--dark);color:var(--white)}
#play.btn.icon:hover{background:var(--accent);border-color:var(--accent);color:var(--text)}
.pill{display:inline-block;font:600 10.5px/1 var(--mono);letter-spacing:.3px;padding:.35rem .65rem;border-radius:var(--btn-radius);white-space:nowrap}
.pill.v-correct{color:var(--v-correct);background:var(--v-correct-fill)}
.pill.v-incorrect{color:var(--v-incorrect);background:var(--v-incorrect-fill)}
.pill.v-not_completed{color:var(--v-notdone);background:var(--white);box-shadow:inset 0 0 0 1px var(--surface-grey)}
.pill.v-insufficient_evidence{color:var(--v-insuff);background:var(--v-insuff-fill)}
.transport{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:4px}
#readout{margin-left:6px}
.legend{display:flex;gap:16px;font-size:var(--fz-sm);color:var(--text);flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;margin-right:6px;border-radius:50%;vertical-align:0}
.track{position:relative;height:30px;cursor:pointer;touch-action:none}
.rail{position:absolute;left:0;right:0;top:10px;height:10px;border-radius:var(--btn-radius);background:var(--white);overflow:hidden}
.played{height:100%;background:var(--accent);width:0}
.playhead{position:absolute;top:0;bottom:0;width:2px;border-radius:2px;background:var(--dark);margin-left:-1px;pointer-events:none}
.marker{position:absolute;top:8px;width:14px;height:14px;margin-left:-7px;transform:rotate(45deg);border:2px solid var(--white);border-radius:3px;padding:0;cursor:pointer;background:var(--v-notdone)}
.v-correct.marker,.legend .v-correct{background:var(--v-correct)}
.v-incorrect.marker,.legend .v-incorrect{background:var(--v-incorrect)}
.v-not_completed.marker,.legend .v-not_completed{background:var(--v-notdone)}
.v-insufficient_evidence.marker,.legend .v-insufficient_evidence{background:var(--v-insuff)}
.marker.active{box-shadow:0 0 0 2px var(--dark)}
.axis{position:relative;height:20px;border-bottom:1px solid var(--surface-grey)}
.axis span{position:absolute;top:0;transform:translateX(-50%);font:11px var(--mono);color:var(--text-muted);white-space:nowrap}
.axis span:first-child{transform:none}
.cap{font-size:var(--fz-sm);font-weight:var(--fw-500);color:var(--text)}
.lanes{display:flex;flex-direction:column}
.lane{position:relative;height:32px;border-bottom:1px solid var(--surface-grey);background:repeating-linear-gradient(90deg,transparent 0 calc(10% - 1px),var(--surface-grey) calc(10% - 1px) 10%)}
.bar{position:absolute;top:6px;height:20px;border-radius:var(--btn-radius);background:var(--accent-green);min-width:20px;display:flex;align-items:center}
.bar::before{content:"";position:absolute;left:0;top:0;width:20px;height:20px;border-radius:50%;background:var(--dark)}
.bar.resolved::after{content:"";position:absolute;right:0;top:0;width:20px;height:20px;border-radius:50%;background:var(--v-correct)}
.bar.open::after{content:"\\2192";position:absolute;right:8px;top:0;font-size:13px;line-height:20px;color:var(--dark);font-weight:700}
.bar-label{position:relative;font:600 11px/20px var(--mono);color:var(--dark);padding-left:26px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;padding-right:22px}
.tick{position:absolute;top:6px;width:2px;height:8px;border-radius:1px;background:rgba(45,45,45,.35)}
.evalbar{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-top:8px;padding:1rem 1.25rem;background:var(--dark-2);color:var(--white);border-radius:1.25rem;font-size:var(--fz-sm)}
.evalbar .muted{color:var(--white-50)}
.decision{font:600 11px/1 var(--mono);letter-spacing:.3px;text-transform:uppercase;padding:.45rem .8rem;border-radius:var(--btn-radius);white-space:nowrap}
.decision.accepted{color:var(--dark);background:var(--accent)}
.decision.rejected{color:var(--white);background:var(--v-incorrect)}
.decision.inconclusive{color:var(--dark);background:var(--v-insuff-fill)}
.decision.no_proposal,.decision.none{color:var(--dark);background:var(--surface-grey)}
.side{background:var(--surface);border-radius:var(--card-radius);display:flex;flex-direction:column;min-width:0;overflow:hidden;padding-bottom:.5rem}
.tabs{display:flex;gap:8px;margin:1rem 1rem .25rem}
.tabs button{flex:1;background:var(--surface-dp);border:1px solid transparent;border-radius:var(--btn-radius);padding:.6rem .5rem .55rem;cursor:pointer;color:var(--text-muted);font-size:var(--fz-sm);font-weight:var(--fw-600);line-height:100%;transition:border-color .2s}
.tabs button:hover{border-color:var(--accent-green)}
.tabs button[aria-selected="true"]{color:var(--white);background:var(--dark)}
.tabpanel{padding:0;overflow:auto;max-height:48vh}
.tabpanel:not(#tab-decisions){padding:.25rem 1.25rem 1rem}
#tab-decisions{max-height:none;padding:.5rem 1rem 0}
#decisions{display:flex;flex-direction:column;gap:6px}
.drow{display:flex;align-items:center;gap:10px;width:100%;text-align:left;background:var(--white);border:1px solid transparent;border-radius:1rem;padding:.6rem .8rem;cursor:pointer;font-size:var(--fz-sm);transition:border-color .2s}
.drow:hover{border-color:var(--accent-green)}
.drow.active{border-color:var(--dark-2)}
.drow .chip{background:var(--surface-dp);border-color:transparent}
.drow.active .chip{background:var(--accent);color:var(--dark)}
.drow .comp{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-muted)}
.recalls{font:600 11px var(--mono);color:var(--dark)}
.mono-panel{margin:.75rem 0 0;padding:1rem 1.1rem;border-radius:1.25rem;background:var(--dark-3);color:var(--white)}
.mono-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:.6rem}
.mono-panel .cap{color:var(--white)}
.mono-note{font-size:11px;color:var(--white-50)}
#monologue{font:12px/1.6 var(--mono);color:var(--white);min-height:60px;max-height:260px;overflow:auto;padding-right:4px}
#monologue .empty{color:var(--white-50)}
.mstep{margin-bottom:6px}
.mhead{display:flex;gap:8px;align-items:baseline}
.mglyph{color:var(--accent);width:12px;flex:none;text-align:center}
.mglyph.done{color:var(--accent-green)}
.mverb{color:var(--accent);font-weight:600}
.mverb.done{color:var(--white)}
.mline{position:relative;padding-left:22px;color:var(--white-50);white-space:pre-wrap;word-break:break-word}
.mline.first::before{content:"\\23BF";position:absolute;left:5px;color:var(--divider-dark)}
.mline.unc{color:#E9C46A}.mline.err{color:#F08A8A}.mline.say{color:rgba(255,255,255,.88)}
.caret{display:inline-block;width:6px;height:12px;margin-left:1px;background:var(--accent);vertical-align:-1px;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.cases-panel{padding:1.1rem 1rem .5rem;flex:1;min-height:200px}
.panel-head{display:flex;justify-content:space-between;gap:8px;margin:0 .25rem .6rem}
.case{display:grid;grid-template-columns:136px 1fr;gap:12px;padding:.75rem;border-radius:1.25rem;background:var(--white);margin-bottom:8px}
.case-media{position:relative;aspect-ratio:16/9;background:var(--dark-3);border-radius:.9rem;overflow:hidden}
.case-media video{width:100%;height:100%;object-fit:cover;display:block}
.case-media.missing{background:repeating-linear-gradient(45deg,var(--surface-dp) 0 8px,var(--surface-grey) 8px 16px);display:flex;align-items:flex-end;padding:6px}
.case-media .rec-label{top:5px;right:5px;font-size:8.5px;padding:.25rem .45rem}
.missing-text{font:10px var(--mono);color:var(--text-muted);word-break:break-all}
.case-head{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:var(--fz-sm)}
.tag{font-size:10.5px;font-weight:var(--fw-600);padding:.15rem .55rem;border-radius:var(--btn-radius);color:var(--dark);background:var(--accent)}
.case-line{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:var(--fz-xs);color:var(--text-muted);margin:6px 0}
.case-sum{margin:2px 0;font-size:var(--fz-sm);line-height:150%}
.empty{color:var(--text-muted);margin:.4rem .25rem;font-size:var(--fz-sm)}
.side h3{font-size:var(--fz-sm);font-weight:var(--fw-600);color:var(--text);margin:1rem 0 .5rem}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:var(--fz-sm);margin:0 0 8px}.kv dt{color:var(--text-muted)}.kv dd{margin:0}
.issues{margin:0;padding-left:18px;font-size:var(--fz-sm)}.issues li{margin-bottom:4px}
table.m{width:100%;border-collapse:separate;border-spacing:0;font-size:var(--fz-sm);background:var(--white);border-radius:1rem;overflow:hidden}
table.m th,table.m td{padding:.5rem .65rem;border-bottom:1px solid var(--surface);text-align:left;vertical-align:top}
table.m tr:last-child th,table.m tr:last-child td{border-bottom:0}
table.m thead th{color:var(--text);font-weight:var(--fw-600)}
table.m tbody th{color:var(--text-muted);font-weight:var(--fw-400)}
.checks{list-style:none;padding:0;margin:.6rem 0;font-size:var(--fz-sm)}.ok{color:var(--v-correct)}.bad{color:var(--error)}
.evalhead{display:flex;flex-direction:column;gap:6px;align-items:flex-start;margin:1rem 0 .25rem}.evalhead p{margin:0;font-size:var(--fz-sm)}
.reason{margin:0 0 .75rem;font-size:var(--fz-sm)}
.foot{margin:0;padding:3rem var(--pad) 2rem;color:var(--white-50);font-size:var(--fz-sm);background:var(--dark-2);border-radius:1.5rem 1.5rem 0 0}
#wire{position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:4}
#wire path{fill:none;stroke:var(--dark);stroke-width:1.5;stroke-dasharray:5 5}
#wire circle{fill:var(--dark);stroke:var(--accent);stroke-width:2}
@media (max-width:1124px){:root{--pad:1.25rem}.main{grid-template-columns:minmax(0,1fr)}.tabpanel{max-height:none}}
@media (max-width:767px){:root{--pad:1rem}.card{position:static;width:auto;max-height:none;margin-top:10px;display:none;opacity:1;transform:none;box-shadow:none}.card.show{display:block}.case{grid-template-columns:1fr}.crow{grid-template-columns:1fr;gap:2px}.stage{padding:.75rem}}
"""

BODY = """
<header class="top">
  <div class="brand">Decision Replay</div>
  <div class="meta">
    <span><b>recording</b><code>{{RECORDING}}</code></span>
    <span><b>run</b><code id="h-run">—</code></span>
    <span><b>policy</b><code id="h-policy">—</code></span>
    <span><b>model</b><code>{{MODELS}}</code></span>
    <span><b>backend</b>{{BACKEND}}</span>
  </div>
  <div class="spacer"></div>
  <div class="versions" role="group" aria-label="Policy version">{{ARM_BUTTONS}}</div>
  <label class="toggle"><input type="checkbox" id="auto" checked> Auto-resume 4s</label>
  {{BADGE}}
</header>
{{PROBLEMS}}
<main class="main">
  <section class="stage">
    <div class="screen-wrap">
      <div class="screen">
        <video id="main-video" playsinline preload="metadata" hidden></video>
        <canvas id="mock-canvas" width="1280" height="720" aria-label="Mock placeholder clock"></canvas>
        <span class="rec-label" id="rec-label">Recorded</span>
      </div>
      <aside class="card" id="card" aria-live="polite">
        <div class="card-head"><div><div class="card-title mono" id="card-title"></div><div class="card-sub" id="card-sub"></div></div><span class="arm-tag" id="card-arm"></span></div>
        <div id="card-body"></div>
        <div class="card-foot"><span class="muted" id="card-auto"></span><button type="button" class="btn primary" id="continue">Continue &#9654;</button></div>
      </aside>
    </div>
    <div class="transport">
      <button type="button" class="btn icon" id="prev-cp" aria-label="Previous decision">&#9198;</button>
      <button type="button" class="btn icon" id="play" aria-label="Play">&#9654;</button>
      <button type="button" class="btn icon" id="next-cp" aria-label="Next decision">&#9197;</button>
      <span class="mono" id="readout"></span>
      <span class="spacer"></span>
      <span class="legend"><span><i class="v-correct"></i>correct</span><span><i class="v-incorrect"></i>incorrect</span><span><i class="v-not_completed"></i>not completed</span><span><i class="v-insufficient_evidence"></i>insufficient evidence</span></span>
    </div>
    <div class="track" id="track" aria-label="Timeline"><div class="rail"><div class="played" id="played"></div></div><div id="markers"></div><div class="playhead" id="playhead"></div></div>
    <div class="axis" id="axis" aria-hidden="true"></div>
    <div class="cap">Issues · opened &rarr; resolved</div>
    <div class="lanes" id="lanes"></div>
    {{EVAL_BAR}}
  </section>
  <aside class="side">
    <nav class="tabs" role="tablist">
      <button type="button" role="tab" data-tab="decisions" aria-selected="true">Decisions</button>
      <button type="button" role="tab" data-tab="memory" aria-selected="false">Memory</button>
      <button type="button" role="tab" data-tab="eval" aria-selected="false">Evaluation</button>
    </nav>
    <div class="tabpanel" id="tab-decisions" role="tabpanel"><div id="decisions"></div>
      <section class="mono-panel">
        <div class="mono-head"><span class="cap">Verifier reasoning trace</span><span class="mono-note">chained from saved evidence, memory and recalled cases · templated text, not model-generated</span></div>
        <div id="monologue"></div>
      </section>
    </div>
    <div class="tabpanel" id="tab-memory" role="tabpanel" hidden>{{MEMORY}}</div>
    <div class="tabpanel" id="tab-eval" role="tabpanel" hidden>{{EVAL}}</div>
    <section class="cases-panel" id="cases-panel">
      <div class="panel-head"><span class="cap">Past failures · recalled</span><span class="muted mono" id="cases-at"></span></div>
      <div id="cases"></div>
    </section>
  </aside>
</main>
<footer class="foot">Replays saved records only: no model calls, no reference labels. Recalled cases are limited to the development split and never come from the replayed recording. Dataset: IndustReal (Schoonbeek et al.).</footer>
<svg id="wire" aria-hidden="true"><path id="wire-path" d=""></path><circle id="wire-a" r="0"></circle><circle id="wire-b" r="0"></circle></svg>
<script type="application/json" id="replay-data">{{PAYLOAD}}</script>
"""

JS = r"""
(() => {
  "use strict";
  const P = JSON.parse(document.getElementById("replay-data").textContent);
  const LABEL = {correct: "CORRECT", incorrect: "INCORRECT", not_completed: "NOT DONE", insufficient_evidence: "INSUFFICIENT"};
  const AUTO_SECONDS = 4;
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const pill = (v) => el("span", "pill v-" + v, LABEL[v] || v);
  const fmt = (t) => { if (t == null || !isFinite(t)) return "--:--"; t = Math.max(0, t); const m = Math.floor(t / 60), s = Math.floor(t % 60); return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0"); };
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };

  let arm = P.arms.candidate ? "candidate" : (P.arms.baseline ? "baseline" : null);
  const checkpoints = () => (arm && P.arms[arm] ? P.arms[arm].checkpoints : []);

  function timeToFrame(t) {
    const tm = P.timing; if (!tm) return null;
    if (tm.mode === "fps") return Math.round(t * tm.fps + tm.offset);
    const pts = tm.points; if (t < pts[0][1] || t > pts[pts.length - 1][1]) return null;
    for (let i = 1; i < pts.length; i++) { const a = pts[i - 1], b = pts[i]; if (t <= b[1]) return Math.round(a[0] + (b[0] - a[0]) * (t - a[1]) / (b[1] - a[1])); }
    return null;
  }

  const video = $("main-video"), canvas = $("mock-canvas"), ctx = canvas.getContext("2d");
  let useVideo = false, mockNote = "", vt = 0, vPlaying = false, vLast = null;
  const clock = {
    time: () => (useVideo ? video.currentTime : vt),
    duration: () => (useVideo && isFinite(video.duration) && video.duration > 0 ? video.duration : P.duration),
    playing: () => (useVideo ? !video.paused && !video.ended : vPlaying),
    play() { if (useVideo) { const p = video.play(); if (p && p.catch) p.catch(() => {}); } else { if (vt >= P.duration) vt = 0; vPlaying = true; } },
    pause() { if (useVideo) video.pause(); else vPlaying = false; },
    seek(t) { t = Math.max(0, Math.min(t, clock.duration())); if (useVideo) video.currentTime = t; else vt = t; },
  };
  function useMock(note) {
    const t = useVideo ? video.currentTime : vt;
    useVideo = false; mockNote = note; vt = t || 0; video.hidden = true; canvas.hidden = false;
    const label = $("rec-label");
    if (P.is_mock) { label.textContent = "MOCK placeholder · no video"; label.className = "rec-label mock"; }
    else { label.textContent = "No video file · verdicts are real"; label.className = "rec-label"; }
    renderTimeline();
  }
  function drawMock() {
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#2C2C2C"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.035)"; ctx.lineWidth = 2;
    for (let x = -h; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x, h); ctx.lineTo(x + h, 0); ctx.stroke(); }
    ctx.textAlign = "center";
    const heading = P.is_mock ? "MOCK" : "NO VIDEO";
    ctx.fillStyle = P.is_mock ? "#C77DFF" : "#8A96AD"; ctx.font = "600 72px ui-monospace, Menlo, Consolas, monospace"; ctx.fillText(heading, w / 2, h * 0.22);
    ctx.fillStyle = "#8A96AD"; ctx.font = "26px system-ui, sans-serif"; ctx.fillText(mockNote, w / 2, h * 0.22 + 46);
    const f = timeToFrame(vt);
    ctx.fillStyle = "#E6EAF2"; ctx.font = "34px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText((P.recording_id || "no recording") + "  ·  frame " + (f == null ? "—" : f) + "  ·  " + fmt(vt), w / 2, h * 0.22 + 110);
  }
  let prev = 0, shown = null, countdown = null;
  const card = $("card");

  if (P.video_path) {
    useVideo = true; video.hidden = false; canvas.hidden = true;
    video.addEventListener("error", () => useMock("Video file not found: " + P.video_path));
    video.addEventListener("loadedmetadata", () => renderTimeline());
    video.src = P.video_path;
  } else {
    useMock("No video_path set in timing config");
  }

  function setActive(id) {
    document.querySelectorAll("[data-cp]").forEach((n) => n.classList.toggle("active", n.dataset.cp === id));
    const row = document.querySelector('.drow[data-cp="' + CSS.escape(id || "") + '"]');
    const panel = row && row.closest(".tabpanel");
    if (panel && !panel.hidden) {
      const pr = panel.getBoundingClientRect(), rr = row.getBoundingClientRect();
      if (rr.top < pr.top) panel.scrollTop -= pr.top - rr.top;
      else if (rr.bottom > pr.bottom) panel.scrollTop += rr.bottom - pr.bottom;
    }
  }

  function crow(label, nodes, id) {
    const r = el("div", "crow"); if (id) r.id = id;
    r.appendChild(el("div", "clabel", label));
    const v = el("div", "cval"); nodes.forEach((n) => v.appendChild(n)); r.appendChild(v); return r;
  }
  function list(items, cls) { const ul = el("ul", cls); items.forEach((t) => ul.appendChild(el("li", null, t))); return ul; }

  function fillCard(cp) {
    const a = P.arms[arm];
    $("card-title").textContent = cp.checkpoint_id + " · " + cp.component_id;
    $("card-sub").textContent = "frame " + cp.cursor_frame + " · " + fmt(cp.t) + " · run " + a.run_id;
    $("card-arm").textContent = arm + " · " + a.policy_id;
    const body = $("card-body"); body.replaceChildren();

    const seeing = [];
    if (!cp.evidence.length) seeing.push(el("span", "muted", "No evidence cited"));
    cp.evidence.forEach((ev) => {
      if (ev.missing) { seeing.push(el("div", "err", "Evidence " + ev.evidence_id + " not found in saved evidence")); return; }
      seeing.push(ev.observations.length ? list(ev.observations) : el("div", "muted", "No observations recorded"));
      if (ev.uncertainties.length) seeing.push(list(ev.uncertainties.map((u) => "Uncertain: " + u), "unc"));
      seeing.push(el("div", "src", ev.evidence_id + " · frames " + ev.frame_ids.join(", ")));
    });
    body.appendChild(crow("Seeing", seeing));

    const remembering = cp.open_issues.length
      ? [list(cp.open_issues.map((i) => i.issue_id + (i.since === cp.checkpoint_id ? " opened at this checkpoint" : i.since ? " open since " + i.since : "") + (i.description ? " — " + i.description : "")))]
      : [el("span", "muted", "No open issues")];
    body.appendChild(crow("Remembering", remembering));

    const r = cp.recalled, recalling = [];
    const SRC = {verdict: "Retrieved for this checkpoint (verdict.recalled_case_ids)", policy: "Fallback: cases the rule was learned from (policy.supporting_dev_case_ids)", none: "No recall source saved"};
    if (r.cases.length) {
      const chips = el("div", "chips"); r.cases.forEach((c) => chips.appendChild(el("span", "chip", c.case_id))); recalling.push(chips);
    } else recalling.push(el("span", "muted", "No similar past failures"));
    if (r.overflow) recalling.push(el("div", "src", "+" + r.overflow + " more not shown"));
    if (r.refused.length) recalling.push(el("div", "err", "Refused (not eligible development cases): " + r.refused.join(", ")));
    recalling.push(el("div", "src", SRC[r.source]));
    body.appendChild(crow("Recalling", recalling, "recall-row"));

    const pol = a.policy, rule = [];
    if (!pol) rule.push(el("span", "err", "No valid policy record"));
    else {
      rule.push(pol.checklist.length ? el("div", null, pol.checklist[0]) : el("span", "muted", "No added checklist rule"));
      let meta = pol.policy_id + " · " + pol.status + (pol.parent_policy_id ? " · parent " + pol.parent_policy_id : "");
      if (pol.supporting_dev_case_ids.length) meta += " · learned from " + pol.supporting_dev_case_ids.join(", ");
      rule.push(el("div", "src", meta));
    }
    body.appendChild(crow("Rule applied", rule));

    const decision = el("div"); decision.appendChild(pill(cp.verdict));
    body.appendChild(crow("Decision", [decision, el("div", null, cp.reason)]));
  }

  function caseTile(c) {
    const tile = el("div", "case"), media = el("div", "case-media");
    const label = el("span", "rec-label", "Recorded");
    const v = document.createElement("video");
    v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true; v.setAttribute("muted", "");
    v.addEventListener("error", () => {
      v.remove(); media.classList.add("missing"); media.appendChild(el("div", "missing-text", c.clip_path));
      if (c.source_kind === "mock") { label.textContent = "MOCK · no clip"; label.className = "rec-label mock"; }
      else { label.textContent = "No clip file"; label.className = "rec-label"; }
    });
    v.src = c.clip_path; media.append(v, label);
    const meta = el("div");
    const head = el("div", "case-head"); head.append(el("span", "mono", c.case_id), el("span", "tag", "dev split"), el("span", "muted", c.component_id));
    const line = el("div", "case-line"); line.append("said then", pill(c.verdict_then), "revealed", pill(c.revealed_outcome));
    meta.append(head, line, el("p", "case-sum", c.summary), el("div", "src mono", c.recording_id + " · frames " + c.frame_start + "–" + c.frame_end));
    tile.append(media, meta); return tile;
  }

  function renderCases(cp) {
    const box = $("cases"); box.replaceChildren();
    $("cases-at").textContent = cp ? "at " + cp.checkpoint_id + " · " + arm : "";
    if (!cp) { box.appendChild(el("p", "empty", "Recalled past failures appear here at each decision point.")); return; }
    if (!cp.recalled.cases.length) { box.appendChild(el("p", "empty", "No similar past failures")); return; }
    cp.recalled.cases.forEach((c) => box.appendChild(caseTile(c)));
  }

  function drawWire() {
    const path = $("wire-path"), a = $("wire-a"), b = $("wire-b");
    const clear = () => { path.setAttribute("d", ""); a.setAttribute("r", 0); b.setAttribute("r", 0); };
    const row = $("recall-row"), cp = shown && checkpoints().find((c) => c.checkpoint_id === shown);
    if (!row || !cp || !cp.recalled.cases.length || !card.classList.contains("show")) return clear();
    const r1 = row.getBoundingClientRect(), r2 = $("cases").getBoundingClientRect();
    if (r2.left < r1.right + 24 || r1.height === 0 || r2.top + 20 > window.innerHeight || r1.bottom < 0) return clear();
    const x1 = r1.right, y1 = r1.top + Math.min(r1.height / 2, 14), x2 = r2.left - 4, y2 = r2.top + 20;
    path.setAttribute("d", "M" + x1 + "," + y1 + " C" + (x1 + 90) + "," + y1 + " " + (x2 - 90) + "," + y2 + " " + x2 + "," + y2);
    a.setAttribute("cx", x1); a.setAttribute("cy", y1); a.setAttribute("r", 3);
    b.setAttribute("cx", x2); b.setAttribute("cy", y2); b.setAttribute("r", 3);
  }

  const SPIN = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"];
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  let mono = null;

  function narration(cp) {
    const a = P.arms[arm], memory = P.backend === "local" ? "local memory" : "Atlas";
    const line = (text, cls) => ({text, cls});
    const observe = [];
    if (!cp.evidence.length) observe.push(line("No evidence was cited for this checkpoint.", "err"));
    cp.evidence.forEach((ev) => {
      if (ev.missing) { observe.push(line("Evidence " + ev.evidence_id + " is missing from the saved records.", "err")); return; }
      observe.push(line("Looking at frames " + ev.frame_ids.join(", ") + " (" + ev.evidence_id + ") for " + cp.component_id + "."));
      if (!ev.observations.length) observe.push(line("Nothing usable was observed in these frames."));
      ev.observations.forEach((o) => observe.push(line("I can see: " + o, "say")));
      ev.uncertainties.forEach((u) => observe.push(line("I can't confirm: " + u, "unc")));
    });
    const remembered = cp.open_issues.filter((i) => i.since !== cp.checkpoint_id);
    const opened = cp.open_issues.filter((i) => i.since === cp.checkpoint_id);
    const memoryLines = remembered.length
      ? [line("Session " + a.run_id + " in " + memory + " holds " + remembered.length + " unresolved issue" + (remembered.length > 1 ? "s" : "") + ":")]
          .concat(remembered.map((i) => line(i.issue_id + (i.since ? " (open since " + i.since + ")" : " (no issue event saved)") + (i.description ? ": " + i.description : ""), "say")))
      : [line("No unresolved issues carried into this checkpoint in " + memory + ".")];
    const r = cp.recalled, recall = [];
    if (r.cases.length) {
      recall.push(line(r.source === "verdict"
        ? "Retrieved " + r.cases.length + " similar past failure" + (r.cases.length > 1 ? "s" : "") + " from the development split:"
        : "No retrieval was saved here, so I'm listing the cases my current rule was learned from:"));
      r.cases.forEach((c) => recall.push(line(c.case_id + " · " + c.component_id + ": I said " + (LABEL[c.verdict_then] || c.verdict_then) + ", it turned out " + (LABEL[c.revealed_outcome] || c.revealed_outcome) + ". " + c.summary, "say")));
      const ids = r.cases.map((c) => c.case_id).join(" and ");
      recall.push(line("Each of those was a miss I made before under similar evidence, not just a similar-looking case — " + ids + " should weigh against repeating the same call.", "say"));
    } else recall.push(line("No similar past failures on record for this checkpoint, so memory offers no precedent here."));
    if (r.refused.length) recall.push(line("Refused " + r.refused.join(", ") + ": not eligible development cases.", "err"));
    const pol = a.policy;
    const rule = !pol ? [line("No valid policy record was saved.", "err")]
      : pol.checklist.length ? [line("Checklist " + pol.policy_id + " says: " + pol.checklist[0], "say")]
      : [line("No extra checklist rule in " + pol.policy_id + "; applying the base verification only.")];
    const weighPieces = [];
    if (cp.evidence.some((ev) => !ev.missing && ev.uncertainties.length)) weighPieces.push("what's still uncertain in the frames");
    if (remembered.length) weighPieces.push(remembered.length + " issue" + (remembered.length > 1 ? "s" : "") + " already open in " + memory);
    if (r.cases.length) weighPieces.push("the " + r.cases.length + " past miss" + (r.cases.length > 1 ? "es" : "") + " just recalled");
    if (pol && pol.checklist.length) weighPieces.push("the checklist rule in " + pol.policy_id);
    const weighLine = weighPieces.length
      ? line("Weighing " + weighPieces.join(", ") + " against each other:")
      : line("Nothing on record complicates this one — checking the observations directly against the instruction:");
    const decide = [weighLine, line("Verdict: " + (LABEL[cp.verdict] || cp.verdict) + " — " + cp.reason, "say")];
    opened.forEach((i) => decide.push(line("Opening " + i.issue_id + (i.description ? ": " + i.description : "") + " Saved to " + memory + ".")));
    a.issues.filter((i) => i.resolved && i.resolved_at === cp.checkpoint_id)
      .forEach((i) => decide.push(line("Resolved " + i.issue_id + " with evidence " + i.resolved_by.join(", ") + ".")));
    return [
      {verbs: ["Squinting", "Observing", "Inspecting frames"], done: "Observed", lines: observe},
      {verbs: ["Querying " + memory, "Rummaging through memory", "Consulting " + memory], done: "Checked " + memory, lines: memoryLines},
      {verbs: ["Recalling", "Reminiscing", "Pattern-matching"], done: "Recalled", lines: recall},
      {verbs: ["Cross-checking", "Consulting the checklist", "Tinkering"], done: "Checked the checklist", lines: rule},
      {verbs: ["Weighing it all up", "Deliberating", "Pondering", "Evaluating"], done: "Decided", lines: decide},
    ];
  }

  async function type(node, text, run) {
    if (reduceMotion || run.fast) { node.textContent = text; return; }
    const caret = el("span", "caret"), chars = Math.max(1, Math.ceil(text.length / 22));
    const span = document.createTextNode(""); node.append(span, caret);
    const box = $("monologue");
    for (let i = 0; i < text.length && !run.fast; i += chars) { span.data = text.slice(0, i + chars); box.scrollTop = box.scrollHeight; await wait(16); }
    span.data = text; caret.remove(); box.scrollTop = box.scrollHeight;
  }

  async function monologue(cp) {
    const run = {fast: reduceMotion}; mono = run;
    const box = $("monologue"); box.replaceChildren();
    const seed = checkpoints().indexOf(cp) + (arm === "baseline" ? 1 : 0);
    for (const [n, step] of narration(cp).entries()) {
      if (mono !== run) return false;
      const s = el("div", "mstep"), head = el("div", "mhead");
      const glyph = el("span", "mglyph", SPIN[0]), verb = el("span", "mverb", step.verbs[(seed + n) % step.verbs.length] + "…");
      head.append(glyph, verb); s.appendChild(head); box.appendChild(s); box.scrollTop = box.scrollHeight;
      for (let i = 0, spins = 6 + ((seed + n) % 4); i < spins && !run.fast; i++) { glyph.textContent = SPIN[i % SPIN.length]; await wait(90); }
      if (mono !== run) return false;
      glyph.textContent = "●"; glyph.className = "mglyph done"; verb.textContent = step.done; verb.className = "mverb done";
      for (const [k, ln] of step.lines.entries()) {
        const node = el("div", "mline" + (k === 0 ? " first" : "") + (ln.cls ? " " + ln.cls : "")); s.appendChild(node);
        await type(node, ln.text, run);
        if (mono !== run) return false;
      }
    }
    return true;
  }

  function idleMonologue() {
    mono = null;
    const box = $("monologue"); box.replaceChildren();
    box.appendChild(el("p", "empty", "Press ▶ and the verifier's reasoning trace plays out here at each decision point."));
  }

  function clearAuto() { if (countdown) clearInterval(countdown); countdown = null; $("card-auto").textContent = ""; }
  function startAuto() {
    clearAuto(); if (!$("auto").checked) return;
    let left = AUTO_SECONDS; $("card-auto").textContent = "Resuming in " + left + "s";
    countdown = setInterval(() => { left -= 1; if (left <= 0) cont(); else $("card-auto").textContent = "Resuming in " + left + "s"; }, 1000);
  }

  function show(cp) {
    shown = cp.checkpoint_id; fillCard(cp); card.classList.add("show");
    setActive(cp.checkpoint_id); renderCases(cp);
    requestAnimationFrame(drawWire); clearAuto();
    const id = cp.checkpoint_id, forArm = arm;
    monologue(cp).then((finished) => { if (finished && shown === id && arm === forArm) startAuto(); });
  }
  function hideCard() { shown = null; card.classList.remove("show"); clearAuto(); if (mono) mono.fast = true; drawWire(); }
  function cont() { hideCard(); clock.play(); }
  function jump(cp) { clock.pause(); if (cp.t != null) { clock.seek(cp.t); prev = cp.t; } show(cp); }

  function renderTimeline() {
    const d = clock.duration(), pct = (t) => (Math.max(0, Math.min(1, t / d)) * 100) + "%";
    const axis = $("axis"); axis.replaceChildren();
    const stepS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => d / s <= 8) || 1200;
    for (let t = 0; t < d - stepS * 0.4; t += stepS) { const s = el("span", null, fmt(t)); s.style.left = pct(t); axis.appendChild(s); }
    const markers = $("markers"); markers.replaceChildren();
    checkpoints().forEach((cp) => {
      if (cp.t == null) return;
      const m = el("button", "marker v-" + cp.verdict); m.type = "button"; m.dataset.cp = cp.checkpoint_id;
      m.style.left = pct(cp.t); m.title = cp.checkpoint_id + " · " + cp.component_id + " · " + (LABEL[cp.verdict] || cp.verdict);
      m.setAttribute("aria-label", m.title);
      m.addEventListener("click", (e) => { e.stopPropagation(); jump(cp); });
      markers.appendChild(m);
    });
    const lanes = $("lanes"); lanes.replaceChildren();
    const issues = arm && P.arms[arm] ? P.arms[arm].issues : [];
    if (!issues.length) lanes.appendChild(el("div", "muted", "No issues opened in this run"));
    issues.forEach((is) => {
      const start = is.start == null ? 0 : is.start, end = is.resolved && is.end != null ? is.end : d;
      const lane = el("div", "lane"), bar = el("div", "bar " + (is.resolved ? "resolved" : "open"));
      bar.style.left = pct(start); bar.style.width = (Math.max(0, end - start) / d * 100) + "%";
      bar.title = is.issue_id + ": " + is.description + " — " + (is.resolved ? "resolved at " + is.resolved_at + " by " + is.resolved_by.join(", ") : "still open at end of run");
      bar.appendChild(el("span", "bar-label", is.issue_id));
      is.carried.forEach((t) => { if (end > start) { const k = el("span", "tick"); k.style.left = ((t - start) / (end - start) * 100) + "%"; bar.appendChild(k); } });
      lane.appendChild(bar); lanes.appendChild(lane);
    });
    setActive(shown);
  }

  function renderList() {
    const box = $("decisions"); box.replaceChildren();
    if (!checkpoints().length) { box.appendChild(el("p", "empty", "No valid verdicts for this policy.")); return; }
    checkpoints().forEach((cp) => {
      const row = el("button", "drow"); row.type = "button"; row.dataset.cp = cp.checkpoint_id;
      row.append(el("span", "chip", fmt(cp.t)), el("span", "mono", cp.checkpoint_id), el("span", "comp", cp.component_id));
      if (cp.recalled.cases.length) row.appendChild(el("span", "recalls", "↺" + cp.recalled.cases.length));
      row.appendChild(pill(cp.verdict));
      row.addEventListener("click", () => jump(cp));
      box.appendChild(row);
    });
  }

  function setArm(next) {
    if (!P.arms[next]) return;
    arm = next;
    document.querySelectorAll("[data-arm]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.arm === arm)));
    $("h-run").textContent = P.arms[arm].run_id; $("h-policy").textContent = P.arms[arm].policy_id;
    renderList(); renderTimeline();
    const cp = shown && checkpoints().find((c) => c.checkpoint_id === shown);
    if (cp) show(cp);
    else { hideCard(); renderCases(null); idleMonologue(); }
  }

  function seekFromTrack(e) {
    const r = $("track").getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * clock.duration();
    hideCard(); clock.seek(t); prev = t; setActive(null);
  }

  function step(dir) {
    const now = clock.time(), cps = checkpoints().filter((c) => c.t != null);
    const cp = dir > 0 ? cps.find((c) => c.t > now + 0.05) : cps.filter((c) => c.t < now - 0.05).pop();
    if (cp) jump(cp);
  }

  function togglePlay() { if (shown) cont(); else if (clock.playing()) clock.pause(); else clock.play(); }

  function loop() {
    const ts = performance.now();
    if (!useVideo) {
      if (vPlaying) { if (vLast != null) vt = Math.min(P.duration, vt + (ts - vLast) / 1000); vLast = ts; if (vt >= P.duration) vPlaying = false; }
      else vLast = null;
      drawMock();
    }
    let now = clock.time();
    if (clock.playing()) {
      const cp = checkpoints().find((c) => c.t != null && prev < c.t && c.t <= now + 1e-6);
      if (cp) { clock.pause(); clock.seek(cp.t); now = cp.t; show(cp); }
    }
    prev = now;
    const d = clock.duration(), p = Math.max(0, Math.min(1, now / d)) * 100 + "%";
    $("played").style.width = p; $("playhead").style.left = p;
    const f = timeToFrame(now);
    $("readout").textContent = fmt(now) + " / " + fmt(d) + " · frame " + (f == null ? "—" : f);
    const playing = clock.playing(); $("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;"; $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
    requestAnimationFrame(loop);
  }

  document.querySelectorAll("[data-arm]").forEach((b) => b.addEventListener("click", () => setArm(b.dataset.arm)));
  document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    document.querySelectorAll(".tabpanel").forEach((p) => { p.hidden = p.id !== "tab-" + b.dataset.tab; });
    store.set("replay-tab", b.dataset.tab); drawWire();
  }));
  $("play").addEventListener("click", togglePlay);
  $("continue").addEventListener("click", cont);
  $("prev-cp").addEventListener("click", () => step(-1));
  $("next-cp").addEventListener("click", () => step(1));
  $("track").addEventListener("click", seekFromTrack);
  card.addEventListener("mouseenter", clearAuto);
  card.addEventListener("scroll", drawWire);
  card.addEventListener("transitionend", drawWire);
  $("auto").addEventListener("change", (e) => { store.set("replay-auto", e.target.checked ? "1" : "0"); if (!e.target.checked) clearAuto(); });
  window.addEventListener("resize", drawWire);
  window.addEventListener("scroll", drawWire, true);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  });

  if (store.get("replay-auto") === "0") $("auto").checked = false;
  const tab = new URLSearchParams(location.hash.slice(1)).get("tab") || store.get("replay-tab"); if (tab) { const b = document.querySelector('[data-tab="' + CSS.escape(tab) + '"]'); if (b) b.click(); }
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get("arm") && P.arms[hash.get("arm")]) arm = hash.get("arm");
  if (arm) setArm(arm); else { renderList(); renderTimeline(); }
  renderCases(null); idleMonologue();
  const linked = hash.get("cp") && checkpoints().find((c) => c.checkpoint_id === hash.get("cp"));
  if (linked) jump(linked);
  requestAnimationFrame(loop);
})();
"""
