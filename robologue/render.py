"""Render a self-contained replay page from decisions JSONL. No external assets.

The page shows the event stream with evidence, open issues, and the harness
decision per event, with a memory/stateless toggle. This is a replay view of
the harness output, not a dashboard product.
"""

import html
import json
from pathlib import Path


def _load_jsonl(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def _rows(decisions, observations):
    rows = []
    for decision in decisions:
        if decision.get("status") == "already_processed":
            continue
        event_id = decision["event_id"]
        obs = observations.get(event_id, {})
        action = decision.get("action", "")
        badge = "verify" if action == "request_verification" else "observe"
        issues = "".join(
            f'<span class="issue">{html.escape(iid)}: {html.escape(info.get("description", ""))}</span>'
            for iid, info in decision.get("open_issues", {}).items()
        ) or '<span class="none">none</span>'
        steps = ", ".join(html.escape(s) for s in decision.get("completed_steps", [])) or "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(decision.get('timestamp', '')))}</td>"
            f"<td>{html.escape(obs.get('observation', ''))}</td>"
            f'<td><span class="badge {badge}">{html.escape(action)}</span></td>'
            f"<td>{issues}</td>"
            f"<td>{steps}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render(events_path, memory_path, stateless_path=None, out_path="demo.html"):
    events = _load_jsonl(events_path)
    observations = {e["event_id"]: e for e in events}
    memory_decisions = _load_jsonl(memory_path)
    stateless_decisions = _load_jsonl(stateless_path) if stateless_path else []

    synthetic = any(str(e.get("recording_id", "")).startswith("synthetic") for e in events)
    banner = (
        '<div class="banner">Synthetic fixture: observations are hand-authored for this demo, '
        "not model output.</div>"
        if synthetic else ""
    )
    toggle = ""
    if stateless_decisions:
        toggle = (
            '<div class="toggle">'
            '<button onclick="show(\'memory\')">memory</button> '
            '<button onclick="show(\'stateless\')">stateless</button>'
            "</div>"
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Robologue replay</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
.banner {{ background: #fff3cd; border: 1px solid #e0c36a; padding: 10px; margin-bottom: 16px; }}
.toggle button {{ margin-right: 8px; padding: 6px 14px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.badge {{ padding: 2px 8px; border-radius: 10px; font-size: 12px; }}
.verify {{ background: #ffd9d9; }} .observe {{ background: #d9ecff; }}
.issue {{ display: inline-block; background: #ffe9c9; border: 1px solid #d9a441; border-radius: 6px; padding: 2px 6px; margin: 2px; font-size: 12px; }}
.none {{ color: #888; font-size: 12px; }}
</style></head><body>
<h2>Robologue replay</h2>
{banner}
{toggle}
<div id="memory"><h3>memory mode</h3><table>
<tr><th>t</th><th>observation</th><th>decision</th><th>open issues</th><th>completed steps</th></tr>
{_rows(memory_decisions, observations)}
</table></div>
<div id="stateless" style="display:none"><h3>stateless mode</h3><table>
<tr><th>t</th><th>observation</th><th>decision</th><th>open issues</th><th>completed steps</th></tr>
{_rows(stateless_decisions, observations)}
</table></div>
<script>
function show(which) {{
  document.getElementById('memory').style.display = which === 'memory' ? '' : 'none';
  document.getElementById('stateless').style.display = which === 'stateless' ? '' : 'none';
}}
</script>
</body></html>"""
    out = Path(out_path)
    out.write_text(page)
    return out
