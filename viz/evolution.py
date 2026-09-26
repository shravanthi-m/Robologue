"""Evolution timeline visualization.

Reads the latest benchmark report from the ``evaluations`` collection plus
``architectures`` and ``mutations``, and writes a single static HTML file
with inline SVG: success rate and avg steps per generation across trials,
annotated with the mutations that were adopted. No server, no dashboard.
"""
from __future__ import annotations

import os
import argparse
from html import escape

import db as db_module

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "evolution.html")
W, H = 960, 320
PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 30, 60


def _latest_benchmark(db):
    reports = db.find("evaluations", {"kind": "benchmark"}, limit=100)
    reports += db.find("evaluations", {"kind": "run"}, limit=100)
    if not reports:
        return None
    return max(reports, key=lambda r: r.get("created_at", ""))


def _scale(values, lo, hi, y0, y1):
    span = (hi - lo) or 1.0
    return [y1 - (v - lo) / span * (y1 - y0) for v in values]


def _panel(title, ylabel, trials_data, lo, hi, annotations, mean_color):
    """trials_data: list of (label, [values per generation])."""
    n = len(trials_data[0][1])
    xs = [PAD_L + i * (W - PAD_L - PAD_R) / max(n - 1, 1) for i in range(n)]
    y0, y1 = H - PAD_B, PAD_T
    parts = [f'<text x="{W/2}" y="18" text-anchor="middle" '
             f'font-family="sans-serif" font-size="15" font-weight="bold">{title}</text>']
    # axes
    parts.append(f'<line x1="{PAD_L}" y1="{y0}" x2="{W-PAD_R}" y2="{y0}" stroke="#333"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{y0}" x2="{PAD_L}" y2="{y1}" stroke="#333"/>')
    for i, x in enumerate(xs):
        parts.append(f'<text x="{x}" y="{y0+20}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11">gen {i}</text>')
    for frac in (0.0, 0.5, 1.0):
        v = lo + frac * (hi - lo)
        y = y1 + (1 - frac) * (y0 - y1)
        parts.append(f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" '
                     f'stroke="#ddd"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+4}" text-anchor="end" '
                     f'font-family="sans-serif" font-size="11">{v:.1f}</text>')
    parts.append(f'<text x="14" y="{H/2}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12" '
                 f'transform="rotate(-90 14 {H/2})">{ylabel}</text>')
    # trial lines (light) and mean (bold)
    means = []
    for li, (label, vals) in enumerate(trials_data):
        ys = _scale(vals, lo, hi, y0, y1)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#bbb" '
                     f'stroke-width="1.5"/>')
        means.append(vals)
    mean_vals = [sum(col) / len(col) for col in zip(*means)]
    mys = _scale(mean_vals, lo, hi, y0, y1)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, mys))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{mean_color}" '
                 f'stroke-width="3"/>')
    for x, y in zip(xs, mys):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{mean_color}"/>')
    # mutation annotations
    for gen_idx, text in annotations:
        if gen_idx < len(xs):
            x = xs[gen_idx]
            parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" '
                         f'y2="{y0+34}" stroke="#c00" stroke-dasharray="4,3"/>')
            parts.append(f'<text x="{x+4:.1f}" y="{y0+30}" font-family="sans-serif" '
                         f'font-size="11" fill="#c00">{text}</text>')
    return (f'<svg width="{W}" height="{H}" style="background:#fff;'
            f'border:1px solid #ccc;margin:12px 0">' + "".join(parts) + "</svg>")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "atlas", "local"), default="auto")
    parser.add_argument("--env-file")
    parser.add_argument("--run-id")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args([] if argv is None else argv)
    db = db_module.get_db(args.backend, env_file=args.env_file, run_id=args.run_id)
    report = _latest_benchmark(db)
    if report is None:
        print("No benchmark report in evaluations. Run this first:")
        print("  python3 -m eval.benchmark --trials 3")
        db.close()
        return

    trials = report["trial_reports"]
    n_gen = report["generations"]
    versions = [trials[0]["generations"][g]["version"] for g in range(n_gen)]
    success = [(f"trial {t['trial']}",
                [t["generations"][g]["success_rate"] for g in range(n_gen)])
               for t in trials]
    steps = [(f"trial {t['trial']}",
              [t["generations"][g]["avg_steps"] for g in range(n_gen)])
             for t in trials]

    annotations = []
    for g in range(n_gen):
        m = trials[0]["generations"][g]["mutation"]
        if m and m["accepted"]:
            comp = escape(str(m["new_component"] or "policy change"))
            annotations.append((g, f"{escape(str(m['mutation_type']))} +{comp} &#8594; {escape(str(m['resulting_version']))}"))

    max_steps = max(v for _, vals in steps for v in vals) or 1
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Evolution timeline</title></head>
<body style="font-family:sans-serif;max-width:1000px;margin:0 auto;padding:16px">
<h2>Recursive harness evolution (benchmark {report['eval_id'][:8]})</h2>
<p>Tasks: {', '.join(report['tasks'])} &middot; seeds {report['seeds']} &middot;
{report['trials']} trials &middot; versions: {' &#8594; '.join(versions)}</p>
<p style="color:#555">Thin grey lines are individual trials; the bold line is the mean.
Red markers are mutations the validator accepted on evidence.</p>
{_panel("Success rate per generation", "success rate", success, 0, 1.05, annotations, "#1a73e8")}
{_panel("Average steps per generation", "avg steps", steps, 0, max_steps * 1.1, [], "#0b8043")}
<h3>Mutations</h3>
<ul>
{''.join(f"<li><b>{escape(str(m['mutation_type']))}</b> +{escape(str(m['new_component'] or 'policy'))}: "
         f"{escape(str(m['parent_version']))} &#8594; {escape(str(m['resulting_version']))} "
         f"(accepted={m['accepted']}, lessons validated={m.get('lessons_validated', 0)})</li>"
         for m in db.find('mutations', {}, limit=100))}
</ul>
<p style="color:#555">Every point traces to stored trajectories (trace ids) and
mutation records in the database.</p>
</body></html>"""

    with open(args.out, "w") as f:
        f.write(html)
    db.close()
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
