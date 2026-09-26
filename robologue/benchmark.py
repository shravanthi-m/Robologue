"""Evaluator-only coordinator. Labels never enter run_recording or model context."""

import json
from pathlib import Path
from collections import Counter
from .industreal_labels import reference_records
from .evaluate import score, VERDICTS
from .policies import propose_from_false_corrects, decide_promotion, finalize
from .pipeline import run_recording
from .perception.video import write_json, digest_file

SPLITS = {"development": "03", "validation": "08", "heldout": "09"}
PROTOCOL_VERSION = "native-nine-subject-holdout-v1"


def extended_score(verdicts, references):
    report = score(verdicts, references)
    confusion = {truth: {pred: 0 for pred in VERDICTS} for truth in VERDICTS[:3]}
    refs = {(r["recording_id"], r["component_id"], r["frame"]): r for r in references}
    for v in verdicts:
        ref = refs.get((v["recording_id"], v["component_id"], v["cursor_frame"]))
        if ref:
            confusion[ref["reference"]][v["verdict"]] += 1
    report["confusion_matrix"] = confusion
    report["coverage"] = (
        report["judged"] / report["scorable"] if report["scorable"] else None
    )
    report["per_class"] = {}
    for truth in VERDICTS[:3]:
        tp = confusion[truth][truth]
        predicted = sum(row[truth] for row in confusion.values())
        support = sum(confusion[truth].values())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        report["per_class"][truth] = {
            "precision": precision,
            "recall": recall,
            "support": support,
            "f1": 2 * tp / (support + predicted) if support + predicted else None,
        }
    n = report["scorable"]
    if n:
        p = report["state_accuracy"]
        z = 1.96
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        delta = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
        report["descriptive_wilson_95"] = [centre - delta, centre + delta]
        report["interval_caveat"] = (
            "Component states are correlated within nine recordings; this interval is descriptive, not independent-trial evidence."
        )
    return report


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    )
    temporary.replace(path)


def run_benchmark(
    root, output, store, client, *, run_id="native-eval-v1", runner=run_recording
):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    components = json.loads((root / "components.json").read_text())["components"]
    records = sorted(p.name for p in (root / "native").iterdir() if p.is_dir())
    if len(records) != 9 or any(r[:2] not in SPLITS.values() for r in records):
        raise ValueError(
            "This frozen benchmark requires the supplied nine recordings from subjects 03,08,09"
        )
    refs = {
        r: list(reference_records(r, root / "native" / r / "PSR_labels_raw.csv"))
        for r in records
    }
    if any(
        {v["component_id"] for v in values} != set(components)
        for values in refs.values()
    ):
        raise ValueError(
            "Public component catalog must match the evaluator reference schema"
        )
    groups = {
        split: [r for r in records if r.startswith(subject + "_")]
        for split, subject in SPLITS.items()
    }
    maps = {
        r: json.loads((output / "media" / "time_maps" / f"{r}.json").read_text())
        for r in records
    }
    protocol = {
        "version": PROTOCOL_VERSION,
        "groups": groups,
        "components": components,
        "reference_file_sha256": {
            r: digest_file(root / "native" / r / "PSR_labels_raw.csv") for r in records
        },
        "checkpoints": {r: sorted({v["frame"] for v in refs[r]}) for r in records},
        "scope": "Exploratory subject-disjoint split INSIDE official test_p1. Not the official IndustReal train/validation/test benchmark.",
        "checkpoint_schedule": "Numeric PSR annotation frames selected by evaluator; state values and trial identity never enter model prompts. This is checkpoint verification, not autonomous checkpoint discovery.",
        "selection": "One deterministic dev proposal; fixed paired validation gate; final subject evaluated after selection freezes.",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Frozen evaluation protocol changed")
    write_json(protocol_path, protocol)
    runs = {}
    all_decisions = []

    def phase(name, recordings, mode, policy=None, candidate=False):
        results = []
        for rec in recordings:
            result = runner(
                store,
                f"{run_id}-{name}",
                rec,
                root / "native" / rec / "rgb",
                protocol["checkpoints"][rec],
                components,
                maps[rec],
                client,
                output / "runtime",
                mode=mode,
                policy=policy,
                allow_candidate=candidate,
            )
            if not result["complete"]:
                raise ValueError("Incomplete run cannot be evaluated")
            results.extend(result["verdicts"])
            all_decisions.extend(result["decisions"])
        write_jsonl(output / f"{name}.jsonl", results)
        return results

    def references(recordings):
        return [v for r in recordings for v in refs[r]]

    # Do not inspect heldout labels for policy construction or promotion.
    dev = phase("development", groups["development"], "memory")
    validation = phase("validation", groups["validation"], "memory")
    candidate = propose_from_false_corrects(
        dev,
        references(groups["development"]),
        "connection-check-v2",
        "baseline-v1",
        components,
    )
    frozen_path = output / "selection.json"
    if frozen_path.exists():
        selection = json.loads(frozen_path.read_text())
        if selection["proposed_policy"] != candidate:
            raise ValueError("Development proposal changed after selection freeze")
    else:
        selection = {
            "proposed_policy": candidate,
            "promotion": None,
            "frozen_policy": None,
        }
        if candidate:
            trial = phase(
                "candidate-validation", groups["validation"], "memory", candidate, True
            )
            decision = decide_promotion(
                extended_score(validation, references(groups["validation"])),
                extended_score(trial, references(groups["validation"])),
                2 * sum(len(protocol["checkpoints"][r]) for r in groups["validation"]),
            )
            selection.update(
                promotion=decision,
                frozen_policy=(
                    finalize(candidate, decision["decision"])
                    if decision["decision"] in ("accept", "reject")
                    else None
                ),
            )
        write_json(frozen_path, selection)
    selected = selection["frozen_policy"]
    accepted = selected if selected and selected["status"] == "accepted" else None
    heldout = phase("heldout-baseline", groups["heldout"], "memory")
    selected_heldout = (
        phase("heldout-selected", groups["heldout"], "memory", accepted)
        if accepted
        else heldout
    )
    # Baseline comparison is predeclared; never used to pick or tune policies.
    stateless = phase("stateless", records, "stateless")
    baseline = dev + validation + heldout
    all_refs = references(records)
    runs["memory_baseline"] = extended_score(baseline, all_refs)
    runs["stateless"] = extended_score(stateless, all_refs)
    runs["development"] = extended_score(dev, references(groups["development"]))
    runs["validation"] = extended_score(validation, references(groups["validation"]))
    runs["heldout_baseline"] = extended_score(heldout, references(groups["heldout"]))
    runs["heldout_selected"] = extended_score(
        selected_heldout, references(groups["heldout"])
    )
    candidate_path = output / "candidate-validation.jsonl"
    if candidate_path.exists():
        trial = [
            json.loads(s) for s in candidate_path.read_text().splitlines() if s.strip()
        ]
        runs["candidate_validation"] = extended_score(
            trial, references(groups["validation"])
        )
    write_jsonl(output / "memory-baseline.jsonl", baseline)
    write_jsonl(output / "decisions.jsonl", all_decisions)
    distribution = Counter(r["reference"] for r in all_refs)
    failures = [v for v in baseline if v.get("provider_failed")]
    report = {
        "status": "complete",
        "protocol": protocol,
        "selection": selection,
        "runs": runs,
        "reference_distribution": dict(distribution),
        "scored_states": len(all_refs),
        "naive_constant_accuracy": {
            k: v / len(all_refs) for k, v in distribution.items()
        },
        "budget": client.budget.summary(),
        "provider_failed_component_verdicts": len(failures),
        "input": {
            "native_recordings": len(records),
            "public_components": len(components),
            "RGB_frames_per_inspection": "up to 3 past/current frames; never future",
            "memory": "last 3 model-generated checkpoints, bounded; no labels",
        },
        "output": {
            "neutral_inspections": "immutable store",
            "component_verdicts": "JSONL: four outcomes, rationale, cited frame IDs",
            "durable_decisions": "store and decisions.jsonl",
            "metrics": "report.json",
            "selected_policy": "selection.json",
        },
        "limitations": [
            "Only 16 incorrect reference states across the supplied native recordings.",
            "All nine native recordings come from official test_p1; internal development/validation must not be presented as official heldout results.",
            "Public component descriptions provide no CAD or correct-assembly reference images.",
            "All 86 MP4s are decoded and audited; state accuracy covers the nine recordings with supplied PSR labels.",
            "Action labels are audited for identity/ranges; this run measures component state verification, not action recognition.",
            "Budget ledger counts all dispatched attempts, including failures; per-verdict model_calls can undercount crash-before-commit attempts.",
        ],
    }
    write_json(output / "report.json", report)
    from .report import render_report

    render_report(report, output / "report.html")
    return report
