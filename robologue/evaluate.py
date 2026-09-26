"""Person 3: hidden-label scorer for IndustReal component verification.

Evaluator-only: reference records carry ground truth and must never enter
agent tools or verifier context. Scores verdicts (Interface B) against
references from industreal_labels.

Verdict semantics (frozen):
  - "correct": the component is correctly assembled.
  - "incorrect": the component is incorrectly assembled (-1).
  - "not_completed": the component is not yet assembled (0). This is NOT an
    error shortcut: unfinished work during normal assembly may be expected.
  - "insufficient_evidence": abstention. Never counted as a correct
    prediction, so all-abstain runs score zero accuracy and cannot win
    promotion.

Metrics: state accuracy over all scorable cases, false-correct count
(approvals of incorrect or unfinished components), incorrect-state recall
(None when no incorrect references exist, never zero-or-perfect by default),
abstention count, model calls and latency.
"""

VERDICTS = ("correct", "incorrect", "not_completed", "insufficient_evidence")

REQUIRED_VERDICT_FIELDS = (
    "schema_version", "source_kind", "run_id", "recording_id",
    "checkpoint_id", "cursor_frame", "component_id", "verdict",
    "policy_id", "model_id",
)


class EvaluationError(ValueError):
    """Raised when a verdict or reference set cannot be scored."""


def validate_verdict(verdict):
    """Consumer-side check of Interface B. Person 2 produces these; we verify."""
    missing = [f for f in REQUIRED_VERDICT_FIELDS if f not in verdict]
    if missing:
        raise EvaluationError(f"Verdict missing fields: {missing}")
    if verdict["verdict"] not in VERDICTS:
        raise EvaluationError(f"Unknown verdict: {verdict['verdict']!r}")
    if not isinstance(verdict["cursor_frame"], int) or verdict["cursor_frame"] < 0:
        raise EvaluationError("cursor_frame must be a nonnegative int")
    return True


def match_pairs(verdicts, references):
    """Join verdicts to references by (recording_id, component_id, frame).

    Only known reference points are scored; verdicts without a matching
    reference are excluded and reported, never forward-filled or silently
    misattributed. The same frame number from another recording cannot match.
    """
    index = {}
    for ref in references:
        key = (ref["recording_id"], ref["component_id"], ref["frame"])
        if key in index:
            raise EvaluationError(f"Duplicate reference for {key}")
        index[key] = ref
    pairs, excluded = [], []
    for verdict in verdicts:
        validate_verdict(verdict)
        key = (verdict["recording_id"], verdict["component_id"], verdict["cursor_frame"])
        ref = index.get(key)
        if ref is None:
            excluded.append(verdict["checkpoint_id"])
        else:
            pairs.append((verdict, ref))
    return pairs, excluded


def score(verdicts, references, run_meta=None):
    """Score verdicts against references. Returns the evaluation report dict."""
    pairs, excluded = match_pairs(verdicts, references)
    run_meta = run_meta or {}

    n_correct = 0
    false_correct = 0
    incorrect_total = 0
    incorrect_recalled = 0
    abstentions = 0
    model_calls = 0
    latency_ms = 0.0
    per_component = {}
    per_recording = {}

    for verdict, ref in pairs:
        model_calls += verdict.get("model_calls", 0)
        latency_ms += verdict.get("latency_ms", 0.0)
        comp = per_component.setdefault(verdict["component_id"],
                                        {"scorable": 0, "correct": 0, "false_correct": 0})
        rec = per_recording.setdefault(verdict["recording_id"],
                                       {"scorable": 0, "correct": 0, "false_correct": 0})
        comp["scorable"] += 1
        rec["scorable"] += 1
        label = verdict["verdict"]
        truth = ref["reference"]
        if label == "insufficient_evidence":
            abstentions += 1
        else:
            if label == truth:
                n_correct += 1
                comp["correct"] += 1
                rec["correct"] += 1
            if label == "correct" and truth != "correct":
                # False approval: signed off an incorrect or unfinished component.
                false_correct += 1
                comp["false_correct"] += 1
                rec["false_correct"] += 1
        if truth == "incorrect":
            # Abstaining on an incorrect component is not recalling it.
            incorrect_total += 1
            if label == "incorrect":
                incorrect_recalled += 1

    scorable = len(pairs)
    judged = scorable - abstentions
    report = {
        "n_verdicts": len(verdicts),
        "n_references": len(references),
        "scorable": scorable,
        "judged": judged,
        "excluded_checkpoint_ids": excluded,
        "state_accuracy": n_correct / scorable if scorable else None,
        "false_correct_count": false_correct,
        "incorrect_recall": (incorrect_recalled / incorrect_total
                             if incorrect_total else None),
        "incorrect_reference_count": incorrect_total,
        "abstention_count": abstentions,
        "model_calls": model_calls + run_meta.get("model_calls", 0),
        "latency_ms": latency_ms + run_meta.get("latency_ms", 0.0),
        "per_component": per_component,
        "per_recording": per_recording,
    }
    return report
