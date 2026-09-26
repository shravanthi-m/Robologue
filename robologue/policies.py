"""Person 3: candidate policy records, deterministic proposal template, promotion.

A candidate checklist rule is derived from development misses (never assumed),
compared against the frozen baseline on validation recordings under equal
budgets, and accepted only if it reduces false-correct count without
decreasing state accuracy or violating the inference budget. Ties and
regressions are rejected; insufficient reference data is inconclusive.
The final test recording is scored once, after the policy is frozen, and is
never used for selection.

Person 3 owns evaluation and promotion. Person 2 loads only the explicitly
selected immutable policy at run start.
"""

from .evaluate import match_pairs

STATUSES = ("candidate", "accepted", "rejected")
DECISIONS = ("accept", "reject", "inconclusive")

REQUIRED_POLICY_FIELDS = (
    "schema_version",
    "policy_id",
    "parent_policy_id",
    "status",
    "checklist",
    "supporting_dev_case_ids",
)

TEMPLATE_ID = "require-explicit-connection-evidence-v1"
TEMPLATE_TEXT = (
    "Before declaring {component} correct, require explicit visual evidence "
    "that its connection is present ({description}). An occluded or unseen "
    "attachment is insufficient_evidence, not correct."
)


class PolicyError(ValueError):
    """Raised for malformed policies or undecidable comparisons."""


def validate_policy(policy):
    missing = [f for f in REQUIRED_POLICY_FIELDS if f not in policy]
    if missing:
        raise PolicyError(f"Policy missing fields: {missing}")
    if policy["status"] not in STATUSES:
        raise PolicyError(f"Unknown status: {policy['status']!r}")
    if not isinstance(policy["checklist"], list) or len(policy["checklist"]) > 1:
        raise PolicyError("This release allows at most one added checklist instruction")
    if any(
        not isinstance(s, str) or not s.strip() or len(s) > 1500
        for s in policy["checklist"]
    ):
        raise PolicyError("Checklist instructions must be bounded nonempty strings")
    if not isinstance(policy["supporting_dev_case_ids"], list) or any(
        not isinstance(s, str) for s in policy["supporting_dev_case_ids"]
    ):
        raise PolicyError("Supporting development cases must be strings")
    return True


def propose_from_false_corrects(
    verdicts, references, policy_id, parent_policy_id, descriptions=None
):
    """Deterministic proposal template triggered by false-correct dev cases.

    Groups development false approvals (verdict "correct" on an incorrect or
    unfinished component) by component and emits one bounded rule for the
    worst component. Returns None when no eligible miss exists: no proposal is
    reported rather than manufacturing an improvement.
    """
    descriptions = descriptions or {}
    pairs, _ = match_pairs(verdicts, references)
    misses = [
        (v, r)
        for v, r in pairs
        if v["verdict"] == "correct" and r["reference"] != "correct"
    ]
    if not misses:
        return None
    by_component = {}
    for verdict, ref in misses:
        by_component.setdefault(verdict["component_id"], []).append((verdict, ref))
    # Deterministic: most misses first, ties broken by component id.
    target = sorted(by_component, key=lambda c: (-len(by_component[c]), c))[0]
    cases = sorted({v["checkpoint_id"] for v, _ in by_component[target]})
    description = descriptions.get(target, "the component is fully attached")
    candidate = {
        "schema_version": 1,
        "policy_id": policy_id,
        "parent_policy_id": parent_policy_id,
        "status": "candidate",
        "checklist": [TEMPLATE_TEXT.format(component=target, description=description)],
        "supporting_dev_case_ids": cases,
        "template_id": TEMPLATE_ID,
        "rationale": (
            f"{len(by_component[target])} false-correct development case(s) "
            f"on {target}; rule requires explicit connection evidence."
        ),
    }
    validate_policy(candidate)
    return candidate


def decide_promotion(baseline_report, candidate_report, max_model_calls):
    """Frozen promotion rule. Returns an audit record, always with reasons."""
    reasons = []
    if (
        "scored_case_keys" in baseline_report
        and "scored_case_keys" in candidate_report
        and baseline_report["scored_case_keys"] != candidate_report["scored_case_keys"]
    ):
        return {
            "decision": "reject",
            "reasons": [
                "Candidate and baseline must cover the identical paired cohort"
            ],
            "baseline": _summary(baseline_report),
            "candidate": _summary(candidate_report),
            "budget": max_model_calls,
        }
    judged_b = baseline_report["judged"]
    judged_c = candidate_report["judged"]
    if judged_b == 0:
        return {
            "decision": "inconclusive",
            "reasons": ["Insufficient reference data: baseline has no judged cases"],
            "baseline": _summary(baseline_report),
            "candidate": _summary(candidate_report),
            "budget": max_model_calls,
        }
    if judged_c == 0:
        # A candidate that produces nothing scorable (e.g. all-abstain) cannot
        # demonstrate improvement. This is a rejection, not inconclusive data.
        return {
            "decision": "reject",
            "reasons": ["Candidate produced no judged cases; abstention cannot win"],
            "baseline": _summary(baseline_report),
            "candidate": _summary(candidate_report),
            "budget": max_model_calls,
        }

    fc_b = baseline_report["false_correct_count"]
    fc_c = candidate_report["false_correct_count"]
    acc_b = baseline_report["state_accuracy"]
    acc_c = candidate_report["state_accuracy"]
    calls_c = candidate_report["model_calls"]

    checks = []
    if fc_c < fc_b:
        reasons.append(f"False-corrects fell {fc_b} -> {fc_c}")
        checks.append(True)
    else:
        reasons.append(f"False-corrects did not fall ({fc_b} -> {fc_c}); ties rejected")
        checks.append(False)
    if acc_c is not None and acc_b is not None and acc_c >= acc_b:
        reasons.append(f"State accuracy held {acc_b:.3f} -> {acc_c:.3f}")
        checks.append(True)
    else:
        reasons.append(f"State accuracy decreased ({acc_b} -> {acc_c})")
        checks.append(False)
    if calls_c <= max_model_calls:
        reasons.append(
            f"Inference budget respected ({calls_c} <= {max_model_calls} calls)"
        )
        checks.append(True)
    else:
        reasons.append(
            f"Inference budget violated ({calls_c} > {max_model_calls} calls)"
        )
        checks.append(False)

    decision = "accept" if all(checks) else "reject"
    return {
        "decision": decision,
        "reasons": reasons,
        "baseline": _summary(baseline_report),
        "candidate": _summary(candidate_report),
        "budget": max_model_calls,
    }


def _summary(report):
    return {
        "state_accuracy": report["state_accuracy"],
        "false_correct_count": report["false_correct_count"],
        "incorrect_recall": report["incorrect_recall"],
        "abstention_count": report["abstention_count"],
        "judged": report["judged"],
        "model_calls": report["model_calls"],
    }


def finalize(candidate, decision):
    """Freeze the selected policy after validation. Returns a new record."""
    if decision not in ("accept", "reject"):
        raise PolicyError(f"Cannot finalize with decision {decision!r}")
    frozen = dict(candidate)
    frozen["status"] = "accepted" if decision == "accept" else "rejected"
    validate_policy(frozen)
    return frozen
