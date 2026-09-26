"""Person 3: scorer / policy tests. All predictions are dummies; no agent exists yet."""
import json
import tempfile
import unittest
from pathlib import Path

from robologue.evaluate import EvaluationError, match_pairs, score, validate_verdict
from robologue.industreal_labels import (
    LabelError,
    load_procedure_info,
    reference_records,
)
from robologue.policies import (
    PolicyError,
    decide_promotion,
    finalize,
    propose_from_false_corrects,
    validate_policy,
)

TRUTH = {"incorrect": -1, "not_completed": 0, "correct": 1}


def ref(rid, comp, frame, truth):
    return {"recording_id": rid, "component_id": comp, "frame": frame,
            "frame_name": f"{frame:06d}.jpg", "raw_state": TRUTH[truth],
            "reference": truth, "description": "", "evaluation_only": True}


def verdict(rid, comp, frame, label, **extra):
    base = {"schema_version": 1, "source_kind": "mock", "run_id": "mock-baseline",
            "recording_id": rid, "checkpoint_id": f"{rid}:{comp}:{frame}",
            "cursor_frame": frame, "component_id": comp, "verdict": label,
            "policy_id": "baseline-v1", "model_id": "mock"}
    base.update(extra)
    return base


REFS = [ref("rec-A", "component-0", 0, "correct"),
        ref("rec-A", "component-0", 1, "incorrect"),
        ref("rec-A", "component-0", 2, "not_completed"),
        ref("rec-A", "component-1", 0, "correct")]


class LabelAdapterTests(unittest.TestCase):
    def _csv(self, directory, rows):
        path = Path(directory) / "PSR_labels_raw.csv"
        path.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
        return path

    def test_parses_official_no_header_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory, [["000000.jpg", 1, 0],
                                         ["000001.jpg", -1, 0],
                                         ["000002.jpg", 1, 1]])
            records = list(reference_records("rec-A", path, components=[0]))
            self.assertEqual([r["reference"] for r in records],
                             ["correct", "incorrect", "correct"])
            self.assertEqual(records[1]["frame"], 1)
            self.assertTrue(all(r["evaluation_only"] for r in records))

    def test_component_filter_and_procedure_info(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory, [["000000.jpg", 1, 0]])
            info = Path(directory) / "procedure_info.json"
            info.write_text(json.dumps([
                {"id": 0, "description": "Install base", "install": True, "state_idx": 0},
                {"id": 3, "description": "Install wheel", "install": True, "state_idx": 1}]))
            self.assertEqual(load_procedure_info(info), {0: "Install base", 1: "Install wheel"})
            records = list(reference_records("rec-A", path, info, components=[1]))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["description"], "Install wheel")
            self.assertEqual(records[0]["reference"], "not_completed")

    def test_bad_states_and_widths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(LabelError):
                list(reference_records("r", self._csv(directory, [["000000.jpg", 2]])))
            with self.assertRaises(LabelError):
                list(reference_records("r", self._csv(directory,
                                                      [["000000.jpg", 1], ["000001.jpg", 1, 0]])))
            with self.assertRaises(LabelError):
                list(reference_records("r", self._csv(directory, [["frameX.jpg", 1]])))


class ScoringTests(unittest.TestCase):
    def test_perfect_dummy(self):
        verdicts = [verdict("rec-A", "component-0", 0, "correct"),
                    verdict("rec-A", "component-0", 1, "incorrect"),
                    verdict("rec-A", "component-0", 2, "not_completed"),
                    verdict("rec-A", "component-1", 0, "correct")]
        report = score(verdicts, REFS)
        self.assertEqual(report["state_accuracy"], 1.0)
        self.assertEqual(report["false_correct_count"], 0)
        self.assertEqual(report["incorrect_recall"], 1.0)
        self.assertEqual(report["abstention_count"], 0)

    def test_false_correct_and_miss(self):
        verdicts = [verdict("rec-A", "component-0", 0, "correct"),
                    verdict("rec-A", "component-0", 1, "correct"),  # false approval
                    verdict("rec-A", "component-0", 2, "not_completed"),
                    verdict("rec-A", "component-1", 0, "correct")]
        report = score(verdicts, REFS)
        self.assertEqual(report["false_correct_count"], 1)
        self.assertEqual(report["incorrect_recall"], 0.0)
        self.assertEqual(report["state_accuracy"], 0.75)

    def test_not_completed_is_not_an_error_shortcut(self):
        # Approving unfinished work is a false-correct, not a neutral miss.
        verdicts = [verdict("rec-A", "component-0", 2, "correct")]
        report = score(verdicts, [ref("rec-A", "component-0", 2, "not_completed")])
        self.assertEqual(report["false_correct_count"], 1)
        self.assertEqual(report["state_accuracy"], 0.0)
        # Calling unfinished work "incorrect" is wrong, but not a false approval.
        verdicts = [verdict("rec-A", "component-0", 2, "incorrect")]
        report = score(verdicts, [ref("rec-A", "component-0", 2, "not_completed")])
        self.assertEqual(report["false_correct_count"], 0)
        self.assertEqual(report["state_accuracy"], 0.0)

    def test_abstention_never_counts_as_correct(self):
        verdicts = [verdict("rec-A", "component-0", f, "insufficient_evidence")
                    for f in (0, 1, 2)]
        refs = REFS[:3]
        report = score(verdicts, refs)
        self.assertEqual(report["state_accuracy"], 0.0)
        self.assertEqual(report["abstention_count"], 3)
        self.assertEqual(report["incorrect_recall"], 0.0)  # incorrect ref exists, abstained: not recalled

    def test_recall_unavailable_without_incorrect_references(self):
        refs = [ref("rec-A", "component-0", 0, "correct")]
        report = score([verdict("rec-A", "component-0", 0, "correct")], refs)
        self.assertIsNone(report["incorrect_recall"])
        self.assertEqual(report["state_accuracy"], 1.0)

    def test_same_frame_other_recording_cannot_match(self):
        verdicts = [verdict("rec-B", "component-0", 0, "correct")]
        report = score(verdicts, REFS)
        self.assertEqual(report["scorable"], 0)
        self.assertEqual(report["excluded_checkpoint_ids"], ["rec-B:component-0:0"])
        self.assertIsNone(report["state_accuracy"])

    def test_bad_verdict_rejected(self):
        good = verdict("rec-A", "component-0", 0, "correct")
        self.assertTrue(validate_verdict(good))
        with self.assertRaises(EvaluationError):
            validate_verdict(dict(good, verdict="maybe"))
        with self.assertRaises(EvaluationError):
            validate_verdict({k: v for k, v in good.items() if k != "policy_id"})

    def test_model_calls_and_latency_summed(self):
        verdicts = [verdict("rec-A", "component-0", 0, "correct",
                            model_calls=1, latency_ms=120.5)]
        report = score(verdicts, REFS[:1], run_meta={"model_calls": 10})
        self.assertEqual(report["model_calls"], 11)
        self.assertAlmostEqual(report["latency_ms"], 120.5)


def _report(acc, fc, judged=4, calls=8, recall=1.0, abst=0):
    return {"state_accuracy": acc, "false_correct_count": fc,
            "incorrect_recall": recall, "abstention_count": abst,
            "judged": judged, "model_calls": calls}


class PromotionTests(unittest.TestCase):
    def test_accept_real_improvement(self):
        result = decide_promotion(_report(0.75, 2), _report(0.80, 1), 24)
        self.assertEqual(result["decision"], "accept")
        self.assertTrue(result["reasons"])

    def test_reject_tie_on_false_corrects(self):
        result = decide_promotion(_report(0.75, 1), _report(0.90, 1), 24)
        self.assertEqual(result["decision"], "reject")

    def test_reject_accuracy_regression(self):
        result = decide_promotion(_report(0.80, 2), _report(0.70, 1), 24)
        self.assertEqual(result["decision"], "reject")

    def test_reject_budget_violation(self):
        result = decide_promotion(_report(0.75, 2), _report(0.80, 1, calls=30), 24)
        self.assertEqual(result["decision"], "reject")

    def test_inconclusive_without_judged_cases(self):
        result = decide_promotion(_report(None, 0, judged=0), _report(0.8, 1), 24)
        self.assertEqual(result["decision"], "inconclusive")

    def test_all_abstain_candidate_cannot_win(self):
        refs = REFS[:3]
        abstain = [verdict("rec-A", "component-0", f, "insufficient_evidence")
                   for f in (0, 1, 2)]
        cand = score(abstain, refs)
        base = score([verdict("rec-A", "component-0", 0, "correct"),
                      verdict("rec-A", "component-0", 1, "incorrect"),
                      verdict("rec-A", "component-0", 2, "not_completed")], refs)
        result = decide_promotion(base, cand, 24)
        self.assertEqual(result["decision"], "reject")

    def test_finalize_freezes_status(self):
        candidate = {"schema_version": 1, "policy_id": "p2", "parent_policy_id": "p1",
                     "status": "candidate", "checklist": ["rule"],
                     "supporting_dev_case_ids": ["c1"]}
        self.assertEqual(finalize(candidate, "accept")["status"], "accepted")
        self.assertEqual(finalize(candidate, "reject")["status"], "rejected")
        with self.assertRaises(PolicyError):
            finalize(candidate, "maybe")


class ProposalTemplateTests(unittest.TestCase):
    def test_proposes_from_false_corrects(self):
        verdicts = [verdict("rec-A", "component-0", 1, "correct"),   # false approval
                    verdict("rec-A", "component-0", 0, "correct")]
        candidate = propose_from_false_corrects(
            verdicts, REFS, "checklist-v2", "baseline-v1",
            descriptions={"component-0": "the wheel is seated on the axle"})
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["status"], "candidate")
        self.assertEqual(len(candidate["checklist"]), 1)
        self.assertIn("insufficient_evidence", candidate["checklist"][0])
        self.assertEqual(candidate["supporting_dev_case_ids"], ["rec-A:component-0:1"])
        self.assertEqual(candidate["parent_policy_id"], "baseline-v1")

    def test_no_proposal_without_eligible_miss(self):
        verdicts = [verdict("rec-A", "component-0", 0, "correct"),
                    verdict("rec-A", "component-0", 1, "incorrect")]
        self.assertIsNone(propose_from_false_corrects(
            verdicts, REFS, "checklist-v2", "baseline-v1"))

    def test_deterministic_component_choice(self):
        refs = [ref("rec-A", "component-0", 1, "incorrect"),
                ref("rec-A", "component-1", 0, "incorrect")]
        verdicts = [verdict("rec-A", "component-1", 0, "correct"),  # false approval
                    verdict("rec-A", "component-0", 1, "correct")]  # false approval
        first = propose_from_false_corrects(verdicts, refs, "v2", "v1")
        second = propose_from_false_corrects(list(reversed(verdicts)), refs, "v2", "v1")
        self.assertEqual(first["checklist"], second["checklist"])


if __name__ == "__main__":
    unittest.main()


class LabelAdapterEdgeTests(unittest.TestCase):
    def _csv_text(self, directory, text):
        path = Path(directory) / "PSR_labels_raw.csv"
        path.write_text(text)
        return path

    def test_blank_lines_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv_text(directory, "000000.jpg,1\n\n   \n000001.jpg,0\n")
            records = list(reference_records("rec-A", path, components=[0]))
            self.assertEqual([r["reference"] for r in records],
                             ["correct", "not_completed"])

    def test_empty_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv_text(directory, "")
            with self.assertRaises(LabelError):
                list(reference_records("r", path))

    def test_install_description_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            info = Path(directory) / "procedure_info.json"
            info.write_text(json.dumps([
                {"id": 0, "description": "Remove base", "install": False, "state_idx": 0},
                {"id": 1, "description": "Install base", "install": True, "state_idx": 0}]))
            self.assertEqual(load_procedure_info(info), {0: "Install base"})

    def test_frame_number_from_any_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv_text(directory, "000042.png,1\n")
            records = list(reference_records("rec-A", path, components=[0]))
            self.assertEqual(records[0]["frame"], 42)
            self.assertEqual(records[0]["frame_name"], "000042.png")


class ScoringEdgeTests(unittest.TestCase):
    def test_duplicate_reference_keys_rejected(self):
        refs = [ref("rec-A", "component-0", 0, "correct"),
                ref("rec-A", "component-0", 0, "incorrect")]
        with self.assertRaises(EvaluationError):
            score([verdict("rec-A", "component-0", 0, "correct")], refs)

    def test_negative_cursor_frame_rejected(self):
        with self.assertRaises(EvaluationError):
            validate_verdict(verdict("rec-A", "component-0", -1, "correct"))

    def test_non_int_cursor_frame_rejected(self):
        good = verdict("rec-A", "component-0", 0, "correct")
        with self.assertRaises(EvaluationError):
            validate_verdict(dict(good, cursor_frame="0"))

    def test_empty_verdicts_score_none(self):
        report = score([], REFS)
        self.assertEqual(report["scorable"], 0)
        self.assertEqual(report["excluded_checkpoint_ids"], [])
        self.assertIsNone(report["state_accuracy"])

    def test_per_component_breakdown(self):
        verdicts = [verdict("rec-A", "component-0", 0, "correct"),
                    verdict("rec-A", "component-0", 1, "correct")]  # false approval
        report = score(verdicts, REFS[:2])
        self.assertEqual(report["per_component"]["component-0"],
                         {"scorable": 2, "correct": 1, "false_correct": 1})
        self.assertEqual(report["per_recording"]["rec-A"]["scorable"], 2)

    def test_excluded_verdicts_do_not_pollute_breakdown(self):
        verdicts = [verdict("rec-B", "component-9", 7, "correct")]
        report = score(verdicts, REFS)
        self.assertEqual(report["per_component"], {})
        self.assertEqual(report["per_recording"], {})


class PolicyEdgeTests(unittest.TestCase):
    def _candidate(self, **overrides):
        base = {"schema_version": 1, "policy_id": "p2", "parent_policy_id": "p1",
                "status": "candidate", "checklist": ["rule"],
                "supporting_dev_case_ids": ["c1"]}
        base.update(overrides)
        return base

    def test_rejects_two_checklist_items(self):
        with self.assertRaises(PolicyError):
            validate_policy(self._candidate(checklist=["a", "b"]))

    def test_rejects_bad_status(self):
        with self.assertRaises(PolicyError):
            validate_policy(self._candidate(status="draft"))

    def test_rejects_missing_field(self):
        candidate = self._candidate()
        del candidate["policy_id"]
        with self.assertRaises(PolicyError):
            validate_policy(candidate)

    def test_propose_uses_default_description(self):
        verdicts = [verdict("rec-A", "component-0", 1, "correct")]
        candidate = propose_from_false_corrects(
            verdicts, REFS, "checklist-v2", "baseline-v1")
        self.assertIn("fully attached", candidate["checklist"][0])

    def test_finalize_returns_new_record(self):
        candidate = self._candidate()
        frozen = finalize(candidate, "accept")
        self.assertEqual(frozen["status"], "accepted")
        self.assertEqual(candidate["status"], "candidate")

    def test_promotion_reasons_mention_counts(self):
        result = decide_promotion(_report(0.75, 2), _report(0.80, 1), 24)
        self.assertIn("2 -> 1", result["reasons"][0])
        self.assertEqual(len(result["reasons"]), 3)
