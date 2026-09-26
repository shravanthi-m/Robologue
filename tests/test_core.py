import tempfile
import unittest
from pathlib import Path

from robologue.captaincook import evaluation_records
from robologue.harness import advance, replay
from robologue.store import SQLiteStore


def event(n, **extra):
    return {"event_id": str(n), "recording_id": "r", "timestamp": n, "observation": "Observed action", **extra}


class MemoryTests(unittest.TestCase):
    def test_restart_retains_issue_and_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite"
            first = event(1, issue={"id": "x", "description": "Unverified"})
            store = SQLiteStore(path)
            list(replay(store, "s", [first]))
            store.close()
            store = SQLiteStore(path)
            decisions = list(replay(store, "s", [first, event(2)]))
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["action"], "request_verification")
            self.assertIn("x", decisions[0]["open_issues"])
            store.close()

    def test_resolution_requires_explicit_evidence_event(self):
        state, _ = advance(None, event(1, issue={"id": "x", "description": "Unverified"}))
        state, decision = advance(state, event(2))
        self.assertIn("x", decision["open_issues"])
        _, decision = advance(state, event(3, resolves=["x"]))
        self.assertEqual(decision["open_issues"], {})

    def test_stateless_baseline_forgets(self):
        state, _ = advance(None, event(1, issue={"id": "x", "description": "Unverified"}), "stateless")
        _, decision = advance(state, event(2), "stateless")
        self.assertEqual(decision["open_issues"], {})

    def test_labels_rejected(self):
        for field in ("errors", "has_errors", "modified_description", "is_error"):
            with self.assertRaises(ValueError):
                advance(None, event(1, **{field: True}))

    def test_invalid_replay(self):
        state, _ = advance(None, event(2))
        for invalid in (event(1), event(2, observation="Changed"), event(3, recording_id="other")):
            with self.assertRaises(ValueError):
                advance(state, invalid)

    def test_missing_step_stays_evaluator_only(self):
        data = [{"recording_id": "1_10", "step_annotations": [
            {"step_id": 4, "start_time": -1, "end_time": -1, "errors": [{"tag": "Missing Step"}]},
            {"step_id": 4, "start_time": 2, "end_time": 3, "errors": []}]}]
        records = list(evaluation_records(data))
        self.assertTrue(records[0]["unlocalized"])
        self.assertIsNone(records[0]["start_time"])
        self.assertNotEqual(records[0]["annotation_id"], records[1]["annotation_id"])
        with self.assertRaises(ValueError):
            advance(None, records[0])

    def test_nonfinite_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            advance(None, event(1, timestamp=float("nan")))


if __name__ == "__main__":
    unittest.main()


class PolicyLoadingTests(unittest.TestCase):
    def _policy(self, **overrides):
        base = {"schema_version": 1, "policy_id": "checklist-v2",
                "parent_policy_id": "baseline-v1", "status": "accepted",
                "checklist": ["Require explicit connection evidence."],
                "supporting_dev_case_ids": ["c1"]}
        base.update(overrides)
        return base

    def test_checklist_rides_on_verification_requests(self):
        policy = self._policy()
        state, decision = advance(None, event(1, issue={"id": "x", "description": "Unverified"}),
                                  policy=policy)
        self.assertEqual(decision["action"], "request_verification")
        self.assertEqual(decision["verification_checklist"], policy["checklist"])
        self.assertEqual(decision["policy_id"], "checklist-v2")

    def test_no_checklist_without_policy(self):
        _, decision = advance(None, event(1, issue={"id": "x", "description": "Unverified"}))
        self.assertNotIn("verification_checklist", decision)
        self.assertNotIn("policy_id", decision)

    def test_no_checklist_when_nothing_to_verify(self):
        policy = self._policy()
        _, decision = advance(None, event(1), policy=policy)
        self.assertEqual(decision["action"], "continue_observing")
        self.assertNotIn("verification_checklist", decision)

    def test_load_policy_accepts_report_and_record(self):
        import json
        from robologue.cli import load_policy
        with tempfile.TemporaryDirectory() as directory:
            record_path = Path(directory) / "policy.json"
            record_path.write_text(json.dumps(self._policy()))
            self.assertEqual(load_policy(record_path)["policy_id"], "checklist-v2")
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({"frozen_policy": self._policy()}))
            self.assertEqual(load_policy(report_path)["policy_id"], "checklist-v2")

    def test_load_policy_rejects_non_accepted(self):
        import json
        from robologue.cli import load_policy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(self._policy(status="candidate")))
            with self.assertRaises(ValueError):
                load_policy(path)
            path.write_text(json.dumps({"frozen_policy": None}))
            with self.assertRaises(ValueError):
                load_policy(path)
