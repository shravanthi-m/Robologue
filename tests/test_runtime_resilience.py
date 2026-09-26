"""Offline fault injection for provider accounting and durable RGB checkpoints."""

import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from robologue.budget import BudgetExceeded
from robologue.model_client import CALL_RESERVATION, CallBudget, DEFAULT_MODEL, OpenRouterClient
from robologue.pipeline import run_recording
from robologue.store import SQLiteStore


SECRET = "fake-api-key-must-never-appear"
UNTRUSTED = "provider-private-message-must-never-appear"


def response(content='{"ok": true}', *, usage=None, finish_reason="stop"):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"cost": 0.003} if usage is None else usage,
    }


class ProviderResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.sqlite"
        self.budget = CallBudget(self.path)
        self.addCleanup(lambda: self.budget.close())
        self.client = OpenRouterClient(self.budget, api_key=SECRET)

    def dispatch(self, raw):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as http:
            try:
                return self.client.chat(DEFAULT_MODEL, "system", {})
            finally:
                self.assertEqual(http.call_count, 1)

    def assert_safe_failure(self, raw, *, stage, cost=0.003, uncertain=0):
        with self.assertRaisesRegex(ValueError, stage) as error:
            self.dispatch(raw)
        self.assertNotIn(SECRET, str(error.exception))
        self.assertNotIn(UNTRUSTED, str(error.exception))
        attempt = self.budget.store.load(self.budget.session)["attempts"][-1]
        self.assertEqual(attempt["status"], "failed")
        self.assertEqual(attempt["failure_kind"], stage)
        self.assertEqual(attempt["accounted_usd"], cost)
        self.assertEqual(self.budget.summary()["uncertain_charge_calls"], uncertain)
        self.assertNotIn(SECRET, json.dumps(attempt))
        self.assertNotIn(UNTRUSTED, json.dumps(attempt))

    def test_provider_error_payload_is_sanitized_and_retains_unknown_charge(self):
        self.assert_safe_failure(
            {"error": {"message": UNTRUSTED, "metadata": {"raw": SECRET}}},
            stage="response_schema", cost=CALL_RESERVATION, uncertain=1,
        )

    def test_invalid_outer_json_is_sanitized_and_retains_unknown_charge(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(UNTRUSTED.encode())) as http:
            with self.assertRaisesRegex(ValueError, "transport") as error:
                self.client.chat(DEFAULT_MODEL, "system", {})
        self.assertEqual(http.call_count, 1)
        self.assertNotIn(UNTRUSTED, str(error.exception))
        self.assertEqual(self.budget.summary()["accounted_usd"], CALL_RESERVATION)
        self.assertEqual(self.budget.summary()["failed_calls"], 1)

    def test_http_error_does_not_expose_body_reason_or_key(self):
        failure = urllib.error.HTTPError(
            "https://provider.invalid/" + SECRET, 429, UNTRUSTED,
            {}, io.BytesIO((SECRET + UNTRUSTED).encode()),
        )
        with patch("urllib.request.urlopen", side_effect=failure) as http:
            with self.assertRaisesRegex(ValueError, "HTTP 429") as error:
                self.client.chat(DEFAULT_MODEL, "system", {})
        self.assertEqual(http.call_count, 1)
        self.assertNotIn(SECRET, str(error.exception))
        self.assertNotIn(UNTRUSTED, str(error.exception))
        self.assertEqual(self.budget.summary()["failed_calls"], 1)
        self.assertEqual(self.budget.summary()["accounted_usd"], CALL_RESERVATION)

    def test_transport_failure_status_and_unknown_charge_survive_restart(self):
        self.budget.close()
        self.budget = CallBudget(self.path)
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(UNTRUSTED)) as http:
            with self.assertRaisesRegex(ValueError, "transport") as error:
                self.client = OpenRouterClient(self.budget, api_key=SECRET)
                self.client.chat(DEFAULT_MODEL, "system", {})
        self.assertEqual(http.call_count, 1)
        self.assertNotIn(UNTRUSTED, str(error.exception))
        self.budget.close()
        self.budget = CallBudget(self.path)
        entry = self.budget.store.load(self.budget.session)["attempts"][0]
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["failure_kind"], "transport")
        self.assertEqual(self.budget.summary()["uncertain_charge_calls"], 1)
        self.assertEqual(self.budget.summary()["accounted_usd"], CALL_RESERVATION)

    def test_billed_invalid_json_keeps_settled_cost_tokens_and_failure_on_restart(self):
        raw = response(UNTRUSTED, usage={"cost": 0.003, "prompt_tokens": 12, "completion_tokens": 8})
        self.assert_safe_failure(raw, stage="invalid_json")
        self.budget.close()
        self.budget = CallBudget(self.path)
        entry = self.budget.store.load(self.budget.session)["attempts"][0]
        self.assertEqual(entry["prompt_tokens"], 12)
        self.assertEqual(entry["completion_tokens"], 8)
        self.assertTrue(entry["provider_reported_cost"])
        self.assertEqual(self.budget.summary()["failed_calls"], 1)
        self.assertEqual(self.budget.summary()["reported_cost_usd"], 0.003)

    def test_empty_content_keeps_settled_charge(self):
        self.assert_safe_failure(response(""), stage="empty_content")

    def test_incomplete_completion_keeps_settled_charge(self):
        self.assert_safe_failure(response(finish_reason="length"), stage="completion_length")

    def test_provider_finish_reason_is_not_copied_into_error_or_persistence(self):
        raw = response(UNTRUSTED, finish_reason=UNTRUSTED)
        self.assert_safe_failure(raw, stage="completion_unknown")

    def test_missing_choices_keeps_settled_charge(self):
        self.assert_safe_failure({"usage": {"cost": 0.003}}, stage="response_schema")

    def test_non_object_choice_is_a_sanitized_persistent_failure(self):
        for choice in [None, [], UNTRUSTED, 7]:
            with self.subTest(choice=choice):
                self.assert_safe_failure({"choices": [choice], "usage": {"cost": 0.003}}, stage="response_schema")

    def test_empty_choices_and_missing_message_are_sanitized_billed_failures(self):
        for choices in [[], [{}], [{"message": {}}], [{"message": None}]]:
            with self.subTest(choices=choices):
                self.assert_safe_failure({"choices": choices, "usage": {"cost": 0.003}}, stage=(
                    "response_schema" if not choices else "completion_unknown"
                ))

    def test_non_object_provider_root_keeps_unknown_reservations(self):
        for count, raw in enumerate([None, [], UNTRUSTED, 7], start=1):
            with self.subTest(raw=raw):
                self.assert_safe_failure(raw, stage="response_schema", cost=CALL_RESERVATION, uncertain=count)

    def test_invalid_reported_cost_is_never_treated_as_a_refund(self):
        for count, cost in enumerate([None, -1, True, float("nan"), float("inf"), "0.001"], start=1):
            with self.subTest(cost=cost):
                self.assert_safe_failure(response(UNTRUSTED, usage={"cost": cost}), stage="invalid_json",
                                         cost=CALL_RESERVATION, uncertain=count)

    def test_known_zero_cost_is_preserved_when_content_is_invalid(self):
        self.assert_safe_failure(response(UNTRUSTED, usage={"cost": 0}), stage="invalid_json", cost=0)

    def test_non_object_usage_keeps_unknown_charge(self):
        self.assert_safe_failure(response(usage=[UNTRUSTED]), stage="response_schema", cost=CALL_RESERVATION, uncertain=1)

    def test_abandoned_reservation_cannot_be_refunded_by_restart(self):
        self.budget.close()
        path = Path(self.temp.name) / "small.sqlite"
        self.budget = CallBudget(path, max_usd=0.04, max_calls=1)
        self.budget.reserve("interrupted-before-response")
        self.budget.close()
        self.budget = CallBudget(path, max_usd=0.04, max_calls=1)
        client = OpenRouterClient(self.budget, api_key=SECRET)
        with patch("urllib.request.urlopen") as http:
            with self.assertRaises(BudgetExceeded):
                client.chat(DEFAULT_MODEL, "system", {})
        http.assert_not_called()
        self.assertEqual(self.budget.summary()["dispatched_calls"], 1)
        self.assertEqual(self.budget.summary()["uncertain_charge_calls"], 1)


class OfflineVision:
    """Deterministic inference with real reservation and settlement writes."""
    def __init__(self, budget):
        self.budget = budget
        self.contexts = []

    def chat(self, model, system, context, images=(), *, max_tokens=2000):
        self.contexts.append(copy.deepcopy(context))
        attempt = self.budget.reserve("offline-" + str(len(self.contexts)))
        self.budget.finish(attempt, {"cost": 0.001})
        if "components" in context:
            value = {"components": [
                {"component_id": name, "verdict": "correct", "rationale": "Visible current attachment.",
                 "evidence_frame_ids": [context["current_frame"]]}
                for name in context["components"]
            ]}
        else:
            value = {"observations": ["Assembly visible."], "uncertainties": []}
        return value, {"latency_seconds": 0, "usage": {"cost": 0.001}}


class CheckpointResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = self.root / "rgb"
        self.rgb.mkdir()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(self.rgb / "000000.jpg")
        self.store_path = self.root / "runtime.sqlite"
        self.store = SQLiteStore(self.store_path)
        self.addCleanup(lambda: self.store.close())
        self.budget = CallBudget(self.root / "budget.sqlite")
        self.addCleanup(self.budget.close)
        self.client = OfflineVision(self.budget)
        self.session = "pipeline:resilience:memory:sample"
        self.components = {"component-0": "Public base"}
        self.mapping = {"schema_version": 1, "verified": True, "recording_id": "sample",
                        "mapping_id": "offline", "frame_seconds": {"0": 0}}
        self.policy = {"schema_version": 1, "policy_id": "pinned-v2", "parent_policy_id": "baseline-v1",
                       "status": "accepted", "checklist": ["Require visible connection."],
                       "supporting_dev_case_ids": ["dev:0"]}

    def run_pipeline(self, **kwargs):
        return run_recording(self.store, "resilience", "sample", self.rgb, [0], self.components,
                             self.mapping, self.client, self.root / "cache", **kwargs)

    def restart_store(self):
        self.store.close()
        self.store = SQLiteStore(self.store_path)

    def test_reopen_flushes_unprojected_outbox_without_new_inference(self):
        with patch.object(self.store, "put_decision", side_effect=RuntimeError("before projection")):
            with self.assertRaises(RuntimeError):
                self.run_pipeline()
        state = self.store.load(self.session)
        self.assertEqual(state["cursor"], 1)
        self.assertIsNotNone(state["pending"])
        self.assertEqual(self.store.decisions(self.session), [])
        calls = self.budget.summary()["dispatched_calls"]
        self.restart_store()
        result = self.run_pipeline()
        self.assertTrue(result["complete"])
        self.assertFalse(result["pending"])
        self.assertEqual(len(result["verdicts"]), 1)
        self.assertEqual(self.budget.summary()["dispatched_calls"], calls)

    def test_reopen_replays_already_projected_outbox_idempotently(self):
        original = self.store.put_decision
        def write_then_crash(session, decision):
            original(session, decision)
            raise RuntimeError("after projection before acknowledgement")
        with patch.object(self.store, "put_decision", side_effect=write_then_crash):
            with self.assertRaises(RuntimeError):
                self.run_pipeline()
        saved = self.store.decisions(self.session)
        self.assertEqual(len(saved), 1)
        calls = self.budget.summary()["dispatched_calls"]
        self.restart_store()
        result = self.run_pipeline()
        self.assertEqual(result["decisions"], saved)
        self.assertFalse(result["pending"])
        self.assertEqual(self.budget.summary()["dispatched_calls"], calls)

    def test_policy_change_is_rejected_before_pending_outbox_is_flushed(self):
        with patch.object(self.store, "put_decision", side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError):
                self.run_pipeline(policy=self.policy)
        changed = copy.deepcopy(self.policy)
        changed["checklist"] = ["A changed rule with the same policy ID."]
        calls = self.budget.summary()["dispatched_calls"]
        self.restart_store()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_pipeline(policy=changed)
        self.assertEqual(self.store.decisions(self.session), [])
        self.assertIsNotNone(self.store.load(self.session)["pending"])
        self.assertEqual(self.budget.summary()["dispatched_calls"], calls)
        self.assertTrue(self.run_pipeline(policy=self.policy)["complete"])

    def test_full_policy_content_is_pinned_across_completed_resume(self):
        self.run_pipeline(policy=self.policy)
        calls = self.budget.summary()["dispatched_calls"]
        for field, value in [("policy_id", "different-id"), ("checklist", ["Changed checklist."]),
                             ("supporting_dev_case_ids", ["dev:1"]), ("status", "candidate")]:
            with self.subTest(field=field):
                changed = dict(self.policy, **{field: value})
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.run_pipeline(policy=changed, allow_candidate=True)
        self.assertEqual(self.budget.summary()["dispatched_calls"], calls)

    def test_rejected_policy_never_dispatches_even_with_candidate_permission(self):
        with self.assertRaisesRegex(ValueError, "Only accepted policies"):
            self.run_pipeline(policy=dict(self.policy, status="rejected"), allow_candidate=True)
        self.assertEqual(self.budget.summary()["dispatched_calls"], 0)
        self.assertIsNone(self.store.load(self.session))

    def test_verifier_failure_abstention_and_safe_rationale_survive_restart(self):
        def failure(*args, **kwargs):
            raise ValueError(SECRET + UNTRUSTED)
        first = self.run_pipeline(verify_fn=failure)
        calls = self.budget.summary()["dispatched_calls"]
        self.restart_store()
        result = self.run_pipeline(verify_fn=failure)
        self.assertEqual(result["verdicts"], first["verdicts"])
        self.assertEqual(result["verdicts"][0]["verdict"], "insufficient_evidence")
        self.assertTrue(result["verdicts"][0]["provider_failed"])
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn(UNTRUSTED, json.dumps(result))
        self.assertEqual(self.budget.summary()["dispatched_calls"], calls)


if __name__ == "__main__":
    unittest.main()
