"""Offline fault injection for provider accounting and durable RGB checkpoints."""

import copy
import io
from http.client import HTTPException, IncompleteRead
import ssl
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from robologue.benchmark import run_benchmark
from robologue.verifier import verify
from robologue.perception.video import digest_file
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

    def test_tls_and_connection_dispatch_errors_are_safe_persistent_failures(self):
        failures = [ssl.SSLError(UNTRUSTED), ConnectionError(UNTRUSTED),
                    HTTPException(UNTRUSTED)]
        for count, failure in enumerate(failures, start=1):
            with self.subTest(failure=type(failure).__name__):
                with patch("urllib.request.urlopen", side_effect=failure) as http:
                    with self.assertRaisesRegex(ValueError, "transport") as error:
                        self.client.chat(DEFAULT_MODEL, "system", {})
                self.assertEqual(http.call_count, 1)
                self.assertNotIn(UNTRUSTED, str(error.exception))
                entry = self.budget.store.load(self.budget.session)["attempts"][-1]
                self.assertEqual(entry["status"], "failed")
                self.assertEqual(entry["failure_kind"], "transport")
                self.assertEqual(self.budget.summary()["uncertain_charge_calls"], count)
                self.assertAlmostEqual(self.budget.summary()["accounted_usd"], count * CALL_RESERVATION)

    def test_tls_and_truncated_response_reads_preserve_unknown_charges(self):
        class BrokenRead(io.BytesIO):
            def __init__(self, failure):
                super().__init__()
                self.failure = failure
            def read(self, *args, **kwargs):
                raise self.failure
        for count, failure in enumerate([ssl.SSLError(UNTRUSTED),
                                         IncompleteRead(UNTRUSTED.encode(), 100)], start=1):
            with self.subTest(failure=type(failure).__name__):
                with patch("urllib.request.urlopen", return_value=BrokenRead(failure)) as http:
                    with self.assertRaisesRegex(ValueError, "transport") as error:
                        self.client.chat(DEFAULT_MODEL, "system", {})
                self.assertEqual(http.call_count, 1)
                self.assertNotIn(UNTRUSTED, str(error.exception))
                self.assertEqual(self.budget.summary()["failed_calls"], count)
                self.assertEqual(self.budget.summary()["uncertain_charge_calls"], count)
                self.assertAlmostEqual(self.budget.summary()["accounted_usd"], count * CALL_RESERVATION)

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
        self.images = []
        self.systems = []
        self.image_labels = []

    def chat(self, model, system, context, images=(), *, max_tokens=2000, image_labels=None):
        self.contexts.append(copy.deepcopy(context))
        self.images.append(list(images))
        self.systems.append(system)
        self.image_labels.append(image_labels)
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

    def test_reference_change_is_rejected_before_completed_resume_dispatch(self):
        reference = self.root / "public-key.png"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(reference)
        self.run_pipeline(reference_image=reference)
        before = self.budget.summary()
        Image.new("RGB", (32, 32), (50, 60, 70)).save(reference)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_pipeline(reference_image=reference)
        self.assertEqual(self.budget.summary(), before)

    def test_changed_prompts_refuse_resume_before_inference(self):
        self.run_pipeline()
        before = self.budget.summary()
        for target in ["robologue.verifier.PROMPT", "robologue.perception.inspect.PROMPT"]:
            with self.subTest(target=target):
                with patch(target, "Changed prompt"):
                    with self.assertRaisesRegex(ValueError, "configuration changed"):
                        self.run_pipeline()
        self.assertEqual(self.budget.summary(), before)

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


class BudgetExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.sqlite"
        self.budget = CallBudget(self.path, max_calls=2)
        self.addCleanup(lambda: self.budget.close())

    def test_extension_preserves_all_charges_and_audits_dollars_across_restart(self):
        known = self.budget.reserve("known")
        self.budget.finish(known, {"cost": 0.002, "total_tokens": 11})
        unknown = self.budget.reserve("unknown")
        self.budget.mark_failed(unknown, "transport")
        previous = self.budget.store.load(self.budget.session)
        self.budget.extend_call_limit(500)
        state = self.budget.store.load(self.budget.session)
        self.assertEqual(state["attempts"], previous["attempts"])
        self.assertEqual(state["config"]["max_usd"], 10)
        self.assertEqual(state["config"]["reservation"], CALL_RESERVATION)
        self.assertEqual(state["limit_changes"], [
            {"previous": 2, "new": 500, "dollar_cap_unchanged": 10}
        ])
        self.budget.close()
        self.budget = CallBudget(self.path, max_calls=500)
        summary = self.budget.summary()
        self.assertEqual(summary["dispatched_calls"], 2)
        self.assertEqual(summary["accounted_usd"], 0.042)
        self.assertEqual(summary["uncertain_charge_calls"], 1)
        self.assertEqual(summary["failed_calls"], 1)
        self.budget.reserve("third-after-extension")
        self.assertEqual(self.budget.summary()["dispatched_calls"], 3)

    def test_extension_rejects_reset_lower_equal_invalid_and_over_500(self):
        before = self.budget.store.load(self.budget.session)
        for limit in [0, 1, 2, 501, True, 3.5, "500", None]:
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.budget.extend_call_limit(limit)
                self.assertEqual(self.budget.store.load(self.budget.session), before)
                self.assertEqual(self.budget.config, before["config"])

    def test_reopen_cannot_reset_extended_call_limit_or_change_dollar_cap(self):
        self.budget.extend_call_limit(500)
        for options in [{"max_calls": 2}, {"max_calls": 500, "max_usd": 9}]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "preserve"):
                    CallBudget(self.path, **options)
        self.assertEqual(self.budget.summary()["max_calls"], 500)
        self.assertEqual(self.budget.summary()["max_usd"], 10)

    def test_extension_preserves_smaller_dollar_cap_and_exhaustion(self):
        self.budget.close()
        path = Path(self.temp.name) / "small.sqlite"
        self.budget = CallBudget(path, max_usd=0.04, max_calls=1)
        self.budget.reserve("unknown-charge")
        self.budget.extend_call_limit(500)
        self.assertEqual(self.budget.summary()["max_usd"], 0.04)
        with self.assertRaises(BudgetExceeded):
            self.budget.reserve("cannot-spend-beyond-dollars")
        self.assertEqual(self.budget.summary()["dispatched_calls"], 1)

    def test_extension_does_not_unlock_provider_pricing_violation(self):
        attempt = self.budget.reserve("expensive")
        with self.assertRaises(BudgetExceeded):
            self.budget.finish(attempt, {"cost": 0.05})
        self.budget.extend_call_limit(500)
        with self.assertRaises(BudgetExceeded):
            self.budget.reserve("still-locked")
        self.assertEqual(self.budget.summary()["accounted_usd"], 0.05)


class PublicReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.budget = CallBudget(self.root / "budget.sqlite")
        self.addCleanup(self.budget.close)
        self.client = OfflineVision(self.budget)
        self.frame = self.root / "private-recording-name.jpg"
        self.reference = self.root / "private-CAD-path.png"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(self.frame)
        Image.new("RGB", (32, 32), (100, 120, 130)).save(self.reference)
        self.packet = {"source_kind": "rgb_vlm", "cursor_frame": 0,
                       "frames": [{"frame_id": 0, "sha256": digest_file(self.frame), "path": str(self.frame)}],
                       "observations": ["Attachment visible."], "uncertainties": []}
        self.components = {"component-0": "Public base"}

    def invoke(self, **kwargs):
        return verify(self.packet, self.components, [], self.client, self.root / "cache", **kwargs)

    def test_changed_reference_bytes_invalidate_cache_at_same_path(self):
        _, first = self.invoke(reference_image=self.reference)
        _, repeated = self.invoke(reference_image=self.reference)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(repeated["cache_hit"])
        Image.new("RGB", (32, 32), (200, 10, 130)).save(self.reference)
        _, changed = self.invoke(reference_image=self.reference)
        self.assertFalse(changed["cache_hit"])
        self.assertNotEqual(changed["request_key"], first["request_key"])
        self.assertEqual(self.budget.summary()["dispatched_calls"], 2)

    def test_identical_reference_at_new_path_reuses_cache(self):
        _, first = self.invoke(reference_image=self.reference)
        renamed = self.root / "other-reference-name.png"
        renamed.write_bytes(self.reference.read_bytes())
        _, second = self.invoke(reference_image=renamed)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["request_key"], first["request_key"])
        self.assertEqual(self.budget.summary()["dispatched_calls"], 1)

    def test_reference_follows_current_frames_and_paths_never_enter_context(self):
        self.invoke(reference_image=self.reference)
        self.assertEqual(self.client.images[0], [str(self.frame), self.reference])
        context = self.client.contexts[0]
        self.assertEqual(context["frames"], [0])
        self.assertEqual(self.client.image_labels[0][0], "CURRENT RECORDING FRAME ID 0")
        self.assertIn("NOT a recording frame", self.client.image_labels[0][-1])
        self.assertIn("not a recording frame", context["public_reference"]["kind"])
        text = json.dumps(context) + self.client.systems[0]
        for private in [str(self.root), self.frame.name, self.reference.name]:
            self.assertNotIn(private, text)

    def test_reference_presence_changes_cache_identity(self):
        _, absent = self.invoke()
        _, present = self.invoke(reference_image=self.reference)
        self.assertNotEqual(absent["request_key"], present["request_key"])
        self.assertNotIn("public_reference", self.client.contexts[0])
        self.assertIn("public_reference", self.client.contexts[1])

    def test_verifier_wire_captions_distinguish_current_frame_and_public_reference(self):
        packet = copy.deepcopy(self.packet)
        packet["cursor_frame"] = 2
        packet["frames"] = [dict(self.packet["frames"][0], frame_id=i) for i in range(3)]
        raw = response(json.dumps({"components": [{"component_id": "component-0", "verdict": "correct",
                                                  "rationale": "Current connection visible.", "evidence_frame_ids": [2]}]}))
        client = OpenRouterClient(self.budget, api_key=SECRET)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as http:
            answer, _ = verify(packet, self.components, [], client, self.root / "wire-cache", reference_image=self.reference)
        content = json.loads(http.call_args.args[0].data)["messages"][1]["content"]
        self.assertEqual([item["type"] for item in content], ["text"] + ["text", "image_url"] * 4)
        captions = [content[i]["text"] for i in [1, 3, 5, 7]]
        self.assertEqual(captions[:3], ["EARLIER RECORDING FRAME ID 0", "EARLIER RECORDING FRAME ID 1",
                                        "CURRENT RECORDING FRAME ID 2"])
        self.assertIn("PUBLIC CAD COMPONENT KEY", captions[-1])
        self.assertIn("NOT a recording frame", captions[-1])
        self.assertEqual(answer[0]["evidence_frame_ids"], [2])
        self.assertNotIn(str(self.root), json.dumps(content))
        self.assertNotIn(self.reference.name, json.dumps(content))

    def test_invalid_image_labels_never_reserve_or_dispatch(self):
        client = OpenRouterClient(self.budget, api_key=SECRET)
        for labels in [[], ["x" * 161], [7]]:
            with self.subTest(labels=labels):
                with patch("urllib.request.urlopen") as http:
                    with self.assertRaisesRegex(ValueError, "image labels"):
                        client.chat(DEFAULT_MODEL, "system", {}, [self.reference], image_labels=labels)
                http.assert_not_called()
        self.assertEqual(self.budget.summary()["dispatched_calls"], 0)

    def test_wire_payload_allows_four_images_without_exposing_paths(self):
        client = OpenRouterClient(self.budget, api_key=SECRET)
        paths = [self.frame, self.frame, self.frame, self.reference]
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response()).encode())) as http:
            client.chat(DEFAULT_MODEL, "public CAD key", {"frames": [0, 1, 2]}, paths)
        payload = json.loads(http.call_args.args[0].data)
        images = payload["messages"][1]["content"][1:]
        self.assertEqual(len(images), 4)
        self.assertTrue(all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in images))
        self.assertNotIn(str(self.root), json.dumps(payload))
        before = self.budget.summary()["dispatched_calls"]
        with patch("urllib.request.urlopen") as http:
            with self.assertRaises(ValueError):
                client.chat(DEFAULT_MODEL, "system", {}, paths + [self.reference])
        http.assert_not_called()
        self.assertEqual(self.budget.summary()["dispatched_calls"], before)


class DevelopmentStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        self.output = Path(self.temp.name) / "output"
        self.runtime = Path(self.temp.name) / "shared-runtime"
        self.root.mkdir()
        self.components = {f"component-{i}": f"Public component {i}" for i in range(11)}
        self.catalog = Path(self.temp.name) / "public-catalog.json"
        self.catalog.write_text(json.dumps({"components": self.components}))
        self.records = [f"{subject}_assy_{i}_1" for subject in ("03", "08", "09") for i in range(3)]
        for rec in self.records:
            native = self.root / "native" / rec
            native.mkdir(parents=True)
            (native / "PSR_labels_raw.csv").write_text("000000.jpg,0," + ",".join(["1"] * 10) + "\n")
            maps = self.output / "media" / "time_maps"
            maps.mkdir(parents=True, exist_ok=True)
            (maps / f"{rec}.json").write_text(json.dumps({"recording_id": rec}))
        self.store = SQLiteStore(Path(self.temp.name) / "runtime.sqlite")
        self.addCleanup(self.store.close)
        self.budget = CallBudget(Path(self.temp.name) / "budget.sqlite")
        self.addCleanup(self.budget.close)
        self.client = OfflineVision(self.budget)
        self.invocations = []
        self.inferences = []

    def runner(self, store, run_id, rec, rgb, checkpoints, components, mapping, client, work, **kwargs):
        self.invocations.append((run_id, rec, kwargs))
        self.assertEqual(Path(work), self.runtime)
        self.assertEqual(components, self.components)
        if rec.startswith("09"):
            self.assertTrue((self.output / "selection.json").exists())
        identity = run_id + ":" + rec
        cached = store.load(identity)
        if cached:
            return cached
        self.inferences.append((run_id, rec))
        attempt = client.budget.reserve(identity)
        client.budget.finish(attempt, {"cost": 0.001})
        verdicts = [
            {"schema_version": 1, "source_kind": "mock", "run_id": run_id, "recording_id": rec,
             "checkpoint_id": f"{rec}:0:{name}", "cursor_frame": 0, "component_id": name,
             "verdict": "not_completed" if kwargs.get("policy") and name == "component-0" else "correct",
             "policy_id": (kwargs.get("policy") or {}).get("policy_id", "baseline-v1"),
             "model_id": "offline", "model_calls": int(i == 0), "latency_ms": 0}
            for i, name in enumerate(components)
        ]
        result = {"complete": True, "verdicts": verdicts, "decisions": []}
        store.save(identity, result)
        return result

    def run_benchmark(self, **kwargs):
        return run_benchmark(self.root, self.output, self.store, self.client, runner=self.runner,
                             runtime_dir=self.runtime, components_file=self.catalog, **kwargs)

    def test_development_only_dispatches_only_three_development_recordings(self):
        result = self.run_benchmark(development_only=True)
        self.assertEqual(result["status"], "development_complete")
        self.assertEqual(set(result["runs"]), {"development"})
        self.assertEqual(len(self.invocations), 3)
        self.assertTrue(all(rec.startswith("03_") for _, rec, _ in self.invocations))
        self.assertEqual(self.budget.summary()["dispatched_calls"], 3)
        self.assertTrue((self.output / "protocol.json").exists())
        self.assertTrue((self.output / "development-report.json").exists())
        self.assertFalse((self.output / "selection.json").exists())
        self.assertFalse((self.output / "report.json").exists())

    def test_development_resumes_full_protocol_without_rebilling_development(self):
        dev = self.run_benchmark(development_only=True)
        protocol = (self.output / "protocol.json").read_bytes()
        self.invocations.clear()
        self.inferences.clear()
        full = self.run_benchmark()
        self.assertEqual(full["status"], "complete")
        self.assertEqual((self.output / "protocol.json").read_bytes(), protocol)
        self.assertEqual(full["runs"]["development"], dev["runs"]["development"])
        self.assertFalse(any(rec.startswith("03_") and run.endswith("development") for run, rec in self.inferences))
        self.assertTrue(any(rec.startswith("08_") for _, rec, _ in self.invocations))
        self.assertTrue(any(rec.startswith("09_") for _, rec, _ in self.invocations))
        self.assertEqual(full["selection"]["promotion"]["decision"], "accept")
        self.assertEqual(full["runs"]["heldout_selected"]["state_accuracy"], 1)
        self.assertEqual(full["scored_states"], 99)
        self.assertEqual(self.budget.summary()["dispatched_calls"], 24)
        before = self.budget.summary()
        resumed = self.run_benchmark()
        self.assertEqual(resumed["selection"], full["selection"])
        self.assertEqual(self.budget.summary(), before)

    def test_changed_public_catalog_refuses_full_resume_before_inference(self):
        self.run_benchmark(development_only=True)
        changed = dict(self.components, **{"component-0": "Changed public geometry"})
        self.catalog.write_text(json.dumps({"components": changed}))
        self.invocations.clear()
        with self.assertRaisesRegex(ValueError, "Frozen evaluation protocol changed"):
            self.run_benchmark()
        self.assertEqual(self.invocations, [])
        self.assertEqual(self.budget.summary()["dispatched_calls"], 3)


if __name__ == "__main__":
    unittest.main()
