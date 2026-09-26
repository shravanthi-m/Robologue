"""Offline integration, failure recovery, spending limits and paired evaluation."""

import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from robologue.store import SQLiteStore
from robologue.storage_common import CheckpointConflict
from robologue.model_client import CallBudget, OpenRouterClient, DEFAULT_MODEL
from robologue.budget import BudgetExceeded
from robologue.pipeline import run_recording
from robologue.verifier import validate_answer
from robologue.evaluate import score, EvaluationError
from robologue.policies import decide_promotion
from robologue.dataset import extract_selected
from robologue.benchmark import run_benchmark, extended_score


def verdict(frame=0, component="component-0", label="correct"):
    return dict(
        schema_version=1,
        source_kind="mock",
        run_id="offline",
        recording_id="r",
        checkpoint_id=f"r:{frame}:{component}",
        cursor_frame=frame,
        component_id=component,
        verdict=label,
        policy_id="baseline-v1",
        model_id="mock",
        model_calls=0,
        latency_ms=0,
    )


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.sqlite"

    def test_unknown_charge_and_restart(self):
        b = CallBudget(self.path, max_usd=0.08, max_calls=2)
        b.reserve("unknown")
        b.close()
        b = CallBudget(self.path, max_usd=0.08, max_calls=2)
        self.addCleanup(b.close)
        second = b.reserve("failed")
        b.finish(second, status="failed")
        with self.assertRaises(BudgetExceeded):
            b.reserve("over")
        self.assertEqual(b.summary()["accounted_usd"], 0.08)
        self.assertEqual(b.summary()["uncertain_charge_calls"], 2)

    def test_settlement_and_call_cap(self):
        b = CallBudget(self.path, max_usd=0.04, max_calls=2)
        self.addCleanup(b.close)
        a = b.reserve("ok")
        b.finish(a, {"cost": 0.001, "total_tokens": 10})
        with self.assertRaises(BudgetExceeded):
            b.reserve("remaining-less-than-reservation")
        self.assertEqual(b.summary()["reported_cost_usd"], 0.001)

    def test_pricing_violation_locks_future_dispatch(self):
        b = CallBudget(self.path)
        self.addCleanup(b.close)
        a = b.reserve("unexpected")
        with self.assertRaises(BudgetExceeded):
            b.finish(a, {"cost": 0.05})
        self.assertEqual(b.summary()["accounted_usd"], 0.05)
        with self.assertRaises(BudgetExceeded):
            b.reserve("stop")

    def test_rejects_cap_change_and_unauthorized_caps(self):
        b = CallBudget(self.path)
        b.close()
        with self.assertRaises(ValueError):
            CallBudget(self.path, max_usd=9)
        for cap in [11, True, 0, float("nan")]:
            with self.assertRaises(ValueError):
                CallBudget(self.path, max_usd=cap)

    def test_bad_input_makes_no_call(self):
        b = CallBudget(self.path)
        self.addCleanup(b.close)
        c = OpenRouterClient(b, api_key="fake")
        with patch("urllib.request.urlopen") as http:
            for model, context, tokens in [
                ("other", {}, 2),
                (DEFAULT_MODEL, {"text": "a" * 20000}, 2),
                (DEFAULT_MODEL, {}, 2001),
            ]:
                with self.assertRaises(ValueError):
                    c.chat(model, "system", context, max_tokens=tokens)
            http.assert_not_called()
        self.assertEqual(b.summary()["dispatched_calls"], 0)

    def test_http_json_charge_and_no_key_leak(self):
        b = CallBudget(self.path)
        self.addCleanup(b.close)
        c = OpenRouterClient(b, api_key="secret-do-not-log")
        raw = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"components":[]}'}}
            ],
            "usage": {"cost": 0.002},
        }
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())
        ):
            value, _ = c.chat(DEFAULT_MODEL, "system", {})
        self.assertEqual(value, {"components": []})
        self.assertNotIn("secret-do-not-log", json.dumps(b.store.load(b.session)))
        self.assertEqual(b.summary()["accounted_usd"], 0.002)

    def test_invalid_response_retains_known_charge(self):
        b = CallBudget(self.path)
        self.addCleanup(b.close)
        raw = {
            "choices": [{"message": {"content": "invalid"}}],
            "usage": {"cost": 0.003},
        }
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())
        ):
            with self.assertRaises(ValueError):
                OpenRouterClient(b, api_key="fake").chat(DEFAULT_MODEL, "system", {})
        self.assertEqual(b.summary()["accounted_usd"], 0.003)

    def test_transport_failure_reserves_cost_without_retry(self):
        import urllib.error

        b = CallBudget(self.path)
        self.addCleanup(b.close)
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("offline")
        ) as http:
            with self.assertRaises(ValueError):
                OpenRouterClient(b, api_key="fake").chat(DEFAULT_MODEL, "system", {})
        self.assertEqual(http.call_count, 1)
        self.assertEqual(b.summary()["accounted_usd"], 0.04)
        self.assertEqual(b.summary()["failed_calls"], 1)


class FakeVision:
    def __init__(self, budget):
        self.budget = budget
        self.contexts = []

    def chat(self, model, system, context, images=(), max_tokens=2000):
        self.contexts.append(copy.deepcopy(context))
        permit = self.budget.reserve(str(len(self.contexts)))
        self.budget.finish(permit, {"cost": 0.001})
        if "components" in context:
            value = {
                "components": [
                    {
                        "component_id": c,
                        "verdict": "correct",
                        "rationale": "Current frame attachment visible.",
                        "evidence_frame_ids": [context["current_frame"]],
                    }
                    for c in context["components"]
                ]
            }
        else:
            value = {
                "observations": ["A colored assembly is visible."],
                "uncertainties": ["Attachment detail unclear."],
            }
        return value, {"latency_seconds": 0.01, "usage": {"cost": 0.001}}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = self.root / "rgb"
        self.rgb.mkdir()
        for i in range(5):
            Image.new("RGB", (32, 32), (i * 40, 20, 100)).save(self.rgb / f"{i:06}.jpg")
        self.components = {"component-0": "Public base", "component-1": "Public wheel"}
        self.mapping = {
            "schema_version": 1,
            "verified": True,
            "recording_id": "sample",
            "mapping_id": "offline-pts",
            "frame_seconds": {str(i): i * 0.1 for i in range(5)},
        }
        self.store = SQLiteStore(self.root / "runtime.sqlite")
        self.addCleanup(self.store.close)
        self.budget = CallBudget(self.root / "budget.sqlite")
        self.addCleanup(self.budget.close)
        self.client = FakeVision(self.budget)

    def run_pipeline(self, **kwargs):
        return run_recording(
            self.store,
            "test",
            "sample",
            self.rgb,
            [0, 2, 4],
            self.components,
            self.mapping,
            self.client,
            self.root / "cache",
            **kwargs,
        )

    def test_full_rgb_pipeline_cache_resume_and_bounded_memory(self):
        a = self.run_pipeline(stop_after=1)
        self.assertFalse(a["complete"])
        self.assertEqual(len(a["verdicts"]), 2)
        self.assertEqual(self.budget.summary()["dispatched_calls"], 2)
        b = self.run_pipeline()
        self.assertTrue(b["complete"])
        self.assertEqual(len(b["verdicts"]), 6)
        self.assertEqual([d["memory_items"] for d in b["decisions"]], [0, 1, 2])
        calls = self.budget.summary()["dispatched_calls"]
        self.assertEqual(self.run_pipeline()["verdicts"], b["verdicts"])
        self.assertEqual(self.budget.summary()["dispatched_calls"], calls)
        self.assertNotIn("PSR", json.dumps(self.client.contexts))
        self.assertNotIn(str(self.root), json.dumps(self.client.contexts))
        self.assertNotIn("recording_id", json.dumps(self.client.contexts))

    def test_crash_after_checkpoint_recovers_projection_without_new_inference(self):
        with patch.object(
            self.store, "put_decision", side_effect=RuntimeError("forced crash")
        ):
            with self.assertRaises(RuntimeError):
                self.run_pipeline(stop_after=1)
        self.assertEqual(self.budget.summary()["dispatched_calls"], 2)
        self.assertEqual(self.store.decisions("pipeline:test:memory:sample"), [])
        result = self.run_pipeline(stop_after=1)
        self.assertEqual(result["committed_checkpoints"], 2)
        self.assertEqual(self.budget.summary()["dispatched_calls"], 4)
        self.assertEqual(len(result["decisions"]), 2)

    def test_resume_pins_full_time_map(self):
        self.run_pipeline(stop_after=1)
        self.mapping["frame_seconds"]["4"] = 0.5
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_pipeline()

    def test_stateless_no_memory_and_reuses_neutral_images(self):
        self.run_pipeline()
        before = self.budget.summary()["dispatched_calls"]
        result = self.run_pipeline(mode="stateless")
        self.assertTrue(all(d["memory_items"] == 0 for d in result["decisions"]))
        self.assertEqual(self.budget.summary()["dispatched_calls"] - before, 2)

    def test_invalid_verifier_abstains_and_commits(self):
        def broken(*args, **kwargs):
            raise ValueError("bad JSON")

        result = self.run_pipeline(stop_after=1, verify_fn=broken)
        self.assertTrue(
            all(v["verdict"] == "insufficient_evidence" for v in result["verdicts"])
        )
        self.assertTrue(all(v["provider_failed"] for v in result["verdicts"]))

    def test_stale_writer_and_changed_immutable_evidence(self):
        self.store.compare_and_swap("s", {"revision": 0}, None)
        self.store.compare_and_swap("s", {"revision": 1}, 0)
        with self.assertRaises(CheckpointConflict):
            self.store.compare_and_swap("s", {"revision": 1}, 0)
        self.store.put_inspection("s", "e", {"text": "a"})
        with self.assertRaises(CheckpointConflict):
            self.store.put_inspection("s", "e", {"text": "b"})

    def test_future_and_unverified_maps_rejected(self):
        self.mapping["verified"] = False
        with self.assertRaises(ValueError):
            self.run_pipeline()

    def test_policy_reaches_verifier_and_candidate_requires_explicit_validation(self):
        policy = {
            "schema_version": 1,
            "policy_id": "rule-v2",
            "parent_policy_id": "baseline-v1",
            "status": "candidate",
            "checklist": ["Require current connection evidence."],
            "supporting_dev_case_ids": ["dev:0"],
        }
        with self.assertRaises(ValueError):
            self.run_pipeline(policy=policy)
        result = self.run_pipeline(policy=policy, allow_candidate=True, stop_after=1)
        contexts = [c for c in self.client.contexts if "components" in c]
        self.assertEqual(contexts[0]["verification_checklist"], policy["checklist"])
        self.assertTrue(all(v["policy_id"] == "rule-v2" for v in result["verdicts"]))
        with self.assertRaises(ValueError):
            self.run_pipeline(policy=dict(policy, status="accepted"))


class SchemaTests(unittest.TestCase):
    def answer(self, ids, verdict="correct"):
        return {
            "components": [
                {
                    "component_id": "c",
                    "verdict": verdict,
                    "rationale": "Visible.",
                    "evidence_frame_ids": ids,
                }
            ]
        }

    def test_old_evidence_cannot_sign_off_current_assembly(self):
        result = validate_answer(self.answer([0]), {"c": "Base"}, [0, 2])
        self.assertEqual(result[0]["verdict"], "insufficient_evidence")

    def test_unavailable_duplicate_and_missing_components(self):
        for answer in [
            self.answer([3]),
            {"components": []},
            {"components": [self.answer([2])["components"][0]] * 2},
        ]:
            with self.assertRaises(ValueError):
                validate_answer(answer, {"c": "Base"}, [0, 2])

    def test_duplicate_verdict_rejected(self):
        ref = [
            {
                "recording_id": "r",
                "component_id": "component-0",
                "frame": 0,
                "reference": "correct",
            }
        ]
        with self.assertRaises(EvaluationError):
            score([verdict(), verdict()], ref)

    def test_invalid_counts_bool_cursor_and_latency_rejected(self):
        for changes in [
            {"cursor_frame": True},
            {"model_calls": -1},
            {"latency_ms": float("nan")},
            {"schema_version": 2},
        ]:
            with self.assertRaises(EvaluationError):
                score([dict(verdict(), **changes)], [])

    def test_promotion_requires_identical_case_cohort(self):
        ref = [
            {
                "recording_id": "r",
                "component_id": "component-0",
                "frame": 0,
                "reference": "not_completed",
            },
            {
                "recording_id": "r",
                "component_id": "component-0",
                "frame": 1,
                "reference": "correct",
            },
        ]
        base = score([verdict(), verdict(1)], ref)
        candidate = score([verdict(1)], ref)
        self.assertEqual(decide_promotion(base, candidate, 10)["decision"], "reject")

    def test_confusion_counts_abstentions_as_misses(self):
        refs = [
            {
                "recording_id": "r",
                "component_id": "component-0",
                "frame": i,
                "reference": t,
            }
            for i, t in enumerate(["correct", "incorrect", "not_completed"])
        ]
        result = extended_score(
            [
                verdict(0),
                verdict(1, label="insufficient_evidence"),
                verdict(2, label="not_completed"),
            ],
            refs,
        )
        self.assertAlmostEqual(result["state_accuracy"], 2 / 3)
        self.assertEqual(result["incorrect_recall"], 0)
        self.assertEqual(
            result["confusion_matrix"]["incorrect"]["insufficient_evidence"], 1
        )


class ArchiveTests(unittest.TestCase):
    def test_same_size_corruption_is_reextracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "a.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("rgb.jpg", "good")
            extract_selected(archive, root / "out", lambda n: True)
            (root / "out" / "rgb.jpg").write_text("evil")
            extract_selected(archive, root / "out", lambda n: True)
            self.assertEqual((root / "out" / "rgb.jpg").read_text(), "good")

    def test_traversal_rejected_and_selected_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "a.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../evil.txt", "bad")
                z.writestr("rgb/000001.jpg", "image")
                z.writestr("truth.csv", "hidden")
            with self.assertRaises(ValueError):
                extract_selected(archive, root / "out", lambda n: True)
            extract_selected(archive, root / "clean", lambda n: n.startswith("rgb/"))
            self.assertFalse((root / "clean" / "truth.csv").exists())


class CoordinatorTests(unittest.TestCase):
    def test_split_selection_freezes_before_heldout_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out = Path(tmp) / "eval"
            components = {f"component-{i}": f"Public {i}" for i in range(11)}
            root.mkdir()
            (root / "components.json").write_text(
                json.dumps({"components": components})
            )
            records = [
                f"{subject}_assy_{i}_1"
                for subject in ("03", "08", "09")
                for i in range(3)
            ]
            for rec in records:
                native = root / "native" / rec
                native.mkdir(parents=True)
                (native / "PSR_labels_raw.csv").write_text(
                    "000000.jpg,0," + ",".join(["1"] * 10) + "\n"
                )
                maps = out / "media" / "time_maps"
                maps.mkdir(parents=True, exist_ok=True)
                (maps / f"{rec}.json").write_text(json.dumps({"recording_id": rec}))
            budget = CallBudget(out / "budget.sqlite")
            store = SQLiteStore(out / "runtime.sqlite")
            calls = []

            def runner(
                store,
                run_id,
                rec,
                rgb,
                checkpoints,
                definitions,
                mapping,
                client,
                work,
                **kwargs,
            ):
                if rec.startswith("09"):
                    self.assertTrue((out / "selection.json").exists())
                calls.append((rec, kwargs))
                values = [
                    dict(
                        verdict(
                            component=c,
                            label=(
                                "not_completed"
                                if kwargs.get("policy") and c == "component-0"
                                else "correct"
                            ),
                        ),
                        recording_id=rec,
                        run_id=run_id,
                        checkpoint_id=f"{rec}:0:{c}",
                    )
                    for c in definitions
                ]
                return {"complete": True, "verdicts": values, "decisions": []}

            try:
                report = run_benchmark(
                    root, out, store, FakeVision(budget), runner=runner
                )
                self.assertEqual(report["selection"]["promotion"]["decision"], "accept")
                self.assertEqual(
                    report["runs"]["heldout_selected"]["state_accuracy"], 1
                )
                self.assertEqual(report["scored_states"], 99)
                self.assertTrue((out / "report.html").exists())
                report2 = run_benchmark(
                    root, out, store, FakeVision(budget), runner=runner
                )
                self.assertEqual(report["selection"], report2["selection"])
            finally:
                store.close()
                budget.close()


if __name__ == "__main__":
    unittest.main()
