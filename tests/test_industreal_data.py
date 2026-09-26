"""Offline synthetic JPEGs and mocked HTTP only; no reference labels or paid calls."""
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robologue import industreal_data as data
from robologue.perception.inspect import openrouter_inspect


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class IndustRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.media.name) / "fixture"
        data.create_fixture(cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = self.root / "hidden-label-wrong-wheel" / "RGB"
        shutil.copytree(self.fixture / "RGB", self.rgb)
        self.manifest = self.root / "recordings.json"
        self.cache = self.root / "cache"
        data.register_frames(self.manifest, "recording-secret", self.rgb)

    def sample(self, **changes):
        opts = dict(manifest_path=self.manifest, recording_id="recording-secret",
                    start_frame=10, end_frame=40, observed_until_frame=40, cache_dir=self.cache)
        return data.sample_frames(**(opts | changes))

    def inspect(self, **changes):
        opts = dict(manifest_path=self.manifest, recording_id="recording-secret",
                    checkpoint_id="cp1", component_id="wheel", start_frame=10,
                    end_frame=40, observed_until_frame=40, cache_dir=self.cache, model="test-model")
        return data.get_evidence(**(opts | changes))

    def test_numeric_order_gaps_and_latest_single_frame(self):
        self.assertEqual([f["frame_id"] for f in self.sample()["frames"]], [10, 20, 40])
        self.assertEqual(self.sample(n_frames=1)["frames"][0]["frame_id"], 40)
        self.assertNotIn("timestamp", self.sample()["frames"][0])
        for f in self.sample()["frames"]:
            self.assertEqual(Path(f["path"]).read_bytes(), Path(f["source_path"]).read_bytes())

    def test_future_empty_and_malformed_ranges(self):
        for changes in [dict(observed_until_frame=20), dict(start_frame=21, end_frame=39),
                        dict(start_frame=0), dict(end_frame=41), dict(observed_until_frame=41),
                        dict(n_frames=4), dict(n_frames=True), dict(start_frame=10.0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.sample(**changes)

    def test_missing_and_changed_sources(self):
        (self.rgb / "000010.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            self.sample()
        (self.rgb / "000010.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.sample()

    def test_corrupt_image_never_reaches_provider(self):
        (self.rgb / "000010.jpg").write_bytes(b"not an image")
        data.register_frames(self.manifest, "recording-secret", self.rgb, replace=True)
        with patch.object(data.vision, "openrouter_inspect") as provider:
            with self.assertRaises(ValueError):
                self.inspect()
            provider.assert_not_called()

    def test_registration_reads_no_annotation_files_and_rejects_ambiguous_ids(self):
        (self.rgb / "PSR_labels_raw.csv").write_text("SECRET_LABEL_DO_NOT_READ")
        record = data.register_frames(self.root / "another.json", "clean", self.rgb)
        self.assertEqual(record["frame_count"], 3)
        shutil.copyfile(self.rgb / "000010.jpg", self.rgb / "10.jpeg")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            data.register_frames(self.root / "third.json", "bad", self.rgb)

    def test_rejects_symlinks_and_unmapped_names(self):
        (self.rgb / "50.jpg").symlink_to(self.rgb / "000010.jpg")
        with self.assertRaises(ValueError):
            data.register_frames(self.root / "other.json", "bad", self.rgb)
        (self.rgb / "50.jpg").unlink()
        shutil.copyfile(self.rgb / "000010.jpg", self.rgb / "error-wheel.jpg")
        with self.assertRaises(ValueError):
            data.register_frames(self.root / "other.json", "bad", self.rgb)

    def test_real_cache_provenance_invalidation_and_cursor_gate(self):
        response = ({"observations": ["A wheel is visible."], "uncertainties": ["Connection occluded."]},
                    {"provider": "mock-http", "usage": {"total_tokens": 8}})
        with patch.object(data.vision, "openrouter_inspect", return_value=response) as provider:
            a = self.inspect()
            b = self.inspect(checkpoint_id="cp2")
            self.assertFalse(a["cache_hit"])
            self.assertTrue(b["cache_hit"])
            self.assertNotEqual(a["evidence_id"], b["evidence_id"])
            self.assertEqual(b["new_api_calls"], 0)
            self.assertEqual(b["logical_tool_calls"], 1)
            self.assertEqual(provider.call_count, 1)
            self.inspect(model="changed-model")
            self.assertEqual(provider.call_count, 2)
            with self.assertRaises(ValueError):
                self.inspect(observed_until_frame=20)
            self.assertEqual(provider.call_count, 2)
            supplied = provider.call_args.args[0]
            self.assertEqual(set(supplied[0]), {"path", "frame_id"})

    def test_mock_cannot_be_confused_with_real_evidence(self):
        with patch.object(data.vision, "openrouter_inspect") as provider:
            packet = self.inspect(mock=True)
            self.assertEqual(packet["source_kind"], "mock")
            self.assertEqual(packet["new_api_calls"], 0)
            provider.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            data.get_evidence(self.fixture / "recordings.json", "synthetic-demo", "cp", "wheel",
                              10, 40, 40, model="real", cache_dir=self.cache)

    def test_http_payload_uses_ids_not_unverified_seconds_or_names(self):
        response = {"model": "vision", "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"observations": ["Colored frame."], "uncertainties": []})}}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
            openrouter_inspect([{"frame_id": 10, "path": str(self.rgb / "000010.jpg")}],
                               "vision", api_key="fake", request_timeout=30)
        payload = call.call_args.args[0].data.decode()
        self.assertIn("Frame ID 10", payload)
        self.assertNotIn("seconds", payload)
        self.assertNotIn("wrong-wheel", payload)
        self.assertNotIn("000010.jpg", payload)
        self.assertEqual(call.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
