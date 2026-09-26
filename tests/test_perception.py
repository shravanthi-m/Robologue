"""Synthetic-media and mocked-provider tests; no credentials or paid calls."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robologue.harness import validate
from robologue.perception import register_video, sample_video, inspect_video, observation_event
from robologue.perception.inspect import openrouter_inspect, validate_observations


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
class PerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory()
        cls.video = Path(cls.media.name) / "hidden-error-label-in-filename.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", "color=c=red:s=160x120:r=5:d=1",
            "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=5:d=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.video)
        ], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "recordings.json"
        self.cache = self.root / "cache"
        register_video(self.manifest, "synthetic", self.video, source="synthetic-test")

    def sample(self, **kwargs):
        options = {"manifest_path": self.manifest, "recording_id": "synthetic",
                   "t_start": 0, "t_end": .9, "n_frames": 3, "observed_until": .9,
                   "cache_dir": self.cache}
        return sample_video(**(options | kwargs))

    def test_sample_actual_timestamps_and_pixels(self):
        packet = self.sample()
        self.assertEqual([f["timestamp"] for f in packet["frames"]], [0.0, 0.4, 0.8])
        for frame in packet["frames"]:
            pixel = subprocess.run([
                "ffmpeg", "-v", "error", "-i", frame["path"], "-vf", "scale=1:1",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"
            ], check=True, capture_output=True, timeout=30).stdout
            self.assertGreater(pixel[0], 200)  # red, not future blue frames
            self.assertLess(pixel[2], 30)

    def test_future_request_rejected_even_if_cache_exists(self):
        self.sample()
        with self.assertRaises(ValueError):
            self.sample(observed_until=.5)

    def test_invalid_windows(self):
        for change in ({"t_start": -1}, {"t_end": float("nan")}, {"n_frames": 0},
                       {"n_frames": 9}, {"t_start": 1, "t_end": .9}, {"observed_until": 10}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.sample(**change)

    def test_empty_interval_and_duplicate_sampling(self):
        with self.assertRaises(ValueError):
            self.sample(t_start=.01, t_end=.1, observed_until=.1)
        result = self.sample(t_start=0, t_end=.1, n_frames=8)
        self.assertEqual(len(result["frames"]), 1)
        self.assertTrue(any("Fewer" in s for s in result["uncertainties"]))

    def test_cache_and_prompt_model_invalidation(self):
        def call(model="test-vision"):
            return inspect_video(self.manifest, "synthetic", 0, .9, 3, .9, model=model, cache_dir=self.cache)
        response = ({"observations": ["A red image is visible."], "uncertainties": ["No physical task is depicted."]},
                    {"usage": {"total_tokens": 10}, "latency_seconds": .1})
        with patch("robologue.perception.inspect.openrouter_inspect", return_value=response) as provider:
            first = call()
            second = call()
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(provider.call_count, 1)
            self.assertEqual(first["inspection_id"], second["inspection_id"])
            call("different-model")
            with patch("robologue.perception.inspect.PROMPT", "changed prompt"):
                call()
            self.assertEqual(provider.call_count, 3)
            validate(observation_event(first))
            self.assertNotIn("issue", observation_event(first))

    def test_corrupted_frame_regenerated(self):
        packet = self.sample()
        frame = Path(packet["frames"][0]["path"])
        frame.write_bytes(b"broken")
        second = self.sample()
        self.assertGreater(frame.stat().st_size, 100)
        self.assertEqual(packet["frames"][0]["sha256"], second["frames"][0]["sha256"])

    def test_registration_no_accidental_overwrite(self):
        with self.assertRaises(ValueError):
            register_video(self.manifest, "synthetic", self.video)

    def test_api_payload_hides_filenames_and_validates_response(self):
        frames = self.sample()["frames"]
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps({"model": "test", "usage": {}, "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps({
                        "observations": ["A red image."], "uncertainties": ["No task is visible."]})}}
                ]}).encode()
        with patch("urllib.request.urlopen", return_value=Response()) as network:
            result, metadata = openrouter_inspect(frames, "test", api_key="fake-test-key")
            request = network.call_args.args[0]
            body = json.loads(request.data)
            self.assertNotIn("hidden-error-label-in-filename", request.data.decode())
            self.assertNotIn(str(self.root), request.data.decode())
            self.assertEqual(body["model"], "test")
            self.assertEqual(len([p for p in body["messages"][1]["content"] if p["type"] == "image_url"]), 3)
            self.assertEqual(result["observations"], ["A red image."])

    def test_malformed_vlm_output_rejected(self):
        for value in ({"verdict": "incorrect"}, {"observations": [], "uncertainties": []},
                      {"observations": "text", "uncertainties": []}):
            with self.assertRaises(ValueError):
                validate_observations(value)


if __name__ == "__main__":
    unittest.main()
