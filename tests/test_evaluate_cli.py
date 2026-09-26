"""Tests for the `evaluate` CLI command: the runnable end-to-end
IndustReal-format pipeline (labels -> score -> propose -> promote)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from robologue.cli import main

FIXTURES = Path(__file__).parent.parent / "examples" / "eval-synthetic"


def run_cli(*argv):
    old = sys.argv
    sys.argv = ["cli", *argv]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            main()
    finally:
        sys.argv = old
    return buffer.getvalue()


class EvaluateCommandTests(unittest.TestCase):
    def test_scores_and_proposes_without_candidate(self):
        out = run_cli("evaluate", str(FIXTURES / "PSR_labels_raw.csv"),
                      str(FIXTURES / "baseline-verdicts.jsonl"),
                      "--recording", "rec-SYN")
        self.assertIn("References: 10", out)
        self.assertIn("state_accuracy=0.800", out)
        self.assertIn("false_corrects=2", out)
        self.assertIn("require-explicit-connection-evidence-v1", out)
        self.assertNotIn("Promotion decision", out)

    def test_promotion_accepts_on_fewer_false_corrects(self):
        out = run_cli("evaluate", str(FIXTURES / "PSR_labels_raw.csv"),
                      str(FIXTURES / "baseline-verdicts.jsonl"),
                      "--recording", "rec-SYN",
                      "--candidate-verdicts", str(FIXTURES / "candidate-verdicts.jsonl"),
                      "--policy-id", "checklist-v2",
                      "--parent-policy-id", "baseline-v1",
                      "--budget", "24")
        self.assertIn("Promotion decision: accept", out)
        self.assertIn("False-corrects fell 2 -> 0", out)
        self.assertIn("frozen as accepted", out)

    def test_json_report_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            run_cli("evaluate", str(FIXTURES / "PSR_labels_raw.csv"),
                    str(FIXTURES / "baseline-verdicts.jsonl"),
                    "--recording", "rec-SYN",
                    "--candidate-verdicts", str(FIXTURES / "candidate-verdicts.jsonl"),
                    "--budget", "24",
                    "--output", str(report_path))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["recording_id"], "rec-SYN")
            self.assertEqual(report["n_references"], 10)
            self.assertEqual(report["baseline"]["false_correct_count"], 2)
            self.assertEqual(report["candidate"]["false_correct_count"], 0)
            self.assertEqual(report["promotion"]["decision"], "accept")
            self.assertEqual(report["frozen_policy"]["status"], "accepted")
            self.assertEqual(report["proposed_policy"]["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
