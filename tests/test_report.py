import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from demo import demo
from demo.report import build, frame_to_seconds

FIXTURES = Path(__file__).resolve().parent.parent / "demo" / "contracts"


def jsonl(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def js(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def inputs(evaluation="accepted"):
    return {
        "verdicts": {"baseline": jsonl("verdicts_baseline.jsonl"), "candidate": jsonl("verdicts_candidate.jsonl")},
        "evidence": jsonl("evidence.jsonl"), "evaluation": js(f"evaluation_{evaluation}.json"),
        "cases": jsonl("cases.jsonl"),
        "policies": {"baseline": js("policy_baseline.json"), "candidate": js("policy_candidate.json")},
        "timing": js("timing.json"), "events": jsonl("issue_events.jsonl"), "snapshot": js("session_snapshot.json"),
    }


def payload(page):
    start = page.index('id="replay-data">') + len('id="replay-data">')
    return json.loads(page[start:page.index("</script>", start)])


def checkpoint(page, arm, checkpoint_id):
    return next(cp for cp in payload(page)["arms"][arm]["checkpoints"] if cp["checkpoint_id"] == checkpoint_id)


class ReportTests(unittest.TestCase):
    def test_fixtures_render_without_errors(self):
        _, errors, notices, is_mock = build(**inputs())
        self.assertEqual(errors, [])
        self.assertTrue(is_mock)
        self.assertEqual(len(notices), 2)

    def test_is_mock_reported_but_not_shown_on_page(self):
        # The page itself no longer shows a MOCK/REAL badge (single real-data view),
        # but build() still reports is_mock accurately for the CLI's own status line.
        page, _, _, is_mock = build(**inputs())
        self.assertTrue(is_mock)
        self.assertNotIn("MOCK", page)
        real = inputs()
        for value in real.values():
            for record in (value.values() if isinstance(value, dict) and "schema_version" not in value else [value]):
                for item in (record if isinstance(record, list) else [record]):
                    item["source_kind"] = "rgb_vlm"
        page, errors, _, is_mock = build(**real)
        self.assertEqual(errors, [])
        self.assertFalse(is_mock)
        self.assertNotIn("MOCK", page)

    def test_only_accepted_says_improved(self):
        self.assertIn("improved", build(**inputs("accepted"))[0])
        for decision in ("rejected", "inconclusive", "no_proposal"):
            page, errors, *_ = build(**inputs(decision))
            self.assertEqual(errors, [])
            self.assertNotIn("improved", page.lower(), decision)

    def test_unavailable_metric_rendered_as_saved(self):
        page, *_ = build(**inputs("rejected"))
        self.assertIn('<span class="muted">unavailable</span>', page)

    def test_at_least_three_verdicts_render(self):
        # This fixture is grounded in an honest read of the mock video, which shows an ordinary
        # successful build with no demonstrated error, so "incorrect" isn't asserted here.
        page, *_ = build(**inputs())
        seen = {cp["verdict"] for arm in payload(page)["arms"].values() for cp in arm["checkpoints"]}
        self.assertEqual(seen, {"correct", "not_completed", "insufficient_evidence"})

    def test_non_development_and_current_recording_cases_are_refused(self):
        data = inputs()
        data["verdicts"]["candidate"][1]["recalled_case_ids"] = ["dev-03", "val-01", "cur-01"]
        page, errors, notices, _ = build(**data)
        shown = [c["case_id"] for c in checkpoint(page, "candidate", "c2")["recalled"]["cases"]]
        self.assertEqual(shown, ["dev-03"])
        self.assertTrue(any("val-01" in e for e in errors))
        self.assertTrue(any("cur-01" in e for e in errors))
        self.assertTrue(any("val-01" in n and "validation" in n for n in notices))

    def test_recall_falls_back_to_policy_supporting_cases(self):
        page, *_ = build(**inputs())
        recalled = checkpoint(page, "candidate", "c4")["recalled"]
        self.assertEqual(recalled["source"], "policy")
        self.assertEqual([c["case_id"] for c in recalled["cases"]], ["dev-03", "dev-07"])
        self.assertEqual(checkpoint(page, "candidate", "c3")["recalled"]["cases"], [])

    def test_baseline_approves_where_candidate_recalls_and_holds(self):
        page, *_ = build(**inputs())
        self.assertEqual(checkpoint(page, "baseline", "c2")["verdict"], "correct")
        held = checkpoint(page, "candidate", "c2")
        self.assertEqual(held["verdict"], "insufficient_evidence")
        self.assertEqual(held["recalled"]["cases"][0]["case_id"], "dev-03")

    def test_issue_survives_unrelated_checkpoint_then_resolves(self):
        page, *_ = build(**inputs())
        lanes = {lane["issue_id"]: lane for lane in payload(page)["arms"]["candidate"]["issues"]}
        wheel = lanes["wheel-connection-unverified"]
        self.assertIn(147.0, wheel["carried"])  # c3 checks the unrelated axle-assembly component
        self.assertTrue(wheel["resolved"])
        self.assertEqual(wheel["resolved_at"], "c6")
        # The real footage's final frame shows both connections seated, so this one honestly
        # resolves too rather than staying open just to demonstrate the case.
        self.assertTrue(lanes["second-wheel-seating-unverified"]["resolved"])

    def test_resolution_without_evidence_keeps_issue_open(self):
        data = inputs()
        data["events"][5]["evidence_ids"] = []
        page, errors, *_ = build(**data)
        lanes = {lane["issue_id"]: lane for lane in payload(page)["arms"]["candidate"]["issues"]}
        self.assertFalse(lanes["wheel-connection-unverified"]["resolved"])
        self.assertTrue(any("cites no evidence" in e for e in errors))

    def test_malformed_records_show_errors_not_crash(self):
        data = inputs()
        data["verdicts"]["baseline"][0]["schema_version"] = 2
        del data["evidence"][2]["observations"]
        data["verdicts"]["candidate"][3]["verdict"] = "probably_fine"
        data["evaluation"] = {"schema_version": 1}
        page, errors, *_ = build(**data)
        self.assertTrue(any("schema_version must be 1" in e for e in errors))
        self.assertTrue(any("missing observations" in e for e in errors))
        self.assertTrue(any("invalid verdict" in e for e in errors))
        self.assertIn("schema_version must be 1", page)
        self.assertIn("No valid evaluation record", page)

    def test_future_frames_rejected(self):
        data = inputs()
        data["evidence"][0]["frame_ids"] = [590, data["evidence"][0]["cursor_frame"] + 1]
        _, errors, *_ = build(**data)
        self.assertTrue(any("future frames" in e for e in errors))

    def test_reason_text_is_escaped(self):
        data = inputs()
        attack = "</script><script>alert(1)</script><img src=x onerror=alert(2)>"
        data["verdicts"]["candidate"][0]["reason"] = attack
        data["evaluation"]["reason"] = attack
        page, *_ = build(**data)
        self.assertNotIn("<script>alert(1)", page)
        self.assertNotIn("<img src=x", page)
        self.assertEqual(checkpoint(page, "candidate", "c1")["reason"], attack)

    def test_page_is_offline(self):
        page, *_ = build(**inputs())
        for needle in ('src="http', "src='http", 'href="http', "<link", "@import"):
            self.assertNotIn(needle, page)

    def test_missing_timing_entry_is_an_error_not_a_default(self):
        data = inputs()
        data["timing"]["recordings"].pop("assembly-demo")
        page, errors, *_ = build(**data)
        self.assertTrue(any("no default FPS" in e for e in errors))
        self.assertIsNone(checkpoint(page, "candidate", "c2")["t"])


class TimingTests(unittest.TestCase):
    def test_fps_and_offset(self):
        timing = {"recordings": {"r": {"fps": 10, "frame_offset": 20}}}
        self.assertEqual(frame_to_seconds(timing, "r", 120), 10.0)
        with self.assertRaises(ValueError):
            frame_to_seconds(timing, "r", 10)

    def test_frame_map_interpolates_and_refuses_out_of_range(self):
        timing = {"recordings": {"r": {"frame_map": {"100": 0.0, "200": 4.0, "300": 6.0}}}}
        self.assertEqual(frame_to_seconds(timing, "r", 150), 2.0)
        self.assertEqual(frame_to_seconds(timing, "r", 250), 5.0)
        for frame in (99, 301):
            with self.assertRaises(ValueError):
                frame_to_seconds(timing, "r", frame)

    def test_invalid_configs(self):
        for entry in ({}, {"fps": 10, "frame_map": {"1": 0, "2": 1}}, {"fps": 0}, {"frame_map": {"1": 0}},
                      {"frame_map": {"1": 2.0, "2": 1.0}}):
            with self.assertRaises(ValueError):
                frame_to_seconds({"recordings": {"r": entry}}, "r", 1)
        with self.assertRaises(ValueError):
            frame_to_seconds({"recordings": {}}, "r", 1)


class CliTests(unittest.TestCase):
    def test_writes_player_and_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = Path(directory) / "contracts"
            fixtures.mkdir()
            for path in FIXTURES.glob("*.json*"):
                (fixtures / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            with (fixtures / "evidence.jsonl").open("a", encoding="utf-8") as target:
                target.write("{not json\n")
            out = Path(directory) / "out" / "player.html"
            with redirect_stdout(io.StringIO()) as stdout:
                demo.main(["--fixtures", str(fixtures), "--out", str(out), "--evaluation", "rejected"])
            page = out.read_text(encoding="utf-8")
            self.assertIn("evidence.jsonl line 7: invalid JSON", page)
            self.assertIn("[MOCK]", stdout.getvalue())
            self.assertIn("Candidate rejected", page)


if __name__ == "__main__":
    unittest.main()
