"""Convert official error_annotations.json into EVALUATOR-ONLY records."""
import math


def evaluation_records(document):
    recordings = document.values() if isinstance(document, dict) else document
    if not isinstance(document, (dict, list)):
        raise ValueError("Expected a recording list or mapping")
    for recording in recordings:
        rid = str(recording["recording_id"])
        for index, step in enumerate(recording["step_annotations"]):
            start, end = float(step["start_time"]), float(step["end_time"])
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("Nonfinite annotation timestamp")
            missing = start == -1 and end == -1
            if not missing and (start < 0 or end < start):
                raise ValueError(f"Invalid interval in {rid}, step index {index}")
            yield {"annotation_id": f"{rid}:{index}", "recording_id": rid,
                   "step_id": step["step_id"], "start_time": None if missing else start,
                   "end_time": None if missing else end, "unlocalized": missing,
                   "errors": step.get("errors", []),
                   "expected_instruction": step.get("description", ""),
                   "evaluation_only": True}
