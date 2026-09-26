"""Person 3: IndustReal reference-label adapter. Evaluator-only.

Reads the official PSR label files and emits normalized reference records.
Only this workstream reads reference labels; they must never enter agent
tools, evidence packets, or verifier context.

File formats (from the official IndustReal repository, PSR/psr_utils.py):
  - PSR_labels_raw.csv: NO header. Each row is
      frame_name, state_0, state_1, ..., state_n
    where frame_name looks like "000123.jpg" and each state is per-component:
      -1 = incorrect, 0 = not completed, 1 = correct.
  - procedure_info.json: list of action entries with id, description, install,
    state_idx, expected_in_assy / expected_in_main.

Reference records carry evaluation_only=True and are keyed by
(recording_id, component_id, frame).
"""

import csv
import json

REFERENCE_OF = {-1: "incorrect", 0: "not_completed", 1: "correct"}


class LabelError(ValueError):
    """Raised when a label file does not match the documented schema."""


def load_raw_states(path):
    """Mirror the official load_raw_psr_csv, with schema validation.

    Returns [(frame_number, frame_name, [states...])]. Frame numbers come from
    the file stem (official code strips the 4-char extension); rows must agree
    on component count and states must be in {-1, 0, 1}.
    """
    rows = []
    with open(path, newline="") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        for lineno, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            frame_name = row[0].strip()
            stem = frame_name.rsplit(".", 1)[0]
            try:
                frame = int(stem)
            except ValueError:
                raise LabelError(f"{path}:{lineno}: cannot parse frame number from {frame_name!r}")
            try:
                states = [int(cell) for cell in row[1:]]
            except ValueError:
                raise LabelError(f"{path}:{lineno}: non-integer component state in {row[1:]!r}")
            for state in states:
                if state not in REFERENCE_OF:
                    raise LabelError(f"{path}:{lineno}: state {state} not in {{-1, 0, 1}}")
            rows.append((frame, frame_name, states))
            if frame < 0 or not states:
                raise LabelError(f"{path}:{lineno}: require nonnegative frame and component states")
    widths = {len(states) for _, _, states in rows}
    if len(widths) > 1:
        raise LabelError(f"{path}: inconsistent component counts across rows: {sorted(widths)}")
    if not rows:
        raise LabelError(f"{path}: no data rows found")
    if len({frame for frame, _, _ in rows}) != len(rows):
        raise LabelError(f"{path}: duplicate checkpoint frames")
    return rows


def load_procedure_info(path):
    """Map component index -> human description.

    Component k owns actions k*3+{0,1,2} (install/incorrect/remove); the
    install action's description names the component. Falls back to any entry
    sharing the state_idx.
    """
    with open(path) as fp:
        entries = json.load(fp)
    descriptions = {}
    for entry in entries:
        idx = entry.get("state_idx")
        if idx is None or idx in descriptions:
            continue
        descriptions[idx] = entry.get("description", "")
    for entry in entries:  # prefer the install action's wording
        idx = entry.get("state_idx")
        # Incorrect-install entries also have install=True in the official
        # file. Prefer the positive install description, not error wording.
        if (idx is not None and entry.get("install")
                and not entry.get("description", "").lower().startswith("incorrect")):
            descriptions[idx] = entry.get("description", descriptions[idx])
    return descriptions


def reference_records(recording_id, raw_csv, procedure_info=None, components=None):
    """Yield normalized reference records for one recording.

    components: optional iterable of component indices to keep (scope cap:
    at most two component types for the MVP). Frame identity is the parsed
    frame number; the same number from another recording never matches because
    recording_id is part of every downstream join key.
    """
    keep = set(components) if components is not None else None
    descriptions = load_procedure_info(procedure_info) if procedure_info else {}
    for frame, frame_name, states in load_raw_states(raw_csv):
        for index, state in enumerate(states):
            if keep is not None and index not in keep:
                continue
            yield {
                "recording_id": recording_id,
                "component_id": f"component-{index}",
                "component_index": index,
                "frame": frame,
                "frame_name": frame_name,
                "raw_state": state,
                "reference": REFERENCE_OF[state],
                "description": descriptions.get(index, ""),
                "evaluation_only": True,
            }
