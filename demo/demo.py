"""One command: load saved records, render the replay player, and copy media next to it."""
import argparse
import json
import shutil
from pathlib import Path

from .report import build

FILES = {
    "timing": "timing.json", "evidence": "evidence.jsonl", "verdicts_baseline": "verdicts_baseline.jsonl",
    "verdicts_candidate": "verdicts_candidate.jsonl", "events": "issue_events.jsonl",
    "snapshot": "session_snapshot.json", "cases": "cases.jsonl", "policy_baseline": "policy_baseline.json",
    "policy_candidate": "policy_candidate.json",
}


def read_jsonl(path, errors):
    if not path.exists():
        errors.append(f"{path.name}: file not found")
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name} line {number}: invalid JSON ({exc.msg})")
    return records


def read_json(path, errors):
    if not path.exists():
        errors.append(f"{path.name}: file not found")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: invalid JSON ({exc.msg})")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the decision-point replay player from saved records")
    parser.add_argument("--fixtures", type=Path, default=Path("demo/contracts"),
                        help="Folder holding the saved records and media/")
    parser.add_argument("--out", type=Path, default=Path("work/player.html"))
    parser.add_argument("--evaluation", choices=["accepted", "rejected", "inconclusive", "no_proposal"],
                        default="accepted", help="Which evaluation_<name>.json fixture to show")
    parser.add_argument("--evaluation-file", type=Path, help="A real evaluation.json; overrides --evaluation")
    for key in FILES:
        parser.add_argument("--" + key.replace("_", "-"), type=Path, help=f"Override {FILES[key]}")
    args = parser.parse_args(argv)

    errors = []
    path = lambda key: getattr(args, key) or args.fixtures / FILES[key]
    evaluation_path = args.evaluation_file or args.fixtures / f"evaluation_{args.evaluation}.json"
    page, errors, notices, is_mock = build(
        verdicts={"baseline": read_jsonl(path("verdicts_baseline"), errors),
                  "candidate": read_jsonl(path("verdicts_candidate"), errors)},
        evidence=read_jsonl(path("evidence"), errors),
        evaluation=read_json(evaluation_path, errors),
        cases=read_jsonl(path("cases"), errors),
        policies={"baseline": read_json(path("policy_baseline"), errors),
                  "candidate": read_json(path("policy_candidate"), errors)},
        timing=read_json(path("timing"), errors),
        events=read_jsonl(path("events"), errors),
        snapshot=read_json(path("snapshot"), errors),
        load_errors=errors,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    media = args.fixtures / "media"
    if media.is_dir() and media.resolve() != (args.out.parent / "media").resolve():
        shutil.copytree(media, args.out.parent / "media", dirs_exist_ok=True)
    print(f"Wrote {args.out} [{'MOCK' if is_mock else 'REAL'}] "
          f"with {len(errors)} error(s) and {len(notices)} notice(s)")
    for problem in errors:
        print(f"  error: {problem}")


if __name__ == "__main__":
    main()
