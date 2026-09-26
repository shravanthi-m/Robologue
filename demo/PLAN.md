# Person 4: Decision-Point Replay Player (final plan)

Owner: b (Person 4). MongoDB "Build in a Day" hackathon, 26 Sep 2026. Build window: about 3 hours.
This file is self-contained and can be pasted whole into another model as its brief.

## 1. What we're building

A **static, self-contained HTML page** that replays a recorded IndustReal assembly video and stops at each
verification checkpoint to show what the fixed verifier **saw, remembered, recalled and decided**. Every item
on the page comes from a saved record. It also plays clips of **past development failures** that the verifier
recalled, lets viewers switch between the **baseline** and **candidate** policy, shows the **Atlas session memory**
before and after a process restart, and shows the **evaluation result** exactly as Person 3 saved it.

The page never calls a model, never reads private labels and never fabricates reasoning.

## 2. Ownership and location

All Person 4 work lives in `demo/`, except the test file and a README section.

```
demo/
  PLAN.md                     this file
  __init__.py
  report.py                   render_report(...) -> HTML string (stdlib only)
  demo.py                     CLI: python -m demo.demo --fixtures demo/contracts --out work/player.html
  contracts/
    timing.json               frame -> seconds config (per recording)
    evidence.jsonl            Interface A
    verdicts_baseline.jsonl   Interface B
    verdicts_candidate.jsonl  Interface B
    issue_events.jsonl        ordered issue history
    session_snapshot.json     Atlas session doc + restart before/after
    cases.jsonl               past-failure case records
    policy_baseline.json      Interface C
    policy_candidate.json     Interface C
    evaluation_accepted.json      Interface D variants
    evaluation_rejected.json
    evaluation_inconclusive.json
    evaluation_no_proposal.json
    media/                    placeholder MOCK clips (main.mp4, dev-03.mp4, dev-07.mp4)
tests/
  test_report.py
```

Don't edit these, which belong to teammates: P1 `cookmemory/industreal_data.py`; P2 `cookmemory/verifier.py`,
`session_runner.py`, `store.py`; P3 `cookmemory/industreal_labels.py`, `evaluation.py`, `policies.py`.

## 3. Data contracts

Every record has `schema_version: 1`. `source_kind` is one of `mock`, `rgb_vlm`, `released_predictions`.

### 3.1 Timing config (new, dynamic)

The frame→time mapping lives in `demo/contracts/timing.json` and can be overridden with `--timing <path>`.
P1 replaces its values with real ones, and no code changes are needed.

```json
{
  "schema_version": 1,
  "source_kind": "mock",
  "recordings": {
    "mock-recording":     {"video_path": "media/main.mp4", "fps": 10, "frame_offset": 0},
    "mock-dev-recording": {"video_path": "media/dev.mp4", "frame_map": {"410": 0.0, "440": 3.0, "470": 6.0}}
  }
}
```

Rules:
- Each recording uses **exactly one** of these modes:
  - `fps` (+ optional `frame_offset`, default 0), where `seconds = (frame - frame_offset) / fps`
  - `frame_map` (frame → seconds), with linear interpolation between known points. This handles dropped frames and
    image folders that were re-encoded as video.
- A frame outside the `frame_map` range is an error, never an extrapolation.
- If a recording is missing from the config, that is a visible error. **There is never a default FPS.**
- `fps` must be a finite number > 0, and `frame_map` must have at least 2 points with increasing times.

### 3.2 Evidence packet (Interface A, P1 → P2)
```json
{"schema_version": 1, "source_kind": "mock", "evidence_id": "mock-e1",
 "recording_id": "mock-recording", "checkpoint_id": "mock-c1", "cursor_frame": 100,
 "component_id": "mock-wheel", "frame_ids": [90, 95, 100],
 "observations": ["Wheel is visible; attachment is occluded."],
 "uncertainties": ["Connection cannot be verified."],
 "provider": {"model_id": "mock", "prompt_version": "evidence-v1"}}
```

### 3.3 Verdict (Interface B, P2 → P3/P4)
```json
{"schema_version": 1, "source_kind": "mock", "run_id": "mock-candidate",
 "recording_id": "mock-recording", "checkpoint_id": "mock-c1", "cursor_frame": 100,
 "component_id": "mock-wheel", "verdict": "insufficient_evidence",
 "reason": "Attachment is not visible.", "evidence_ids": ["mock-e1"],
 "open_issue_ids": ["mock-issue1"], "policy_id": "candidate-v2", "model_id": "mock",
 "recalled_case_ids": ["dev-03"]}
```
- Verdicts: `correct`, `incorrect`, `not_completed`, `insufficient_evidence`.
- `recalled_case_ids` is an **optional proposed addition** that must be announced to the team. When it's absent,
  the player falls back to `policy.supporting_dev_case_ids`, and the card labels which source was used.

### 3.4 Issue event (P2 → P4)
```json
{"schema_version": 1, "source_kind": "mock", "run_id": "mock-candidate", "seq": 1,
 "issue_id": "mock-issue1", "event": "opened", "checkpoint_id": "mock-c2", "cursor_frame": 200,
 "component_id": "mock-wheel", "description": "Wheel connection unverified.", "evidence_ids": ["mock-e2"]}
```
- `event` is one of `opened`, `carried`, `resolved`. Records are ordered by `seq`. `resolved` must cite evidence.
- An issue bar runs from `opened` to `resolved`, or to the end of the run if it never resolves.

### 3.5 Session snapshot (P2 → P4)
```json
{"schema_version": 1, "source_kind": "mock", "backend": "atlas",
 "session_key": {"run_id": "mock-candidate", "recording_id": "mock-recording", "policy_id": "candidate-v2"},
 "last_committed_checkpoint": "mock-c3",
 "open_issues": [{"issue_id": "mock-issue1", "since_checkpoint": "mock-c2", "description": "Wheel connection unverified."}],
 "restart": {
   "before": {"process": "run-1", "last_committed_checkpoint": "mock-c3", "open_issue_ids": ["mock-issue1"], "verdict_count": 3},
   "after":  {"process": "run-2", "last_committed_checkpoint": "mock-c3", "open_issue_ids": ["mock-issue1"], "verdict_count": 3}
 }}
```
`backend` is `atlas` or `local`, and the header shows it.

### 3.6 Past-failure case (new)
```json
{"schema_version": 1, "source_kind": "mock", "case_id": "dev-03", "split": "development",
 "recording_id": "mock-dev-recording", "component_id": "mock-wheel",
 "frame_start": 410, "frame_end": 470, "clip_path": "media/dev-03.mp4",
 "verdict_then": "correct", "revealed_outcome": "incorrect",
 "summary": "Approved without connection evidence; the wheel was not attached."}
```
The player refuses, in code and with a visible error, any case where `split != "development"` or
`recording_id == current recording`.

### 3.7 Policy (Interface C, P3 → P2)
`schema_version`, `policy_id`, `parent_policy_id`, `status` (`candidate` / `accepted` / `rejected`),
`checklist` (at most one added instruction), `supporting_dev_case_ids`.

### 3.8 Evaluation (Interface D, P3 → P4)
```json
{"schema_version": 1, "source_kind": "mock",
 "recording_ids": ["mock-val-recording"], "baseline_policy_id": "baseline-v1", "candidate_policy_id": "candidate-v2",
 "decision": "rejected", "reason": "Tie on false-correct count; ties are rejected.",
 "scored": 6, "excluded": 1,
 "baseline":  {"state_accuracy": 0.67, "false_correct": 1, "incorrect_recall": 0.5, "abstentions": 0, "model_calls": 6, "latency_ms": 4100},
 "candidate": {"state_accuracy": 0.67, "false_correct": 1, "incorrect_recall": "unavailable", "abstentions": 1, "model_calls": 6, "latency_ms": 4300}}
```
- `decision` is one of `accepted`, `rejected`, `inconclusive`, `no_proposal`.
- For `no_proposal` only, `candidate_policy_id` and `candidate` may be `null`.
- Metrics may be `"unavailable"`. They are rendered exactly as saved and never coerced to 0.
- Both arms share one `scored`/`excluded` denominator.

## 4. Design system

**Structure comes from Frame.io's review player. Colours come from Honeycomb.**

### 4.1 Structure (Frame.io review player)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ TOP BAR  run-id · recording · model · backend ● Atlas   [Baseline|Candidate]  [MOCK] │
├───────────────────────────────────────────────┬──────────────────────────────┤
│                                               │ RIGHT PANEL (tabs)           │
│                                               │ [Decisions] [Memory] [Eval]  │
│        MAIN VIDEO  (Recorded)                 │                              │
│        + decision card overlay                │ ▸ 00:10  c1  wheel  ◆ hold   │
│          (bottom-left, slides up on pause)    │ ▸ 00:20  c2  wheel  ◆ incorr │
│                                               │ ▸ 00:30  c3  axle   ◆ ok     │
│                                               │   (active row highlighted,   │
│                                               │    auto-scrolls with playhead)│
│                                               │──────────────────────────────│
├───────────────────────────────────────────────┤ PAST FAILURES (recalled)     │
│ TRANSPORT  ▶ ⏸  00:12 / 01:00  frame 120      │ ┌────────┐ dev-03 · wheel    │
│ SCRUBBER ━━━━━◆━━━━━◆━━━━━━◆━━━━━━━◆━━━━━━━━  │ │ clip   │ said: correct     │
│ ISSUES   ░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░  │ └────────┘ was: incorrect    │
│          ░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓→ │            [dev split]       │
└───────────────────────────────────────────────┴──────────────────────────────┘
 ERRORS strip (only when validation errors exist)
```

These Frame.io patterns map onto our concepts:

| Frame.io pattern | Our use |
|---|---|
| Top bar with asset name and version stack switcher | Run/recording/model/backend, and the **Baseline / Candidate** switcher styled as a version stack |
| Large centred player on a dark canvas | Main recorded video |
| Comment markers on the scrubber | ◆ checkpoint markers coloured by verdict; clicking one seeks |
| Range comments (bar under the scrubber) | **Issue bars**, one lane per issue, from opened to resolved (or to the end with an arrow if still open) |
| Right-side comment list synced to the playhead, each with a timecode chip | **Decisions list**: timecode chip, checkpoint, component, verdict pill. The active row highlights and clicking seeks. |
| Comment detail over the frame | **Decision card** overlay |
| Right-panel tabs | `Decisions` · `Memory` (Atlas snapshot + restart before/after) · `Eval` (footer data, expanded) |
| Frame-accurate timecode readout | `mm:ss` plus `frame N` taken from the timing config |

The decision card rows show saved fields only. If a field is empty, the row shows "—".

| Row | Source |
|---|---|
| Seeing | `evidence.observations`, `evidence.uncertainties`, `frame_ids` |
| Remembering | open issues at this checkpoint (from issue events), e.g. "Issue #1 open since c2" |
| Recalling | `recalled_case_ids` or the fallback `supporting_dev_case_ids`, with the source labelled |
| Rule applied | `policy.checklist` + `policy_id` + `parent_policy_id` |
| Decision | verdict pill + `reason` |

A thin connector line runs from the card's **Recalling** row to the Past Failures panel whenever cases are shown.

### 4.2 Colour and type (after tabella-phenomenon.netlify.app)

Design tokens copied from that site's stylesheet. Its logo and wording are not used. All tokens live in one `:root`
block in `demo/report.py`.

| Token | Value | Use |
|---|---|---|
| `--bg` | `#FFFFFF` | page |
| `--surface` / `--surface-grey` / `--surface-dp` | `#ECEBE8` / `#E4E4E0` / `#F8F8F8` | stage and side cards / dividers, rails / inactive pills, notices |
| `--dark` / `--dark-2` / `--dark-3` | `#2D2D2D` / `#434343` / `#2C2C2C` | primary buttons, selected pills / evaluation bar, footer / video, monologue panel |
| `--text` / `--text-muted` | `#29303D` / `rgba(0,0,0,.4)` | ink, labels |
| `--accent` / `--accent-dark` / `--accent-green` | `#E6FB2D` / `#D9EE1C` / `#E3EF7A` | lime: button hover, played range, active chip, spinner / active press / issue spans and hover borders |
| `--error` | `#C05A5A` | errors and the incorrect verdict |
| verdicts | green `#4E8F5A`, red `#C05A5A`, grey `#7D7D78`, amber `#B7791F` | added to fit the palette; each with a light tint |
| `--mock` | `#6B4BC4` on `#ECE6FA` | MOCK badge; deliberately unlike anything else |

Type: **Red Hat Display** (SIL Open Font License; the file and licence are in `demo/assets/`), embedded in the
page as base64 so it works offline. Weights 400/500/600, sizes 12/14/16/24 px (`--fz-xs/sm/md/lg`), body line-height
150%, headings 120%. IDs, timecodes and the monologue use `ui-monospace`.

Shapes: flat with no shadows (except the card over the video). Buttons and pills use a 30 px radius; primary is
charcoal and turns lime on hover; secondary is a 1 px charcoal outline. Cards use a 24 px radius, inner rows 16–20 px,
and the footer is charcoal with rounded top corners. Tabs and the version switcher are separate pills, with the
active one charcoal. The Decisions list is white rows on the grey card, and the active row gets a charcoal outline.
Issue spans are lime pills with a charcoal start dot, a green end dot when resolved, or an arrow when still open.

Verdict colour is never the only signal. Pills also carry a text label (`CORRECT`, `INCORRECT`, `NOT DONE`, `INSUFFICIENT`).

## 5. Behaviour

1. **Playback.** The video plays, and the scrubber shows ◆ markers and issue lanes.
2. **Decision point.** When `currentTime` crosses a checkpoint time, the video pauses, the card slides up,
   the matching Decisions row highlights, and Past Failures loads 1–3 muted looping clips, or shows
   "No similar past failures." A guard stops seeking from re-triggering the same checkpoint.
3. **Controls.** Continue (Space), auto-resume after 4 s (toggle in the top bar), click ◆ or a Decisions row to jump,
   ←/→ for the previous/next checkpoint. Deep links for the presenter: `player.html#arm=baseline&cp=mock-c2&tab=memory`.
   If the video file is missing, a labelled MOCK canvas clock stands in; missing clips show a striped "MOCK · no clip" tile.
4. **Baseline/Candidate.** The version switcher swaps the verdict set and keeps the playhead where it is. The card
   re-renders so viewers can see baseline approve and candidate recall dev-03 and hold at the same checkpoint.
5. **Verifier reasoning trace** (Decisions tab, under the list). At each decision point a Claude-Code-style spinner
   cycles through verbs (*Squinting…*, *Querying Atlas…*, *Reminiscing…*, *Cross-checking…*, *Weighing it all up…*),
   then first-person lines type out: what was observed, which unresolved issues came back from Atlas memory, which
   past failures were recalled (each explicitly named as a prior miss the verifier weighs against repeating), which
   checklist rule applied, an explicit "weighing X, Y, Z against each other" line naming which of those factors are
   actually in play at this checkpoint, and finally the verdict plus any issue opened or resolved. **Every line is a
   template filled from saved fields — the chaining logic is fixed code, not a model call.** The panel is labelled
   "chained from saved evidence, memory and recalled cases · templated text, not model-generated", so it reads as a
   real train of thought without ever being mistaken for actual model chain-of-thought. Issues opened at this
   checkpoint are never described as retrieved. Auto-resume waits until the trace finishes; Continue finishes it
   instantly; `prefers-reduced-motion` shows it without animation.
6. **Memory tab.** Shows the Atlas snapshot plus Restart before/after side by side: same open issue, same
   last committed checkpoint, same verdict count (no duplicates).
7. **Eval tab / footer.** Baseline vs candidate totals over one denominator, with the decision and reason
   shown verbatim. The word "improved" appears only when `decision == "accepted"`.
8. **Errors strip.** Lists every validation error (file, line, reason). It's visible whenever there are any.

## 6. Honesty rules (non-negotiable)

- The card and the monologue show saved fields only. The monologue's wording is templated narration of those fields,
  never generated or embellished reasoning, and it is labelled as such.
- Every video is labelled **Recorded**. The **MOCK** badge is always visible if any record is `mock`.
- Recalled cases are development split only, never validation or test, and never from the current recording. This is enforced in code.
- Rejected, inconclusive, no-proposal, invalid and incomplete runs render honestly. Success is never hardcoded.
- The page is read-only: no model calls, no label files, no secrets. It works offline with no CDN or web fonts.
- No Streamlit. Output is one HTML file plus the `media/` folder next to it.
- All text goes in via `html.escape` (Python) or `textContent` (JS). The embedded JSON payload is escaped so
  `</script>` can't break out.

## 7. Task list

### Phase 0: Setup (5 min)
- [ ] Branch `person4-report` from `main`
- [ ] Create `demo/__init__.py`, `demo/contracts/`, `demo/contracts/media/`
- [ ] Make sure `work/` is gitignored

### Phase 1: Fixtures (≈20 min)
- [ ] `timing.json`: one `fps` recording, one `frame_map` recording
- [ ] `evidence.jsonl`: 5–6 packets, including one with empty observations and heavy uncertainties
- [ ] `verdicts_baseline.jsonl` / `verdicts_candidate.jsonl` covering the same checkpoints
  - [ ] All four verdicts appear across the two files
  - [ ] At one checkpoint, baseline says `correct` and candidate says `insufficient_evidence` with `recalled_case_ids: ["dev-03"]`
  - [ ] Some candidate rows leave out `recalled_case_ids` (fallback path)
- [ ] `issue_events.jsonl`: issue1 opens at c2, is carried through c3 (an unrelated component) and resolves at c5 with evidence; issue2 stays open until the end
- [ ] `session_snapshot.json` with restart before/after
- [ ] `cases.jsonl`: dev-03 and dev-07 (development), plus one `validation` case and one case from the current recording (both must be refused)
- [ ] `policy_baseline.json`, `policy_candidate.json`
- [ ] Four evaluation variants; at least one has recall `"unavailable"`
- [ ] Placeholder media: solid-colour clips with "MOCK" burned in (ffmpeg), or still frames if ffmpeg isn't available
- [ ] **Announce to the team:** frozen field names, the proposed `recalled_case_ids`, and the `timing.json` format for P1

### Phase 2: `report.py` loading and validation (≈25 min)
- [ ] JSON / JSONL loaders that keep file and line numbers
- [ ] Validators per record type: `schema_version == 1`, required fields, enums (verdict, status, decision, split, source_kind, event)
- [ ] Bad records go into an `errors` list and are rendered, not raised. Unreadable files raise.
- [ ] `frame_to_seconds(timing, recording_id, frame)` for fps and map modes, raising an error for an unknown recording or an out-of-range frame
- [ ] Recall filter (dev split only, not the current recording) and recall-source resolution
- [ ] Mock detection over all loaded records
- [ ] Build one payload per arm: checkpoints ordered by `cursor_frame`, each joined to its evidence, open issues, recalled cases and policy

### Phase 3: HTML player (≈45 min)
- [ ] `render_report(verdicts, evidence, evaluation, cases, policy, timing, events, snapshot) -> str`
- [ ] `:root` tokens from §4.2, with inline CSS and JS
- [ ] Top bar: IDs, backend dot, Baseline/Candidate version switcher, auto-resume toggle, MOCK/REAL badge
- [ ] Player with a "Recorded" label, transport, and timecode + frame readout
- [ ] Scrubber with ◆ markers, playhead, and issue lanes
- [ ] Auto-pause with a re-trigger guard, the decision card overlay, and the connector line
- [ ] Right panel tabs: Decisions (synced list), Memory (snapshot + restart), Eval
- [ ] Past Failures panel with its empty state
- [ ] Keyboard: Space, ←, →
- [ ] Errors strip
- [ ] Offline check: no external URLs in the output

### Phase 4: `demo.py` CLI (≈15 min)
- [ ] `python -m demo.demo --fixtures demo/contracts --out work/player.html [--evaluation accepted|rejected|inconclusive|no_proposal] [--timing PATH]`
- [ ] Overrides for real files: `--verdicts-baseline`, `--verdicts-candidate`, `--evidence`, `--events`, `--snapshot`, `--cases`, `--evaluation-file`
- [ ] Copy `media/` next to the output
- [ ] Print the output path, error count and MOCK/REAL status

### Phase 5: `tests/test_report.py` (≈20 min)
- [ ] MOCK badge present for mock data and absent for all-real data
- [ ] rejected / inconclusive / no_proposal pages never contain "improved"
- [ ] All four verdict classes render
- [ ] Validation-split and current-recording cases are refused
- [ ] A bad `schema_version` or missing field gives a visible error, not a crash
- [ ] Timing: fps maths, map interpolation, missing recording → error, out-of-range → error, no default fps
- [ ] `<script>` in `reason` renders as text
- [ ] Recall fallback to `supporting_dev_case_ids`
- [ ] `"unavailable"` metric rendered as-is
- [ ] `python -m unittest discover -s tests -v` passes, including `test_core.py`

### Phase 6: Manual check (≈10 min)
- [ ] Open `work/player.html` with Wi-Fi off
- [ ] Walk the story: hold → issue persists across the unrelated checkpoint → restart → dev-03 recalled → candidate's verdict differs from baseline's
- [ ] Check at projector resolution and at narrow width

### Phase 7: Real-data integration (as handoffs land)
- [ ] P1: real `timing.json`, the main video or frames, dev clip frame ranges
- [ ] P2: `verdicts.jsonl`, issue events, the Atlas snapshot, restart output; ideally `recalled_case_ids` from Vector Search
- [ ] P3: `evaluation.json`, the policy diff, dev failure cases with revealed outcomes
- [ ] Fix contract mismatches **with the owner**, not by patching around them. Re-run the tests.

### Phase 8: README and submission
- [ ] README: one-command demo, the `timing.json` format, what MOCK means
- [ ] Attribution: IndustReal (Schoonbeek et al., https://github.com/TimSchoonbeek/IndustReal), CaptainCook4D (fallback), models and libraries used
- [ ] Ask b before pushing or opening a PR

## 8. What we need from teammates

| From | What |
|---|---|
| P1 | `timing.json` values (fps or frame map per recording), main video/frames, dev clip frame ranges |
| P2 | `verdicts.jsonl` (with `recalled_case_ids` if adopted), issue events, Atlas session snapshot, restart before/after, optionally Vector Search retrieval |
| P3 | `evaluation.json`, policy files, dev failure cases with revealed outcomes |

## 9. Acceptance

- [ ] Every on-screen item maps to a saved record
- [ ] Mocks are visibly marked
- [ ] Backend, model and policy are identified in the top bar
- [ ] Recalled cases are dev-only, which is enforced and tested
- [ ] Rejected, inconclusive, no-proposal and invalid runs render honestly
- [ ] Opens offline
- [ ] A viewer can follow one story: **hold → issue persists → restart → past failure recalled → verdict changes**
