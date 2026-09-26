# Person 1: recording and evidence tools

You own reliable evidence, not the verifier's final verdict or hidden-label evaluator.
The module lives in `cookmemory/perception/`, independent of the existing replay CLI.

## Implemented

- Local video manifest with source hash, duration, dimensions, and explicit unknown sensor/alignment status.
- Frame sampling based on actual decoded presentation timestamps, including variable-frame-rate footage.
- Rejects future, out-of-range, nonfinite and empty intervals; at most eight frames per inspection.
- Frame cache and VLM inspection cache, keyed by content, sampling, model and prompt.
- OpenRouter image input with neutral structured observations and uncertainties.
- No automatic API retries; no API key saved; no label files read or filename text sent to the VLM.
- Optional conversion into the existing replay event schema; it does not invent completed steps or issues.

No real model call has been verified yet. Synthetic-media tests and mocked HTTP tests
exercise the interfaces. A vision model supporting image input and JSON output is
required. We have not downloaded CaptainCook4D recordings or validated spatial files.

## Your next actions

1. Coordinate with Person 3 on one recipe and a tiny recording subset; let them keep
   the error annotations and split assignments separate from the perception inputs.
2. Download a 360p recording from the official data links. Prefer the GoPro view and
   preserve its recording ID. Confirm the camera/view and timing, rather than silently
   substituting a different stream. Start with one recording, not the entire dataset.
3. Register it and sample an interval you have actually watched. Open the output frames
   and verify alignment with the video. The manifest initially marks alignment unverified.
4. Set your OpenRouter key privately and choose an explicit vision-capable model ID.
   Run one inspection. Check its observations and uncertainty before producing more.
5. Give Person 2 the manifest path and Python tool interface below. Give Person 4 the
   output evidence paths/timestamps. Person 3 separately maps decisions to labels.

Sources: [CaptainCook4D downloader](https://github.com/CaptainCook4D/downloader),
[official download-link manifest](https://github.com/CaptainCook4D/downloader/blob/main/metadata/download_links.json),
[OpenRouter API](https://openrouter.ai/docs/api-reference/overview).
The official GoPro script enumerates the link manifest; do not run a whole-dataset
command blindly when you only need a small subset. Files downloaded by a tool must
be opened as media, never executed. Keep downloaded data under the ignored `data/`.

## Local setup

Python 3.10+ and FFmpeg (`ffmpeg` and `ffprobe` on PATH) are enough. Both tools are
available on the current machine. The module uses Python's standard library.
Run commands from the repository root.

```bash
python3 -m cookmemory.perception --help
python3 -m cookmemory.perception register /absolute/path/to/recording.mp4 --recording RECORDING_ID
python3 -m cookmemory.perception list
python3 -m cookmemory.perception sample --recording RECORDING_ID --start 0 --end 10 --observed-until 10 --frames 3
```

The default manifest is `work/recordings.json`; artifacts are under `work/perception/`.
Substitute real paths and IDs. The interval must fit the video duration. `register`
refuses an existing ID unless you pass `--replace`. If media changes, re-register;
caches are keyed by its content hash. Times are relative to the first decoded video
frame. Check any offset between this origin and dataset annotations yourself.

First indexing scans the video once and can take time; subsequent sampling reuses
the timestamp index. Extraction decodes from the start for exact frame-index selection;
it favors correctness over speed and may be slow for late intervals in long videos.
Each FFmpeg/ffprobe invocation has a 180-second timeout. Use a modest recording first.
If you make a clip, preserve and explicitly handle its offset; do not pass original
recording timestamps into a clipped file without conversion.

## One real VLM inspection

Choose a model supporting both image input and JSON output. Export these privately
in your shell (or use your local secret-management workflow):

```bash
export OPENROUTER_MODEL='provider/model-id'
# Set OPENROUTER_API_KEY privately; do not paste it into chat or commit it.
python3 -m cookmemory.perception inspect --recording RECORDING_ID --start 0 --end 10 --observed-until 10 --frames 3
```

The network request sends only the neutral prompt, frame timestamps, and JPEG bytes.
It does not send local paths, recording IDs, annotations, or expected failure labels.
This does not prevent visual content itself from containing labels: use unannotated
source video and inspect it first. Outputs are predictions, not validated truth.

Repeat the same inspection to use the cache without another paid request. Original
usage/latency remain in `call`; `cache_hit: true` indicates no new API call. The agent
should still account for the logical tool call so cached runs do not get a larger
verification budget. Model aliases can change upstream; freeze a stable model ID and
cache snapshot for experiments and retain the returned model ID in provenance.

## Person 2's direct integration

```python
from cookmemory.perception import inspect_video

# Reserve the run's tool-call budget BEFORE calling this function.
result = inspect_video(
    manifest_path="work/recordings.json",
    recording_id="RECORDING_ID",
    t_start=0,
    t_end=10,
    n_frames=3,
    observed_until=10,
    model="provider/model-id",  # or OPENROUTER_MODEL
)
# Feed observations/uncertainties to the verifier.
# Persist inspection_id, evidence, provenance and call costs in the run trace.
```

Return fields include `inspection_id`, `recording_id`, `window`, `observations`,
`uncertainties`, `evidence` (frame IDs, exact timestamps, local paths, hashes),
`model_version`, `prompt_version`, `provenance`, `call`, `cache_hit`, and an explicitly
unavailable `motion_summary` placeholder. No verdict or error category is returned.
Person 2 must enforce total calls, per-run costs and the trusted observation cursor.
Passing an arbitrary future `observed_until` from model arguments would bypass the
intended gate: the orchestrator, not the model, must supply this value.

To create an optional event for the OLD deterministic replay loop:

```bash
python3 -m cookmemory.perception inspect --recording RECORDING_ID --start 0 --end 10 --observed-until 10 --frames 3 --event-output work/observation-001.jsonl
python3 -m cookmemory.cli replay work/observation-001.jsonl --session real-observation-smoke
```

This only carries descriptive evidence into replay. It intentionally supplies no
`issue`, `resolves`, or `completed_steps`; Person 2's verifier must infer those with
evidence. The original deterministic replay cannot classify errors from this text.
The event output must be a new file. Keep distinct windows ordered when combining
multiple events; do not mix recordings into one session.

## Pose work: after the RGB path is integrated

Do not build a generic hand-joint parser before inspecting the released files.
First verify fields, units, coordinate frames, synchronization, and validity flags
for one spatial stream. Pick one simple justified feature, such as wrist displacement
in a known coordinate frame or tracking coverage. Headset motion can contaminate
camera-frame hand motion. Never label it circular whisking without an actual detector.
Do not assume headset IMU measurements are wrist measurements.

The current tool returns `motion_summary.status = unavailable`. It does not unpickle
untrusted files or fabricate joint features. Add a validated sensor adapter later,
and include its version/source fingerprint in any combined inspection cache key.

## Acceptance checks

```bash
python3 -m unittest discover -s tests -v
```

Tests cover actual timestamp/pixel selection, no future evidence, malformed windows,
cache reuse and invalidation, regenerated corrupt frames, hidden-filename exclusion,
observation-schema validation and compatibility with the existing replay schema.

Before handoff on REAL data, check three intervals manually, verify frame timestamps
against playback, review one real VLM result, and give teammates the concrete manifest
and inspection ID. A synthetic test pass does not establish dataset alignment or
model accuracy. Keep test error labels out of this process.
