# Person 1: IndustReal evidence adapter

Implemented independently of dataset downloads, the verifier, labels and Atlas.
The native JPEG path is `cookmemory/industreal_data.py`. Existing video perception
remains available for CaptainCook4D or downloaded RGB videos.

## Run now, offline

From the repository root, with Python 3.10+ and FFmpeg on PATH:

```bash
python3 -m cookmemory.industreal_data fixture work/industreal-demo
```

This creates three colored JPEGs, a registered manifest and `evidence.json`.
It does not download data, require credentials or call a model. Its output is
explicitly `source_kind: mock`. The output directory must be new. These are
integration fixtures, never assembly evidence or dataset results.

```bash
python3 -m unittest discover -s tests -v
python3 -m cookmemory.industreal_data --help
```

## When some data finishes downloading

Register the explicit `RGB` or `rgb` image directory, not the parent recording:

```bash
python3 -m cookmemory.industreal_data register /absolute/path/to/recording/RGB --recording RECORDING_ID
```

Registration reports the actual first/last numeric source IDs. Only regular JPEG
files with numeric stems are supported; leading zeroes are preserved in source
filenames. CSVs, annotation files and nested directories are not read. If the
release uses another naming scheme, inspect it and add an explicit mapping; do
not rename files silently or pretend indices are timestamps.

Use real IDs from that output in the following example:

```bash
python3 -m cookmemory.industreal_data sample --recording RECORDING_ID --start-frame 10 --end-frame 40 --observed-until-frame 40 --output work/sample.json
```

The adapter chooses up to three existing numeric IDs, decodes them with FFmpeg,
and preserves source filename, ID and content hash. Gaps are not fabricated.
One requested frame selects the latest available frame. It never infers seconds
from nominal FPS. Selected missing, changed, empty or unreadable files fail.

Registration is a snapshot, including partial downloads. Newly downloaded files
appear only after explicitly repeating registration with `--replace`. Do not
register files still being written. Existing output JSON files are not overwritten.

## Real neutral visual inspection

Privately configure `OPENROUTER_API_KEY` and an explicit image-input/JSON-capable
`OPENROUTER_MODEL`. These are not yet verified against actual IndustReal images.

```bash
python3 -m cookmemory.industreal_data inspect --recording RECORDING_ID --checkpoint cp-01 --component REVIEWED_COMPONENT_ID --start-frame 10 --end-frame 40 --observed-until-frame 40 --output work/evidence-01.json
```

This sends JPEG content, numeric frame IDs and the existing neutral observation
prompt. No paths, recording names, component instructions, annotations or expected
states go into the model request. There is one request, a 30-second timeout and
no automatic retry. Output is observations and uncertainty, never a correctness
verdict. `--mock` instead emits a conspicuously marked placeholder without a model.

Cache identity includes selected image hashes/IDs, model, prompt and adapter.
Cache hits retain original call usage in provenance but report `new_api_calls: 0`.
Every request still reports `logical_tool_calls: 1`; Person 2 enforces run budgets.
Source images and cursor limits are checked even on cache hits. A packet's ID also
includes recording, component, checkpoint and cursor; observation cache reuse does
not mix those identities. Corrupt observation cache entries fail; delete the named
entry before intentionally trying another request.

## Person 2 integration

```python
from cookmemory.industreal_data import get_evidence

packet = get_evidence(
    manifest_path="work/industreal/recordings.json",
    recording_id="RECORDING_ID",
    checkpoint_id="cp-01",
    component_id="REVIEWED_COMPONENT_ID",
    start_frame=10,
    end_frame=40,
    observed_until_frame=40,  # trusted runtime cursor, NOT a model argument
    model="configured/model-id",
)
```

The packet matches Interface A in the requirements: schema/source kind, recording,
checkpoint/component, cursor/frame IDs, observations, uncertainties and provider.
It additionally includes `frames` for Person 4, cache/call accounting and provenance.
The verifier should receive observations and numeric evidence IDs rather than a
serialization of source paths/metadata. Call after reserving the runtime budget.
Handle adapter exceptions as an explicit unavailable-evidence result, not a success.

The old deterministic replay does not infer errors from these observations. This
adapter does not implement Persons 2/3's verifier, policy adaptation or label scorer.

## Your remaining data checks

1. Open one raw, unannotated RGB recording; do not use labeled demo overlays.
2. Register its RGB folder and visually check three samples against source images.
3. With Person 3, establish the filename-to-annotation mapping and partitions.
   `annotation_alignment` deliberately stays `unverified` in this adapter; do not
   claim a benchmark until the separate evaluator verifies it.
4. Run one real VLM inspection and review its observations/uncertainty.
5. Give Person 2 the manifest plus one real packet, and Person 4 its frame paths.

Motion summaries remain unavailable. There is no joint parser, RGB-video conversion,
published-detector prediction importer or full dataset download in this module.
For RGB videos, use the existing `cookmemory.perception` adapter, preserve source
frame mapping, and arrange a versioned bridge before treating its output as Interface A.
