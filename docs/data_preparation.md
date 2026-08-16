# Data Preparation Guide

Full pipeline from raw recording to training-ready splits.

---

## Overview

```
Raw WAV (from the MCP recorder, or scripts/record_mics.py, or legacy Audacity)
    → UUID-safe staging + capture_manifest.json
        → Label Studio (annotate mic1's first_crack region)
            → convert_labelstudio_export.py  (human mic1 annotations + pair_id)
                → propagate_annotations.py    (derive mic2 + uncertainty provenance)
                    → chunk_audio.py          (10s WAV chunks)
                        → dataset_splitter.py (pair-level train/val/test splits)
                            → data/splits/    (ready for training)
```

There are three ways raw WAVs land in `data/raw/`. The primary source today is the
**coffee-roaster-mcp recorder**, which captures every supervised roast
automatically; `scripts/record_mics.py` is the bench dual-mic flow; and Audacity
export is the legacy path from the prototype era.

---

## Step 1 — Get raw WAVs into `data/raw/`

### 1a. Recordings from the coffee-roaster-mcp recorder (primary source)

During a supervised roast, the `coffee-roaster-mcp` recorder captures audio for
you (when `recording.enabled` and `recording.autocapture` are set, and after
`set_recording_metadata` has supplied the origin + roast number). Each roast
writes a **per-session capture directory** at
`<export_location>/<session-id>/` — by default
`<log_dir>/captures/<session-id>/`, which is gitignored so the large WAVs are
never committed. `<session-id>` is the roast session's run id.

Each session directory contains:

| File | What it is |
|------|-----------|
| `mic{N}-{origin}-roast{n}.wav` | One WAV per capture device, 1-based device order. `mic1` is the detector's teed stream. **16 kHz, mono, 16-bit PCM.** |
| `roast.recording.json` | The recording manifest (schema v2) — see below. |
| `{origin}-roast{n}-session.json` | An annotation-pipeline session JSON, shape-compatible with `record_mics.py`, so `propagate_annotations.py` (Step 4) reads it directly. |

Do not flatten or rename this directory tree by hand. Original basenames are not
globally unique: two different physical sessions can legitimately contain the
same origin/roast filenames. Stage them with the capture ingester instead:

```bash
# Validate only; hashes every source and writes nothing.
venv/bin/python -m coffee_first_crack.data_prep.ingest_mcp_captures \
  --capture-root /Users/sertanyamaner/roasts/captures \
  --output data/raw/mcp-captures \
  --dry-run

# Stage validated copies.
venv/bin/python -m coffee_first_crack.data_prep.ingest_mcp_captures \
  --capture-root /Users/sertanyamaner/roasts/captures \
  --output data/raw/mcp-captures
```

The capture-directory UUID is the immutable `pair_id`. Staged WAV filenames are
prefixed with it, so duplicate original basenames cannot overwrite one another.
The ingester validates both sidecars, mic metadata, WAV headers, safe basenames,
session identity, and destination uniqueness; copies both streams and both
sidecars; verifies copy checksums; then hashes the sources again to prove they
did not change. Outputs are local and gitignored:

```
data/raw/mcp-captures/
  capture_manifest.json
  source_checksums.sha256
  mic1/       # Label Studio import directory: human-labelled tasks only
  mic2/       # paired automatic-annotation targets; do not import into Label Studio
  sessions/   # UUID-prefixed copies of both session metadata files
```

The `roast.recording.json` manifest (written by
`coffee_roaster_mcp.audio._write_recording_sidecar`) has this shape:

```jsonc
{
  "schema_version": 2,
  "session_id": "<run-id>",
  "recording_started_monotonic_seconds": 1234.56,  // monotonic clock at record start; null if unknown
  "milestones": {                                   // recording-relative seconds, or null per milestone
    "first_crack": 512.3,
    "drop": 640.0
  },
  "streams": [                                       // one entry per WAV, device order (index 0 = detector/mic1)
    {
      "device": "<device label>",
      "wav_filename": "mic1-<origin>-roast<n>.wav",
      "sample_rate": 16000,
      "channels": 1,
      "sample_width_bytes": 2,
      "frame_count": 9600000,
      "duration_seconds": 600.0
    }
    // ...mic2, mic3 as configured
  ]
  // The first (detector) stream's fields are also mirrored at the top level
  // (wav_filename / sample_rate / channels / sample_width_bytes / frame_count /
  // duration_seconds) as a v1 back-compat convenience.
}
```

The staged `capture_manifest.json` is authoritative downstream. It preserves
origin, roast number, mic number/label, original filename, source path, staged
path, checksum, duration, and `pair_id` for every stream.

### 1b. Recordings from `scripts/record_mics.py` (bench dual-mic)

The dual-mic bench flow records a paired session directly. See
`docs/multi_mic_setup.md` for the full hardware setup. In brief:

```bash
python scripts/record_mics.py list-devices
python scripts/record_mics.py record --origin <bean-slug> --roast-num <n>
```

Press **Ctrl-C** to stop. This writes, into `data/raw/`:

```
mic1-<origin>-roast<n>.wav          # detector mic (e.g. FIFINE, ch 0)
mic2-<origin>-roast<n>.wav          # paired mic (e.g. ATR2100x, ch 1)
<origin>-roast<n>-session.json      # hardware metadata + the mics list
```

These WAVs are **44100 Hz**; `chunk_audio.py` resamples to 16 kHz. Sessions
shorter than 60 s are saved with a `_partial` suffix and excluded by convention.
This is a distinct alignment path from MCP capture. The Aggregate Device exposes
one multichannel CoreAudio stream with Drift Correction, so the recorded bench
channels are sample-aligned and the legacy propagation mode remains valid.
coffee-roaster-mcp opens device streams independently; those streams are **not
sample-locked** and must use manifest-aware uncertainty handling in Step 4.

### 1c. Legacy: Audacity export

The prototype-era recordings were exported by hand from Audacity. Kept for
reference — new recordings use one of the two flows above.

1. Open your `.aup3` project in Audacity
2. **File → Export Audio…**
3. Settings:
   - Format: **WAV (Microsoft)**
   - Encoding: **Signed 16-bit PCM**
   - Channels: **Mono** (if stereo, use Tracks → Mix → Mix Stereo Down to Mono first)
   - Sample rate: leave at native **44100 Hz** (pipeline resamples to 16kHz)
4. Filename — use the convention: `mic2-{origin}-roast{n}-{date}.wav`
   - Example: `mic2-brazil-roast5-03-04-26.wav`
5. Save to: `/Users/sertanyamaner/git/coffee-first-crack-detection/data/raw/`

---

## Step 2 — Annotate in Label Studio

### 2a. Start Label Studio (if not running)

```bash
label-studio start
```

Opens at http://localhost:8080

### 2b. Create or open your project

1. Go to **Projects → New Project** (or open your existing first crack project)
2. Project name: e.g. `Coffee First Crack — MCP Mic1 2026-08`

### 2c. Configure the labelling interface (first time only)

1. Go to **Settings → Labelling Interface**
2. Select **Audio/Speech Processing → Audio Classification with Regions** (or paste the XML below)
3. Use this label config:

```xml
<View>
  <Audio name="audio" value="$audio" zoom="true" waveformHeight="100"/>
  <Labels name="label" toName="audio">
    <Label value="first_crack" background="#FF0000"/>
  </Labels>
</View>
```

Only the `first_crack` label is needed. Everything outside annotated regions is implicitly `no_first_crack`.

### 2d. Import audio files

1. Go to your project → **Import**
2. Click **Upload Files** and select every WAV in:
   `/Users/sertanyamaner/git/coffee-first-crack-detection/data/raw/mcp-captures/mic1/`
   - For the 16 Aug 2026 corpus this is exactly **38 mic1 tasks**.
   - Do not import `mcp-captures/mic2/`; those annotations are derived after export.
3. Click **Import**

### 2e. Annotate each file

1. Click a task to open it
2. You will see the waveform. Press **Play** to listen
3. **Draw one `first_crack` region per roast:**
   - Select the `first_crack` label in the left panel
   - Click and drag on the waveform to draw **one region** from the **first pop** to the **end of consistent cracking**
   - This region typically spans 1–5 minutes of audio
   - You do NOT need to annotate individual pops — one continuous region is correct
   - The chunking pipeline (Step 5) will slice this into fixed 10-second training windows
4. Press **Submit** (or Ctrl+Enter) to save the annotation
5. Move to the next task

### 2f. Tips for accurate annotation

- **Zoom in** on the waveform (scroll wheel) to find the first pop precisely
- Use **spacebar** to pause/resume playback
- First crack pops are sharp transient spikes in the waveform — visually distinct from background noise
- When in doubt, mark a slightly wider region rather than too narrow
- If a roast shows no first crack (unlikely but possible), leave no regions drawn and submit

### 2g. Export annotations

1. Go to your project → **Export**
2. Format: **JSON**
3. Click **Export** — this downloads a file like `project-1-at-YYYY-MM-DD-HH-MM-hashcode.json`
4. Move the exported file to (create the local directory if needed):
   ```
   /Users/sertanyamaner/git/coffee-first-crack-detection/data/labels/mcp/labelstudio-export.json
   ```

`data/labels/mcp/` is gitignored. Do not access, copy, or modify Label Studio's
internal database; the JSON export is the only handoff artifact.

---

## Step 3 — Convert Label Studio Export

```bash
python -m coffee_first_crack.data_prep.convert_labelstudio_export \
  --input data/labels/mcp/labelstudio-export.json \
  --output data/labels/mcp \
  --data-root data/raw \
  --manifest data/raw/mcp-captures/capture_manifest.json
```

This produces one `{UUID}__{mic1-stem}.json` annotation file per task and records
the manifest `pair_id`, mic number, original filename, and human Label Studio
provenance. Unknown tasks, mic2 tasks, ambiguous basenames, missing WAVs, and
existing output files fail closed.

For multi-mic sessions, you annotate **mic1 only** in Label Studio; Step 4
propagates that annotation to the paired mics.

---

## Step 4 — Propagate Annotations to Paired Mics

For staged MCP captures, derive the linked mic2 annotation from every converted
human mic1 annotation:

```bash
venv/bin/python scripts/propagate_annotations.py \
  --manifest data/raw/mcp-captures/capture_manifest.json \
  --staging-root data/raw/mcp-captures \
  --audio-root data/raw \
  --labels-dir data/labels/mcp \
  --dry-run

venv/bin/python scripts/propagate_annotations.py \
  --manifest data/raw/mcp-captures/capture_manifest.json \
  --staging-root data/raw/mcp-captures \
  --audio-root data/raw \
  --labels-dir data/labels/mcp
```

MCP devices run on independent clocks and are not sample-locked. Historical
captures do not record exact per-stream start offsets, so copied mic1 boundary
timestamps are not represented as exact mic2 ground truth. Each mic2 JSON records:

- `pair_id` and `derived_from`;
- `derivation_method` and the independent-clock alignment statement;
- that pair's observed duration delta;
- the corpus-wide maximum observed duration delta as
  `alignment_uncertainty_seconds` (3.5 s for the 16 Aug corpus);
- the deterministic training policy.

The chunker excludes every derived mic2 window that intersects the uncertainty
guard around either first-crack boundary. Interior windows remain usable; every
exclusion is written to `chunk_manifest.jsonl`. Missing human annotations,
missing targets, ambiguous identities, and pre-existing derived outputs fail
instead of being silently skipped.

The no-`--manifest` mode remains for the separate sample-aligned Aggregate
Device bench workflow from `scripts/record_mics.py`.

See `docs/multi_mic_setup.md` for the full paired-recording workflow.

---

## Step 5 — Chunk Audio into 10-Second Windows

```bash
python -m coffee_first_crack.data_prep.chunk_audio \
  --labels-dir data/labels \
  --audio-dir data/raw \
  --output-dir data/processed \
  --window-size 10 \
  --sample-rate 44100
```

Slides a fixed 10-second window across each recording. Each window is labelled `first_crack` if ≥50% of it overlaps with annotated first crack regions, otherwise `no_first_crack`.

Output structure:
```
data/processed/
  first_crack/      ← 10s windows that overlap ≥50% with first crack
  no_first_crack/   ← 10s windows of background roast noise
  processing_summary.md
  chunk_manifest.jsonl  # pair_id, stream identity, inclusion/exclusion provenance
```

Chunk filenames encode the source recording and window start time:
`roast-1-costarica-hermosa-hp-a_w0530.0.wav` = window starting at 530.0s.

---

## Step 6 — Stratified Train/Val/Test Split

```bash
python -m coffee_first_crack.data_prep.dataset_splitter \
  --input data/processed \
  --output data/splits \
  --train 0.7 --val 0.15 --test 0.15 \
  --seed 42
```

Splits at the **physical pair/session level**, never at microphone filename
level. Both streams from a roast receive one assignment. Legacy session
sidecars deterministically group existing pairs such as both Panama microphones;
legacy single-mic files receive a unique single-recording pair identity.

Output:
```
data/splits/
  train/{first_crack,no_first_crack}/
  val/{first_crack,no_first_crack}/
  test/{first_crack,no_first_crack}/
  split_report.md
  split_integrity.json  # machine-checkable pair-ID sets and empty intersections
```

`split_integrity.json` asserts that train, validation, and test pair-ID
intersections are empty and reports physical-session counts separately from
stream/recording counts. The fixed seed makes assignments deterministic.

---

## Step 7 — Generate recordings.csv Manifest

```bash
python -c "
from coffee_first_crack.dataset import generate_recordings_manifest
generate_recordings_manifest('data/raw', 'data/recordings.csv')
"
```

This auto-parses filenames to extract microphone and coffee origin metadata.

---

## File Naming Reference

| Format | Example | Notes |
|--------|---------|-------|
| Multi-mic (mic-1) | `mic1-panama-hortigal-estate-roast1.wav` | Recorded simultaneously with mic-2; separate hardware |
| Multi-mic (mic-2) | `mic2-panama-hortigal-estate-roast1.wav` | Recorded simultaneously with mic-1; separate hardware |
| Single mic (mic-2) | `mic2-brazil-roast1-21-02-26-10-37.wav` | Parser extracts mic=mic-2-new, origin=brazil |
| Single mic (mic-2) | `mic2-brazil-santos-roast1-04-04-26-17-52.wav` | Multi-word origins use hyphens |
| Legacy (mic-1) | `roast-1-costarica-hermosa-hp-a.wav` | Handled by legacy mapping table |
| Legacy (mic-1) | `25-10-19_1103-costarica-hermosa-5.alog.wav` | Handled by legacy mapping table |

When recording with two microphones, each mic produces a distinct stream, but
the streams share one physical `pair_id` and can never cross splits.

---

## Current Dataset Status

| Source | Mic | Origin | Files | Status |
|--------|-----|--------|-------|--------|
| Legacy prototype | mic-1-original | costarica-hermosa | 5 roasts | ✅ Annotated |
| Legacy prototype | mic-1-original | brazil | 4 roasts | ✅ Annotated |
| Single-mic recordings | mic-2-new | brazil | 4 roasts | ✅ Annotated |
| Single-mic recordings | mic-2-new | brazil-santos | 2 roasts | ✅ Annotated |
| Multi-mic recordings | mic-1-new (fifine) | panama-hortigal-estate | 3 roasts | ✅ Annotated |
| Multi-mic recordings | mic-2-new (audio-technica) | panama-hortigal-estate | 3 roasts | ✅ Annotated |

**Totals** (baseline_v5, per `data/splits/split_report.md`): 21 recordings →
1,435 chunks (223 first_crack / 1,212 no_first_crack).

The committed baseline report currently groups by stream stem and leaks paired
Panama roasts across splits (roast 1: mic1 train / mic2 test; roast 2: mic2
validation / mic1 test). Treat baseline_v5 evaluation as potentially optimistic
until rebuilt with this pair-aware pipeline.

## Exact resume after the human export

After placing the JSON export at `data/labels/mcp/labelstudio-export.json`, run
Steps 3 and 4 exactly as shown above. Then resume the deterministic rebuild,
training, evaluation, and ONNX comparison with:

```bash
source venv/bin/activate
./scripts/rebuild_and_train.sh baseline_v6_pair_aware

python -m coffee_first_crack.export_onnx \
  --model-dir experiments/baseline_v6_pair_aware/checkpoint-best \
  --output-dir exports/onnx-baseline-v6-pair-aware --quantize

python scripts/evaluate_onnx.py \
  --onnx-dir exports/onnx-baseline-v6-pair-aware/int8 \
  --test-dir data/splits/test \
  --output results/baseline_v6_pair_aware_int8_eval.json
```

Compare the new PyTorch and INT8 metrics with baseline_v5, but do not publish a
new model or replace production ONNX artifacts until the leakage-free evaluation
is complete and reviewed.
