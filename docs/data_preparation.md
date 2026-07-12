# Data Preparation Guide

Full pipeline from raw recording to training-ready splits.

---

## Overview

```
Raw WAV (from the MCP recorder, or scripts/record_mics.py, or legacy Audacity)
    → data/raw/ (mic{N}-{origin}-roast{n}.wav, per naming convention)
        → Label Studio (annotate mic1's first_crack region)
            → convert_labelstudio_export.py  (per-file JSON annotations)
                → propagate_annotations.py    (copy mic1 → paired mics)
                    → chunk_audio.py          (10s WAV chunks)
                        → dataset_splitter.py (train/val/test splits)
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

These WAVs are already at the model's native **16 kHz** (unlike
`record_mics.py`'s 44.1 kHz), and `chunk_audio.py` resamples internally anyway, so
they need no special handling: **copy or rename the `mic*.wav` files into
`data/raw/`** following the naming convention below, and copy the
`{origin}-roast{n}-session.json` alongside them so annotation propagation works.

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

The manifest is informational for the training pipeline (`chunk_audio.py` and
`dataset_splitter.py` do not read it); annotation propagation keys on the
`{origin}-roast{n}-session.json` instead.

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
The `-session.json` here is the same shape the MCP recorder writes, so Step 4
propagation works identically.

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
2. Project name: e.g. `Coffee First Crack — Mic2 Brazil`

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
2. Click **Upload Files** and select the WAV files from `data/raw/`
   - Import all new mic-2 files at once
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
4. Move the exported file to:
   ```
   /Users/sertanyamaner/git/coffee-first-crack-detection/data/labels/
   ```

---

## Step 3 — Convert Label Studio Export

```bash
python -m coffee_first_crack.data_prep.convert_labelstudio_export \
  --input data/labels/project-1-at-2026-04-12-19-41-e5863e6e.json \
  --output data/labels \
  --data-root data/raw
```

This produces one `{stem}.json` annotation file per audio file in `data/labels/`.

For multi-mic sessions, you annotate **mic1 only** in Label Studio; Step 4
propagates that annotation to the paired mics.

---

## Step 4 — Propagate Annotations to Paired Mics

Multi-mic sessions (from the MCP recorder or `record_mics.py`) capture every mic
sample-locked, so their first-crack timestamps are identical. Annotate **mic1
only** in Label Studio, then propagate mic1's converted annotation JSON to every
paired mic in the session:

```bash
python scripts/propagate_annotations.py --dry-run   # preview
python scripts/propagate_annotations.py              # create mic2..N annotation JSONs
```

**How the pairing works.** `propagate_annotations.py` discovers sessions by
reading the `{origin}-roast{n}-session.json` files in `data/raw` (the
`--session-dir`). Each session JSON carries `origin`, `roast_num`, `sample_rate`,
and a `mics` list of `{mic_num, label, file}` entries. The script reads the
primary mic's annotation JSON (`--primary-mic`, default `1`) from `data/labels`
and writes an identical annotation JSON — same `annotations`, per-mic
`audio_file`/`duration` — for every other mic listed in the session. Sessions
with no paired mics, and mics whose annotation JSON already exists (without
`--overwrite`), are skipped.

**This is why the `{origin}-roast{n}-session.json` must sit in `data/raw`
alongside the WAVs** (Step 1). Both the MCP recorder and `record_mics.py` write
that file in exactly the shape this script expects, so MCP-captured sessions
propagate automatically — no extra tooling. If a set of paired WAVs has **no**
session JSON (e.g. hand-assembled from loose files), propagation cannot pair
them; in that case annotate each mic by hand in Label Studio and run Step 3 per
file, skipping this step.

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

Splits at the **recording level** (not chunk level) to prevent data leakage — all chunks from the same source recording go to the same split.

Output:
```
data/splits/
  train/{first_crack,no_first_crack}/
  val/{first_crack,no_first_crack}/
  test/{first_crack,no_first_crack}/
  split_report.md
```

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

When recording with two microphones, each mic produces an independent WAV file. Both are annotated (annotations can be propagated via `scripts/propagate_annotations.py`) and treated as separate recordings for splitting.

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

> When adding new recordings, re-run Steps 3–7 to rebuild the full dataset (the chunker and splitter process all annotation files in `data/labels/`).
