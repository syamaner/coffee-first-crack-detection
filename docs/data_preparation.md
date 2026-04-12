# Data Preparation Guide

Full pipeline from raw Audacity recording to training-ready splits.

---

## Overview

```
Audacity (.aup3)
    → Export WAV
        → Label Studio (annotate first_crack regions)
            → convert_labelstudio_export.py  (JSON annotations)
                → chunk_audio.py             (10s WAV chunks)
                    → dataset_splitter.py    (train/val/test splits)
                        → data/splits/       (ready for training)
```

---

## Step 1 — Export WAV from Audacity

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
   - The chunking pipeline (Step 4) will slice this into fixed 10-second training windows
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

**Multi-mic recordings:** Only mic1 needs to be annotated in Label Studio (both mics are sample-locked, so timestamps are identical). After converting, propagate annotations to the paired mic2 files:

```bash
python scripts/propagate_annotations.py --dry-run   # preview
python scripts/propagate_annotations.py              # create mic2 annotation JSONs
```

See `docs/multi_mic_setup.md` for the full paired-recording workflow.

---

## Step 4 — Chunk Audio into 10-Second Windows

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

## Step 5 — Stratified Train/Val/Test Split

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

## Step 6 — Generate recordings.csv Manifest

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
| Legacy prototype | mic-1-original | costarica-hermosa | 4 roasts | ✅ Annotated |
| Legacy prototype | mic-1-original | brazil | 5 roasts | ✅ Annotated |
| Single-mic recordings | mic-2-new | brazil | 4 roasts | ✅ Annotated |
| Single-mic recordings | mic-2-new | brazil-santos | 2 roasts | ✅ Annotated |
| Multi-mic recordings | mic-1-new (fifine) | panama-hortigal-estate | 3 roasts | ✅ Annotated |
| Multi-mic recordings | mic-2-new (audio-technica) | panama-hortigal-estate | 3 roasts | ✅ Annotated |

**Totals**: 21 recordings → 1,435 chunks (223 first_crack / 1,212 no_first_crack)

> When adding new recordings, re-run Steps 3–6 to rebuild the full dataset (the chunker and splitter process all annotation files in `data/labels/`).
