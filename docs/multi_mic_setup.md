# Multi-Mic Recording Setup

Step-by-step guide for configuring and using the dual-microphone recording setup
for coffee roasting sessions. macOS only.

---

## Hardware

| Slot | Microphone | Type | Role |
|---|---|---|---|
| mic1 | FIFINE K669B | USB condenser | Primary Clock, ch 0 |
| mic2 | Audio-Technica ATR2100x | USB dynamic | Drift Correction, ch 1 |

---

## One-Time macOS Setup (Audio MIDI Setup)

### 1. Create the Aggregate Device

1. Open **Audio MIDI Setup** (`/Applications/Utilities/`)
2. Click **+** (bottom-left) → **Create Aggregate Device**
3. Rename it **`RoastMics`**

### 2. Configure subdevices

In the device list on the right:

| Use | Device | In | Drift Correction |
|---|---|---|---|
| ✅ | USB PnP Audio Device (FIFINE) | 1 | off — this is the Primary Clock |
| ✅ | ATR2100x-USB Microphone | 2 | ✅ |

### 3. Set Clock Source

Set **Clock Source** to **USB PnP Audio Device** (FIFINE).
This makes the FIFINE the timing reference and forces the ATR2100x to resample
to stay sample-locked.

### 4. Verify channel assignment

The **Input Channels** strip at the top should show:

```
Channel 1 (1)   Front Left (2)   Front Right (3)
```

- **Channel 1** = FIFINE → `mic1` in scripts
- **Front Left** = ATR2100x capsule → `mic2` in scripts
- **Front Right** = ATR2100x (stereo duplicate, not used)

The sample rate will show **48.0 kHz** — this is the native rate of both devices.
The recording script captures at 44100 Hz; librosa resamples to 16 kHz for the
AST model during chunking.

---

## Calibrated Gain Settings (validated 2026-04-11)

These settings produced clean recordings without clipping on real Panama Hortigal
Estate roasts. Set these before each session.

### FIFINE K669B (USB PnP Audio Device)

- **Physical knob on mic**: ~60% of max (adjust if still clipping on crack pops)
- **macOS software gain**: `23.00 dB` (slider value `0.677` on the 0–31 scale)

### Audio-Technica ATR2100x

In Audio MIDI Setup → select **ATR2100x-USB Microphone** (the individual device,
not RoastMics):

- **Front Left**: `20.1 dB` (value `0.885`)
- **Front Right**: `20.1 dB` (value `0.889`) — keep matched to Front Left

> **Rule:** hardware-first gain is always preferred. If the FIFINE is clipping,
> turn the physical knob down — don't touch the software slider.
> Software gain amplifies the noise floor equally with the signal.

---

## Aggregate Device Stability

The RoastMics Aggregate Device will show **"Offline device"** or **0 input channels**
if another process holds an exclusive claim on one of the subdevices.

**Fix:** Click the Clock Source dropdown and reselect **USB PnP Audio Device**.
Then uncheck and re-check the Use boxes for both mics. The channel count should
return to 1+2 immediately.

**Prevention:** The first crack detection MCP server is configured to open
`RoastMics` (the aggregate) rather than the raw FIFINE device. Both the MCP
detector and `record_mics.py` can run simultaneously as long as both access
the hardware through the aggregate.

---

## Recording

```bash
# Verify RoastMics is visible with 3 input channels
python scripts/record_mics.py list-devices

# Start a session
python scripts/record_mics.py record --origin <bean-slug> --roast-num <n>
```

Press **Ctrl-C** to stop. Output:

```
data/raw/
  mic1-<origin>-roast<n>.wav          # FIFINE (ch 0)
  mic2-<origin>-roast<n>.wav          # ATR2100x Front Left (ch 1)
  <origin>-roast<n>-session.json      # hardware metadata + timestamps
```

Sessions shorter than 60 seconds are saved with a `_partial` suffix and excluded
from the annotation pipeline by convention.

### Quick level check after a session

```bash
python - <<'EOF'
import numpy as np, soundfile as sf
from pathlib import Path

raw = Path("data/raw")
origin, roast = "brazil", 5   # edit as needed
for mic in (1, 2):
    d, sr = sf.read(str(raw / f"mic{mic}-{origin}-roast{roast}.wav"))
    peak = float(np.max(np.abs(d)))
    print(f"mic{mic}: peak={peak:.3f} ({20*np.log10(peak):+.1f} dBFS)")
EOF
```

Target: peaks between **−12 dBFS and −6 dBFS** during roast events.
If peak = 1.000 on non-knock audio, reduce the FIFINE physical knob by ~20%.

**Memory usage:** the recorder buffers all audio in RAM until Ctrl-C, then writes.
At 44100 Hz, 2 channels, float32, a 20-minute session uses ~420 MB.
This is intentional for the target hardware (M3 Max, 128 GB RAM) and keeps the code
simple. It also means that if a write fails at the end, all captured audio is still
in RAM. Do not use on memory-constrained hardware for sessions longer than ~1 hour.

---

## Annotation Workflow (Paired Recordings)

Because both files are sample-locked, only mic1 needs to be annotated in Label
Studio. The timestamps are identical on mic2.

```
1. Upload mic1-<origin>-roast<n>.wav to Label Studio
2. Draw the first_crack region (use the detector timestamp as a starting point)
3. Export Label Studio JSON
4. Convert:
   python -m coffee_first_crack.data_prep.convert_labelstudio_export \
     --input data/labels/export.json \
     --output data/labels \
     --data-root data/raw
   # produces: data/labels/mic1-<origin>-roast<n>.json

5. Propagate to mic2 (automatic):
   python scripts/propagate_annotations.py
   # produces: data/labels/mic2-<origin>-roast<n>.json

6. Chunk and split as normal:
   python -m coffee_first_crack.data_prep.chunk_audio ...
   python -m coffee_first_crack.data_prep.dataset_splitter ...
```

---

## Channel Mapping Reference

| mic N | Aggregate channel index | sounddevice index | Physical device |
|---|---|---|---|
| mic1 | 0 | 0 | FIFINE K669B (Channel 1) |
| mic2 | 1 | 1 | ATR2100x Front Left |
| — | 2 | 2 | ATR2100x Front Right (unused) |

To add a third mic: add it to the Aggregate Device in Audio MIDI Setup and add
`3: <label>` under `recording.mic_labels` in `configs/default.yaml`. No code
changes required.

---

## configs/default.yaml — recording section

```yaml
recording:
  device: RoastMics
  sample_rate: 44100
  mic_labels:
    1: fifine
    2: audiotechnica
```
