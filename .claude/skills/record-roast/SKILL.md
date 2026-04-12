---
name: record-roast
description: Record a coffee roasting session with the multi-mic synchronized recorder. Use when asked to record a roast, capture audio, start a recording session, or gather new training data.
---

## Record a Roast — Multi-mic Audio Capture

### Prerequisites
- macOS with CoreAudio Aggregate Device `RoastMics` configured in Audio MIDI Setup
  - See `docs/multi_mic_setup.md` for hardware setup and gain calibration
- Microphones physically connected and device visible
- venv active: `source venv/bin/activate`

### Steps

**1. Check devices**
```bash
python scripts/record_mics.py list-devices
```
Confirm `RoastMics` appears with at least 2 input channels.

**2. Start recording**
```bash
python scripts/record_mics.py record \
  --origin {{bean-origin}} \
  --roast-num {{N}}
```

Replace `{{bean-origin}}` with a lowercase slug (e.g. `panama-hortigal-estate`, `brazil-santos`) and `{{N}}` with the next roast number for that origin. Device, sample rate (44100 Hz), mic labels, and gains are resolved from `configs/default.yaml` automatically.

**3. Monitor**
- Heartbeat prints every 30 seconds: `[MM:SS] Recording...`
- Audio status warnings (buffer overruns) go to stderr, rate-limited to every 5s
- The user presses **Ctrl-C** to stop — do NOT stop it programmatically

**4. Output on stop**
- One mono float32 WAV per mic: `mic{N}-{origin}-roast{N}.wav`
- Session JSON: `{origin}-roast{N}-session.json`
- `_partial` suffix if duration < 60s (aborted/test session)
- Files land in `data/raw/` by default

### Custom options

Single mic (setup test):
```bash
python scripts/record_mics.py record --origin test --roast-num 1 --mics 1
```

Override gains:
```bash
python scripts/record_mics.py record --origin brazil-santos --roast-num 3 --gains 1.3 1.0
```

Suppress heartbeat:
```bash
python scripts/record_mics.py record --origin brazil-santos --roast-num 3 --quiet
```

### After recording — what's next

1. **Annotate in Label Studio** — load the WAV, mark the first crack region with a single time-span annotation, export JSON to `data/labels/`
2. **Propagate annotations** to paired mics:
   ```bash
   python scripts/propagate_annotations.py --dry-run
   python scripts/propagate_annotations.py
   ```
3. **Retrain** — see `train-model` skill

### Known limitation
Audio is held in RAM until Ctrl-C (in-memory buffering). A hard `SIGTERM` will lose the session. Always stop with Ctrl-C. (S21 #49 will replace this with streaming disk writes.)
