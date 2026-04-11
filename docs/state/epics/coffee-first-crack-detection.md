# Epic: Coffee First Crack Detection — HuggingFace Model Repository

**GitHub Issue**: [#1](https://github.com/syamaner/coffee-first-crack-detection/issues/1)
**Status**: 🔄 Active — Phase 7 complete (S19 + S20 delivered); recording data collection in progress
**Last Updated**: 2026-04-11

## Objective
Create a standalone, HuggingFace-publishable repository for training, evaluating, and publishing the coffee first crack audio detection model. Extracted from the `coffee-roasting` monorepo. Targets M3+ Mac (MPS), RTX 4090 (CUDA), and Raspberry Pi 5 (ONNX/CPU).

---

## Status Map

### Phase 1 — Scaffold & Data
- [x] S1 [#2](https://github.com/syamaner/coffee-first-crack-detection/issues/2): Scaffold repo structure, pyproject.toml, configs, AGENTS.md, skills ✅
  - GitHub repo created, full directory structure scaffolded
  - pyproject.toml, requirements.txt, requirements-pi.txt, configs/default.yaml
  - AGENTS.md, .claude/skills/ (train-model, evaluate-model, export-onnx, push-to-hub)
  - README.md model card, docs/data_preparation.md, docs/state/
  - Committed e6f8b74, pushed to main
- [x] S2 [#3](https://github.com/syamaner/coffee-first-crack-detection/issues/3): Port dataset.py with HF Datasets integration and filename metadata
  - `src/coffee_first_crack/dataset.py` created ✅
  - Filename parser (new convention + legacy), recordings.csv generation, create_dataloaders()
- [x] S3 [#4](https://github.com/syamaner/coffee-first-crack-detection/issues/4): Port model.py — HF-native ASTForAudioClassification wrapper
  - `src/coffee_first_crack/model.py` created ✅
  - build_model(), build_feature_extractor(), FirstCrackClassifier
- [x] S4 [#5](https://github.com/syamaner/coffee-first-crack-detection/issues/5): Port utils/metrics.py and utils/device.py
  - `src/coffee_first_crack/utils/metrics.py` and `utils/device.py` created ✅

### Phase 2 — Training & Evaluation
- [x] S5 [#6](https://github.com/syamaner/coffee-first-crack-detection/issues/6): Implement train.py with HF Trainer API (class-weighted loss) ✅
- [x] S6 [#7](https://github.com/syamaner/coffee-first-crack-detection/issues/7): Implement evaluate.py with full metrics report ✅
- [x] S7 [#8](https://github.com/syamaner/coffee-first-crack-detection/issues/8): Validate training — 95.6% acc, 0.955 F1, 0.978 ROC-AUC (test set) ✅

### Phase 3 — Inference & Export
- [x] S8 [#9](https://github.com/syamaner/coffee-first-crack-detection/issues/9): Port inference.py — sliding window + streaming FirstCrackDetector ✅
- [x] S9 [#10](https://github.com/syamaner/coffee-first-crack-detection/issues/10): Implement export_onnx.py (FP32 + INT8 quantized) ✅
- [x] S10 [#11](https://github.com/syamaner/coffee-first-crack-detection/issues/11): Create benchmark_platforms.py ✅

### Phase 4 — Publishing
- [x] S11 [#12](https://github.com/syamaner/coffee-first-crack-detection/issues/12): Implement scripts/push_to_hub.py ✅
- [x] S12 [#13](https://github.com/syamaner/coffee-first-crack-detection/issues/13): Write HuggingFace model card README.md ✅
- [x] S13 [#14](https://github.com/syamaner/coffee-first-crack-detection/issues/14): Create notebooks/quickstart.ipynb ✅
  - `notebooks/quickstart.ipynb` — Colab-compatible, loads model + dataset from HF Hub
  - Single-window classification with probability bar chart
  - Sliding-window demo with assembled roast recording and probability timeline plot
- [x] S14 [#15](https://github.com/syamaner/coffee-first-crack-detection/issues/15): Write pytest test suite ✅

### Phase 5 — Edge Validation
- [x] S15 [#22](https://github.com/syamaner/coffee-first-crack-detection/issues/22): Validate ONNX inference on Raspberry Pi 5 ✅
  - ONNX export: FP32 (345MB) + INT8 quantized (90MB)
  - Created `scripts/evaluate_onnx.py` and `scripts/benchmark_onnx_pi.py` (ONNX-only, no PyTorch for inference)
  - Added `--threads` param for power-limited ARM64 devices
  - Quality: 93.3% acc / 0.933 F1 — identical across FP32/INT8 and Mac/RPi5 (zero quantization loss)
  - Latency (INT8): 2.1s @ 4 threads (with fan) / 2.4s @ 2 threads / 4.4s @ 1 thread
  - Latency breakdown: feature extraction 49ms (2%), ONNX inference 2,019ms (98%)
  - Fixed `export_onnx.py` for current `optimum` API (removed deprecated `opset` param)
  - Hardware findings:
    - RPi5 crashes with >1 ONNX thread on 5V/3A PSU (`throttled=0x50000` — under-voltage)
    - No active cooling: idle 47°C → 77°C under load → thermal throttling (`throttled=0xe0000`)
    - With fan: 4 threads stable at 45°C, no throttling, consistent ~2.1s
    - Requires adequate PSU + active cooler for production use
  - **INT8 is the recommended model for RPi5** — 2x faster than FP32, zero quality loss

### Phase 6 — Dataset Expansion & Retraining
- [x] S16 [#26](https://github.com/syamaner/coffee-first-crack-detection/issues/26): Unified data preparation pipeline with fixed-size chunking
  - ✅ New `data_prep` module: `convert_labelstudio_export.py`, `chunk_audio.py`, `dataset_splitter.py`
  - ✅ All 15 files re-annotated with single-region approach in Label Studio
  - ✅ Chunking: 973 chunks (197 FC / 776 NFC, ~20% first_crack), fixed 10s windows
  - ✅ Recording-level split: train=587 (9 recs), val=195 (3 recs), test=191 (3 recs)
  - ✅ Training (baseline_v2, tuned): best epoch 2 — 95.38% val acc / 0.866 F1, early stopped epoch 5
  - ✅ Test eval: **97.4% acc / 0.925 F1 / 100% precision / 86.1% recall / 0.982 ROC-AUC**
  - ✅ Full-recording detection latency: 0.0s, 0.3s (mic-1), 27.4s (mic-2) — all detected
  - Hyperparams tuned: lr 5e-5→2e-5, weight_decay 0.01→0.05, early_stopping 5→3
- [x] S17 [#24](https://github.com/syamaner/coffee-first-crack-detection/issues/24): Capture mic-2 recordings and expand dataset ✅
  - 6 mic-2 recordings captured and annotated (absorbed into S16 dataset expansion)
  - mic2-brazil-roast1–4, mic2-brazil-santos-roast1–2
  - Annotation JSONs in `data/labels/`, included in 15-recording 973-chunk dataset
- [x] S18 [#25](https://github.com/syamaner/coffee-first-crack-detection/issues/25): Add HuggingFace inference Space + widget to model card ✅
  - `widget` block added to README.md frontmatter YAML
  - Two example WAVs uploaded to HF model repo under `audio_examples/` (16kHz, 10s)
  - Gradio Space live at https://huggingface.co/spaces/syamaner/coffee-first-crack-detection
  - Space linked to model card via `models:` field — appears in "Spaces using this model"
  - `spaces/` directory committed to git for version control

### Phase 7 — Data Collection Infrastructure
- [x] S19 [#46](https://github.com/syamaner/coffee-first-crack-detection/issues/46): Multi-mic synchronized recording tool for dataset expansion ✅
  - macOS CoreAudio Aggregate Device (`RoastMics`), 1–N mic support via `--mics` list
  - `scripts/record_mics.py` — `list-devices` + `record` subcommands; labels from `configs/default.yaml`
  - `configs/default.yaml` — `recording` section (device, sample_rate, mic_labels)
  - `docs/multi_mic_setup.md` — full hardware setup guide + calibrated gain settings (2026-04-11)
  - MCP server `find_usb_microphone()` patched to prefer `RoastMics` aggregate (coffee-roasting repo)
  - First real roasts captured: `panama-hortigal-estate-roast1` (12.8 min), `roast2` (15.1 min)
- [x] S20 [#47](https://github.com/syamaner/coffee-first-crack-detection/issues/47): Annotation propagation for paired multi-mic recordings ✅
  - `scripts/propagate_annotations.py` — reads `*-session.json`, propagates primary mic annotation to all paired mics
  - Uses `mic['file']` from session JSON — handles `_partial` suffix and future naming variants
  - Slots between `convert_labelstudio_export.py` and `chunk_audio.py`; zero pipeline changes
  - 16 tests; Copilot review comments addressed (PR #48)

---

## Active Context

**Phase 7 complete.** S19 (#46) + S20 (#47) delivered. PR #48 open on `feature/46-multi-mic-recorder`.

**S19 — Multi-mic recording** (`scripts/record_mics.py`):
- `RoastMics` CoreAudio Aggregate Device: FIFINE K669B (ch 0, Primary Clock) + ATR2100x (ch 1, Drift Correction)
- Calibrated gain: FIFINE 25.5 dB software + ~60% physical knob; ATR2100x Front Left/Right 18.38/18.75 dB
- Indefinite recording, Ctrl-C stop, `_partial` suffix for sessions < 60s
- Session JSON captures hardware labels, gains, duration, ISO timestamp
- First real paired roasts: `panama-hortigal-estate-roast1` (12.8 min), `roast2` (15.1 min) — pending annotation
- MCP conflict resolved: `find_usb_microphone()` now prefers `RoastMics` over raw FIFINE

**S20 — Annotation propagation** (`scripts/propagate_annotations.py`):
- Reads `*-session.json` → copies primary mic annotation JSON to all paired mics
- `--dry-run`, `--overwrite`, `--primary-mic` flags
- Uses `mic['file']` from session JSON for all filename resolution (handles `_partial` suffix)
- 16 tests, ruff + pyright clean, Copilot review comments addressed

**Gradio Space** (S18 / #25): https://huggingface.co/spaces/syamaner/coffee-first-crack-detection
- Dropdown: "First crack (10s)" / "No first crack (10s)" pre-loaded examples
- Upload any 10s WAV → first_crack / no_first_crack probability bars
- Linked to model card via `models:` field (appears in "Spaces using this model")
- `spaces/` committed to git; Gradio 6.11.0, CPU free tier

**Dataset v2**: 973 fixed 10s chunks from 15 recordings (9 legacy + 6 mic2)
- 197 first_crack (~20%) / 776 no_first_crack (~80%)
- Recording-level splitting prevents data leakage
- All files re-annotated with single-region approach (1 first_crack region per roast)

**baseline_v2 training (tuned hyperparams)**:
- Attempt 1 (lr=5e-5): peaked epoch 7 at 95.38% / 0.866 F1, oscillating loss, overfitting
- Attempt 2 (lr=2e-5, weight_decay=0.05, early_stop=3): best epoch 2, early stopped epoch 5
- Test set: **97.4% acc / 0.925 F1 / 100% precision / 86.1% recall / 0.982 ROC-AUC**
- Zero false positives (0 FP), 5 false negatives (5 FN out of 36 FC chunks)

**Full-recording detection latency** (sliding window on raw test WAVs):
- 25-10-19_1236-brazil-3 (mic-1): onset 452.7s → detected 453.0s — **0.3s delay**
- mic2-brazil-roast2 (mic-2): onset 599.6s → detected 627.0s — **27.4s delay**
- roast-2-costarica-hermosa-hp-a (mic-1): onset 441.0s → detected 441.0s — **0.0s delay**
- Mic-1 detection is near-instant; mic-2 has higher delay due to less training data from that mic

**Model on HuggingFace**: https://huggingface.co/syamaner/coffee-first-crack-detection
- baseline_v1: 91.1% test acc / 0.913 F1 / 95.5% first_crack recall / 0.978 ROC-AUC
- Trained on mic-1 only (298 chunks, 6 roasts)

**ONNX Models**: exported from baseline_v1 checkpoint, **published to HuggingFace Hub** (2026-04-04)
- FP32: 345MB → `onnx/fp32/model.onnx` on HF Hub
- INT8: 90MB → `onnx/int8/model_quantized.onnx` on HF Hub — recommended for RPi5
- Config JSONs (`config.json`, `preprocessor_config.json`) also uploaded for `from_pretrained()` support
- Zero quality degradation from quantization (identical confusion matrix)

**RPi5 Validation** (issue #22, branch `feature/22-rpi5-onnx-validation`):
- Quality: 93.3% acc / 0.933 F1 — identical to Mac across all ONNX variants
- **Production config**: INT8, 2 threads, threshold=0.90, adequate PSU + fan → **p50 = 2,452ms**
- 2 threads chosen to leave 2 cores free for MCP server + agent UI running on the same Pi
- Threshold 0.90: precision=0.952, recall=0.909, F1=0.930 — intentional production tradeoff (sweep script recommends 0.95 for zero FPs, but 0.90 preserves higher recall with only 1 FP)
- Threshold sweep + parameter simulation completed — see `results/` for full data
- ONNX inference module (`inference_onnx.py`) loads models from HF Hub with `--profile pi_inference`
- Latency breakdown: feature extraction 49ms (2%), ONNX model inference 2,019ms (98%)
- The AST model (87M params) is the bottleneck — too large for <500ms on RPi5 ARM64 CPU
- Hardware requirements: adequate PSU + active cooler for stable 2-thread operation
- See "RPi5 Validation Results" section below for full breakdown

**Dataset**: NOT yet published — pending push to HF Hub.

**Blog series**: planned 3-part series covering training, MCP servers, and .NET Aspire agent.

---

## Pending Work

### 1. Push baseline_v2 model + dataset to HuggingFace
```bash
python scripts/push_to_hub.py \
  --dataset-dir data/splits \
  --recordings-csv data/recordings.csv \
  --dataset-repo-id syamaner/coffee-first-crack-audio
```

### 5. Update README.md with v2 results
- Replace baseline_v1 metrics with retrained results
- Update `model-index` YAML values

---

## Latest Results

| Run | Accuracy | F1 | Recall (FC) | Notes |
|-----|----------|----|-------------|-------|
| Original prototype | ~93% | ~0.93 | 100% | coffee-roasting monorepo, custom Trainer |
| baseline_v1 (mic-1 only, MPS) | 95.6% (val) / 91.1% (test) | 0.955 / 0.913 | 95.5% | PR #18, 208 train samples |
| baseline_v2 MPS attempt 1 | 95.38% (val, epoch 7) | 0.866 | — | 587 train, overfitting after epoch 7, lr too high |
| **baseline_v2 (tuned, MPS)** | **97.4% (test)** | **0.925** | **86.1%** | **973 chunks, 15 recs, 100% precision, 0 FP** |
| ONNX FP32 (Mac, auto threads) | 93.3% | 0.933 | 95.5% | p50=375ms ✅ |
| ONNX INT8 (Mac, auto threads) | 93.3% | 0.933 | 95.5% | p50=197ms ✅ |
| ONNX INT8 (RPi5, 4 threads, fan) | 93.3% | 0.933 | 95.5% | p50=2,070ms ⭐ recommended |
| ONNX INT8 (RPi5, 2 threads) | 93.3% | 0.933 | 95.5% | p50=2,436ms ⚠️ thermal throttled |
| ONNX INT8 (RPi5, 1 thread) | 93.3% | 0.933 | 95.5% | p50=4,441ms ⚠️ |
| ONNX FP32 (RPi5, 1 thread) | 93.3% | 0.933 | 95.5% | p50=9,412ms ⚠️ |

### RPi5 Validation Results

Tested on RPi5 Model B Rev 1.1 (16GB), aarch64, Python 3.13.5, ONNX Runtime 1.24.4.

| Model | Threads | Fan | p50 (ms) | p95 (ms) | Temp | Throttled | Status |
|-------|---------|-----|----------|----------|------|-----------|--------|
| INT8 (RPi5) | 4 | ✅ | 2,070 | 2,090 | 45°C | No | ⭐ recommended |
| INT8 (RPi5) | 2 | ❌ | 2,436 | 2,704 | 77°C | Yes (thermal) | stable but throttled |
| INT8 (RPi5) | 1 | ❌ | 4,441 | 4,464 | — | No | stable, any PSU |
| FP32 (RPi5) | 1 | ❌ | 9,412 | 9,484 | — | No | stable, any PSU |
| INT8 (Mac) | auto | — | 197 | 200 | — | No | ✅ PASS |
| FP32 (Mac) | auto | — | 375 | 379 | — | No | ✅ PASS |

### Latency Breakdown (INT8, 4 threads, RPi5)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Feature extraction (ASTFeatureExtractor) | 49 | 2% |
| ONNX model inference | 2,019 | 98% |
| **Total** | **2,068** | **100%** |

Feature extraction is negligible. The bottleneck is entirely in the ONNX model forward pass (87M parameters through transformer layers).

### RPi5 Hardware Findings

- **Power**: 5V/3A PSU causes under-voltage crashes (`throttled=0x50000`) at >1 ONNX thread. Adequate PSU needed for multi-thread.
- **Thermal (no fan)**: idle 47°C → 77°C under 2-thread load → thermal throttling (`throttled=0xe0000`). Latency degrades from 2.4s to 2.7s.
- **Thermal (with fan)**: 4 threads stable at 45°C, no throttling, consistent latency.
- **Stability**: 1 thread stable on any PSU; 2+ threads need adequate PSU; 4 threads need adequate PSU + fan.
- **NVMe**: Boot from NVMe (Gen 2, default) — no PCIe stability issues observed.
- **Recommended config**: INT8 model, 2 threads (leaves cores for MCP/UI), threshold=0.90, adequate PSU + active cooler.

### Latency Analysis & Next Steps

- **INT8 is the go-to model for RPi5** — 2x faster than FP32, zero quality loss, 4x smaller (90MB vs 345MB)
- RPi5 is ~10x slower than Mac (M-series) for the same INT8 model
- 2→4 threads gives only ~15% improvement — diminishing returns from parallelism at this model size
- **Root cause**: AST model has 87M parameters — the transformer forward pass dominates (98% of latency)
- Feature extraction (2%) is not worth optimizing
- **Options for <500ms on RPi5**:
  1. Smaller model architecture (MobileNetV3/YAMNet-based — ~3-5M params, estimated ~50-100ms on Pi)
  2. Knowledge distillation from AST into a lightweight student model
  3. Offload inference to companion device (Mac/PC) via network/MCP
  4. Use RPi5 as audio capture + streaming, with inference on a more powerful device

---

## Decisions & Notes
- Using HF Trainer API (not custom Trainer) — enables push_to_hub=True and ecosystem compatibility
- save_pretrained / from_pretrained replaces .pt checkpoints
- Single epic issue (#1) with story checklist; one PR per story
- data/ is .gitignored — large audio files go to HuggingFace Datasets only
- RPi5 is inference-only via ONNX Runtime (no training)
- AGENTS.md is the single project rules file (no WARP.md)
- INT8 dynamic quantization (portable, ARM64-compatible) — zero accuracy loss vs FP32
- ONNX-only scripts (`evaluate_onnx.py`, `benchmark_onnx_pi.py`) avoid 148MB PyTorch install on Pi for model inference, though PyTorch CPU is still needed for `ASTFeatureExtractor` filterbank computation
- Thread limiting (`--threads` flag, default 2 on ARM64) is essential for RPi5 stability with standard PSUs
- RPi5 requires adequate PSU for multi-thread — standard USB-C chargers (even 96W Apple) only provide 5V/3A
- Active cooling recommended for 2+ thread inference — mandatory for 4-thread
- Production Pi runs MCP server + agent UI alongside inference — 2 ONNX threads leaves 2 cores free
- Detection threshold 0.90 chosen from threshold sweep: best F1/FP tradeoff (precision=0.952, 1 FP on 45-sample test set)
- Feature extraction is only 2% of total latency — optimizing it (e.g. C++ STFT) would be wasteful
