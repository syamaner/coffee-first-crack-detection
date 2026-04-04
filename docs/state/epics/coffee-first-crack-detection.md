# Epic: Coffee First Crack Detection — HuggingFace Model Repository

**GitHub Issue**: [#1](https://github.com/syamaner/coffee-first-crack-detection/issues/1)
**Status**: ✅ Phases 1–5 complete — pending dataset annotation, retraining & Pi hardware
**Last Updated**: 2026-04-04

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
  - Latency (INT8): 4.4s @ 1 thread, 2.4s @ 2 threads — above 500ms target
  - Fixed `export_onnx.py` for current `optimum` API (removed deprecated `opset` param)
  - Hardware findings:
    - RPi5 crashes with >1 ONNX thread on 5V/3A PSU (`throttled=0x50000` — under-voltage)
    - No active cooling: idle 47°C → 77°C under load → thermal throttling (`throttled=0xe0000`)
    - Stable at 2 threads with adequate PSU, but thermally throttled without fan
    - Requires official 27W (5V/5A) PSU + active cooler for production use

---

## Active Context

**All 14 stories + RPi5 validation (S15) complete.** 15 stories across 5 phases.

**Model on HuggingFace**: https://huggingface.co/syamaner/coffee-first-crack-detection
- baseline_v1: 91.1% test acc / 0.913 F1 / 95.5% first_crack recall / 0.978 ROC-AUC
- Trained on mic-1 only (298 chunks, 6 roasts)

**ONNX Models**: exported from baseline_v1 checkpoint
- FP32: 345MB (`exports/onnx/fp32/model.onnx`)
- INT8: 90MB (`exports/onnx/int8/model_quantized.onnx`) — recommended for RPi5
- Zero quality degradation from quantization (identical confusion matrix)

**RPi5 Validation** (issue #22, branch `feature/22-rpi5-onnx-validation`):
- Quality: 93.3% acc / 0.933 F1 — identical to Mac across all ONNX variants
- Latency: 2.4s (INT8, 2 threads) / 4.4s (INT8, 1 thread) — does NOT meet 500ms target
- The AST model (87M params) is too large for real-time inference on RPi5 ARM64 CPU
- Hardware requirements: official 27W (5V/5A) PSU + active cooler mandatory
- See "RPi5 Validation Results" section below for full breakdown

**Dataset**: NOT yet published — pending annotation of mic-2 recordings.

**Blog series**: planned 3-part series covering training, MCP servers, and .NET Aspire agent.

---

## Pending Work

### 1. Annotate mic-2 recordings in Label Studio
- 4 WAV files in `data/raw/`: `mic2-brazil-roast{1..4}-*.wav`
- Follow `docs/data_preparation.md` Label Studio steps
- Export JSON → run `convert_labelstudio_export.py`

### 2. Expand dataset and re-split
```bash
python -m coffee_first_crack.data_prep.chunk_audio \
  --labels-dir data/labels --audio-dir data/raw --output-dir data/processed

python -m coffee_first_crack.data_prep.dataset_splitter \
  --input data/processed --output data/splits --train 0.7 --val 0.15 --test 0.15 --seed 42
```

### 3. Retrain on expanded dataset (mic-1 + mic-2)
```bash
python -m coffee_first_crack.train \
  --data-dir data/splits --experiment-name baseline_v2 --push-to-hub
```

### 4. Push updated dataset to HuggingFace
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
| ONNX FP32 (Mac, auto threads) | 93.3% | 0.933 | 95.5% | p50=375ms ✅ |
| ONNX INT8 (Mac, auto threads) | 93.3% | 0.933 | 95.5% | p50=197ms ✅ |
| ONNX INT8 (RPi5, 2 threads) | 93.3% | 0.933 | 95.5% | p50=2,436ms ⚠️ thermal throttled |
| ONNX INT8 (RPi5, 1 thread) | 93.3% | 0.933 | 95.5% | p50=4,441ms ⚠️ |
| ONNX FP32 (RPi5, 1 thread) | 93.3% | 0.933 | 95.5% | p50=9,412ms ⚠️ |

### RPi5 Validation Results

Tested on RPi5 Model B Rev 1.1 (16GB), aarch64, Python 3.13.5, ONNX Runtime 1.24.4.
No active cooling. PSU tested: 5V/3A (crashed at >1 thread), adequate PSU (stable at 2 threads but thermally throttled).

| Model | Threads | p50 (ms) | p95 (ms) | Target | Status |
|-------|---------|----------|----------|--------|--------|
| INT8 (RPi5) | 2 | 2,436 | 2,704 | <500ms | ⚠️ FAIL (throttled) |
| INT8 (RPi5) | 1 | 4,441 | 4,464 | <500ms | ⚠️ FAIL |
| FP32 (RPi5) | 1 | 9,412 | 9,484 | <500ms | ⚠️ FAIL |
| INT8 (Mac) | auto | 197 | 200 | <500ms | ✅ PASS |
| FP32 (Mac) | auto | 375 | 379 | <500ms | ✅ PASS |

### RPi5 Hardware Findings

- **Power**: 5V/3A PSU causes under-voltage crashes (`throttled=0x50000`) at >1 ONNX thread. Need official RPi5 27W (5V/5A) PSU.
- **Thermal**: No active cooling → idle 47°C → 77°C under 2-thread load → thermal throttling (`throttled=0xe0000`). Latency degrades from 2.4s to 2.7s as throttling kicks in.
- **Stability**: 1 thread stable on any PSU; 2 threads stable with adequate PSU; 4 threads untested (needs 27W PSU + active cooler).
- **NVMe**: Boot from NVMe (Gen 2, default) — no PCIe stability issues observed.

### Latency Analysis & Next Steps

- INT8 quantization gives ~2x speedup over FP32 with zero quality loss
- RPi5 is ~12x slower than Mac (M-series) for the same INT8 model at 2 threads
- Even with 4 threads + proper hardware, extrapolated INT8 latency is ~1.2s — still above 500ms
- **Root cause**: AST model has 87M parameters — too large for real-time ARM64 CPU inference
- **Options for <500ms on RPi5**:
  1. Smaller model architecture (MobileNetV3, EfficientNet-B0 based audio classifier)
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
- RPi5 requires 27W (5V/5A) PSU — standard USB-C chargers (even 96W Apple) only provide 5V/3A
- Active cooling is mandatory for sustained multi-thread inference on RPi5
