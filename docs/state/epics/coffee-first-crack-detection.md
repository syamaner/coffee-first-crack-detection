# Epic: Coffee First Crack Detection — HuggingFace Model Repository

**GitHub Issue**: [#1](https://github.com/syamaner/coffee-first-crack-detection/issues/1)
**Status**: ✅ Repository complete — pending dataset annotation & retraining
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
  - Created `scripts/evaluate_onnx.py` and `scripts/benchmark_onnx_pi.py` (no PyTorch dep)
  - Added `--threads` param for power-limited ARM64 devices
  - Quality: 93.3% acc / 0.933 F1 — identical across FP32/INT8 and Mac/RPi5
  - Latency: 4.4s (INT8, 1 thread) — above 500ms target (see notes below)
  - Hardware issue: RPi5 crashes with >1 ONNX thread on 5V/3A PSU (needs 5V/5A + active cooling)

---

## Active Context

**All 14 stories + RPi5 validation complete.**

**Model on HuggingFace**: https://huggingface.co/syamaner/coffee-first-crack-detection
- baseline_v1: 91.1% test acc / 0.913 F1 / 95.5% first_crack recall / 0.978 ROC-AUC
- Trained on mic-1 only (298 chunks, 6 roasts)

**RPi5 Validation** (issue #22, branch `feature/22-rpi5-onnx-validation`):
- INT8 ONNX accuracy matches Mac exactly (93.3% / 0.933 F1) — zero quantization loss
- Latency does NOT meet 500ms target on RPi5 with current AST model
- Hardware blockers: requires 5V/5A PSU (official RPi5 27W) + active cooling
- See "RPi5 Latency Results" section below for full numbers

**Dataset**: NOT yet published — pending annotation of mic-2 recordings.

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
| ONNX FP32 (Mac, auto threads) | 93.3% | 0.933 | 95.5% | p50=375ms |
| ONNX INT8 (Mac, auto threads) | 93.3% | 0.933 | 95.5% | p50=197ms |
| ONNX INT8 (RPi5, 1 thread) | 93.3% | 0.933 | 95.5% | p50=4,441ms |
| ONNX FP32 (RPi5, 1 thread) | 93.3% | 0.933 | 95.5% | p50=9,412ms |

### RPi5 Latency Results

Tested on RPi5 Model B Rev 1.1 (16GB), aarch64, Python 3.13.5, ONNX Runtime 1.24.4.
Limited to 1 thread due to under-voltage crashes with 5V/3A PSU (no active cooling).

| Model | Size | p50 (ms) | Target | Status |
|-------|------|----------|--------|--------|
| INT8 (RPi5, 1 thread) | 90MB | 4,441 | <500ms | ⚠️ FAIL |
| FP32 (RPi5, 1 thread) | 345MB | 9,412 | <500ms | ⚠️ FAIL |
| INT8 (Mac, auto) | 90MB | 197 | <500ms | ✅ PASS |
| FP32 (Mac, auto) | 345MB | 375 | <500ms | ✅ PASS |

**Next steps for latency:**
- Re-test with proper 5V/5A PSU + active cooler (expect ~4x faster with 4 threads → ~1.1s)
- Even with 4 threads, AST model (87M params) is likely too large for <500ms on RPi5
- Consider smaller model architectures (MobileNetV3, EfficientNet-B0) for real-time Pi use
- Alternative: offload inference to a companion device (Mac/PC) via network

---

## Decisions & Notes
- Using HF Trainer API (not custom Trainer) — enables push_to_hub=True and ecosystem compatibility
- save_pretrained / from_pretrained replaces .pt checkpoints
- Single epic issue (#1) with story checklist; one PR per story
- data/ is .gitignored — large audio files go to HuggingFace Datasets only
- RPi5 is inference-only via ONNX Runtime (no training)
- AGENTS.md is the single project rules file (no WARP.md)
