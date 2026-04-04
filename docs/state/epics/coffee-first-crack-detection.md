# Epic: Coffee First Crack Detection — HuggingFace Model Repository

**GitHub Issue**: [#1](https://github.com/syamaner/coffee-first-crack-detection/issues/1)
**Status**: ✅ Complete — all 14 stories done
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

---

## Active Context

**All stories complete.** 14/14 ✅

**PRs merged**:
- PR #16: train.py, evaluate.py, inference.py
- PR #17: export_onnx.py, benchmark_platforms.py, push_to_hub.py, tests
- PR #18: input_values key fix, accelerate dep, build backend fix, training validated
- PR #19: docs state update — 13/14 complete
- S13 delivered via quickstart notebook update: notebooks/quickstart.ipynb

**Blockers**:
- 5 new mic-2 WAV recordings in `data/raw/` need Label Studio annotation before final dataset can be published to HuggingFace. This does not block the notebook.

---

## Latest Results

| Run | Accuracy | F1 | Recall (FC) | Notes |
|-----|----------|----|-------------|-------|
| Original prototype | ~93% | ~0.93 | 100% | coffee-roasting monorepo, custom Trainer |
| baseline_v1 (mic-1 only, MPS) | 95.6% (val) / 91.1% (test) | 0.955 / 0.913 | 95.5% | PR #18, 208 train samples |

---

## Decisions & Notes
- Using HF Trainer API (not custom Trainer) — enables push_to_hub=True and ecosystem compatibility
- save_pretrained / from_pretrained replaces .pt checkpoints
- Single epic issue (#1) with story checklist; one PR per story
- data/ is .gitignored — large audio files go to HuggingFace Datasets only
- RPi5 is inference-only via ONNX Runtime (no training)
- AGENTS.md is the single project rules file (no WARP.md)
