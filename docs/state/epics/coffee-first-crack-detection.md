# Epic: Coffee First Crack Detection — HuggingFace Model Repository

**GitHub Issue**: [#1](https://github.com/syamaner/coffee-first-crack-detection/issues/1)
**Status**: 🟡 In Progress — Phase 1 scaffolding underway
**Last Updated**: 2026-04-03

## Objective
Create a standalone, HuggingFace-publishable repository for training, evaluating, and publishing the coffee first crack audio detection model. Extracted from the `coffee-roasting` monorepo. Targets M3+ Mac (MPS), RTX 4090 (CUDA), and Raspberry Pi 5 (ONNX/CPU).

---

## Status Map

### Phase 1 — Scaffold & Data
- [x] S1 [#2](https://github.com/syamaner/coffee-first-crack-detection/issues/2): Scaffold repo structure, pyproject.toml, configs, AGENTS.md, skills
  - GitHub repo created, directories scaffolded
  - pyproject.toml, requirements.txt, requirements-pi.txt, configs/default.yaml ✅
  - AGENTS.md, .claude/skills/ — pending
- [x] S2 [#3](https://github.com/syamaner/coffee-first-crack-detection/issues/3): Port dataset.py with HF Datasets integration and filename metadata
  - `src/coffee_first_crack/dataset.py` created ✅
  - Filename parser (new convention + legacy), recordings.csv generation, create_dataloaders()
- [x] S3 [#4](https://github.com/syamaner/coffee-first-crack-detection/issues/4): Port model.py — HF-native ASTForAudioClassification wrapper
  - `src/coffee_first_crack/model.py` created ✅
  - build_model(), build_feature_extractor(), FirstCrackClassifier
- [x] S4 [#5](https://github.com/syamaner/coffee-first-crack-detection/issues/5): Port utils/metrics.py and utils/device.py
  - `src/coffee_first_crack/utils/metrics.py` and `utils/device.py` created ✅

### Phase 2 — Training & Evaluation
- [ ] S5 [#6](https://github.com/syamaner/coffee-first-crack-detection/issues/6): Implement train.py with HF Trainer API (class-weighted loss)
- [ ] S6 [#7](https://github.com/syamaner/coffee-first-crack-detection/issues/7): Implement evaluate.py with full metrics report
- [ ] S7 [#8](https://github.com/syamaner/coffee-first-crack-detection/issues/8): Validate training — run locally, verify ~93% accuracy

### Phase 3 — Inference & Export
- [ ] S8 [#9](https://github.com/syamaner/coffee-first-crack-detection/issues/9): Port inference.py — sliding window + streaming FirstCrackDetector
- [ ] S9 [#10](https://github.com/syamaner/coffee-first-crack-detection/issues/10): Implement export_onnx.py (FP32 + INT8 quantized)
- [ ] S10 [#11](https://github.com/syamaner/coffee-first-crack-detection/issues/11): Create benchmark_platforms.py

### Phase 4 — Publishing
- [ ] S11 [#12](https://github.com/syamaner/coffee-first-crack-detection/issues/12): Implement scripts/push_to_hub.py
- [ ] S12 [#13](https://github.com/syamaner/coffee-first-crack-detection/issues/13): Write HuggingFace model card README.md
- [ ] S13 [#14](https://github.com/syamaner/coffee-first-crack-detection/issues/14): Create notebooks/quickstart.ipynb
- [ ] S14 [#15](https://github.com/syamaner/coffee-first-crack-detection/issues/15): Write pytest test suite

---

## Active Context

**Current work**: Completing Phase 1 scaffolding — remaining items are AGENTS.md, .claude/skills/, then committing the initial scaffold + opening PR for S1.

**Files created so far**:
```
src/coffee_first_crack/__init__.py
src/coffee_first_crack/model.py
src/coffee_first_crack/dataset.py
src/coffee_first_crack/utils/__init__.py
src/coffee_first_crack/utils/device.py
src/coffee_first_crack/utils/metrics.py
configs/default.yaml
pyproject.toml
requirements.txt
requirements-pi.txt
docs/state/registry.md
docs/state/epics/coffee-first-crack-detection.md
```

**Blockers**:
- New mic-2 recordings (at-roast1..5.aup3) need Audacity WAV export before they can be annotated and added to the dataset. This is user-side work that unblocks Phase 4 dataset publishing.

---

## Latest Results

| Run | Accuracy | F1 | Recall (FC) | Notes |
|-----|----------|----|-------------|-------|
| Original prototype | ~93% | ~0.93 | 100% | coffee-roasting monorepo, custom Trainer |
| New repo baseline | — | — | — | Not yet run |

---

## Decisions & Notes
- Using HF Trainer API (not custom Trainer) — enables push_to_hub=True and ecosystem compatibility
- save_pretrained / from_pretrained replaces .pt checkpoints
- Single epic issue (#1) with story checklist; one PR per story
- data/ is .gitignored — large audio files go to HuggingFace Datasets only
- RPi5 is inference-only via ONNX Runtime (no training)
- AGENTS.md is the single project rules file (no WARP.md)
