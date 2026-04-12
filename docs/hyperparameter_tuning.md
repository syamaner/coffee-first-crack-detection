# Hyperparameter Tuning

History of training experiments, what broke, what worked, and why.

---

## The Core Problem

The Audio Spectrogram Transformer (AST) has ~86 million parameters. Our dataset has ~900 training chunks (~136 positive). This ratio guarantees overfitting unless carefully managed.

---

## Experiment History

### baseline_v1 — Naive Full Fine-Tuning (15 recordings, mic-1 only)
- **Config**: lr=5e-5, weight_decay=0.01, all 86M params trainable
- **Result**: 91.1% test acc, 0.913 F1, 95.5% recall
- **Issue**: Small dataset (298 chunks from 6 roasts) masked the overfitting — model happened to generalise because the test set was too similar to training data

### baseline_v2 — Tuned LR on Expanded Dataset (15 recordings)
- **Config**: lr=2e-5, weight_decay=0.05, early_stopping=3, all 86M params trainable
- **Result**: 97.4% test acc, 0.925 F1, 86.1% recall, 100% precision
- **Issue**: Zero false positives but missed 14% of first cracks. Training loss dropped to near-zero by epoch 3 — textbook memorisation. The early stopping prevented catastrophic degradation but the model was already overfitting by the best epoch.

### baseline_v3 — Full Fine-Tuning on 21 Recordings (with amplified mic2)
- **Config**: lr=2e-5, weight_decay=0.05, early_stopping=3, all 86M params trainable
- **Result**: 96.0% test acc, 0.872 F1, 93.2% recall, 82.0% precision
- **Issue**: Severe overfitting. Training loss collapsed to 0.0 by epoch 3 while validation loss spiked. More data helped recall (93% vs 86%) but precision dropped (82% vs 100%) — the model learned to say "yes" more often but less accurately. 9 false positives on the test set.

### baseline_v4 — Fully Frozen Backbone (3K trainable params)
- **Config**: lr=5e-6, weight_decay=0.1, entire AST backbone frozen, only classifier head trainable (3,074 params)
- **Result**: Training interrupted at epoch 7. Best val F1=0.494, recall=0.489
- **Issue**: Severe underfitting. AudioSet features don't map directly to coffee-crack sounds — the frozen backbone produced representations that a simple linear head couldn't separate. The learning rate (5e-6) was also far too low for training a freshly initialised head from scratch.

### baseline_v5 — Partial Freeze + Augmentation (current best)
- **Config**: lr=5e-5, weight_decay=0.1, warmup_steps=250, early_stopping=3
- **Freeze strategy**: Freeze all AST layers, then unfreeze last 2 transformer layers + final layernorm (~14M trainable / 72M frozen)
- **Augmentation**: Random amplitude scaling (±30%, 50% probability) + Gaussian noise (amp 0.001–0.005, 50% probability)
- **Result**: 97.7% test acc, 0.921 F1, 97.6% recall, 87.2% precision, 0.997 ROC-AUC
- **Full-file detection**: 3/4 test recordings detected within 8s of ground truth
- **Why it works**: The frozen early layers preserve general audio feature extraction. The unfrozen last 2 layers adapt high-level representations to the coffee-crack domain. Augmentation prevents amplitude memorisation — critical because mic gain varies across recordings.

### baseline_v6 — More Aggressive Regularisation (reverted)
- **Config**: lr=5e-5, weight_decay=0.15 (up from 0.1), warmup_steps=50 (down from 250), early_stopping=3
- **Same freeze + augmentation as v5**
- **Result**: 97.4% test acc, 0.907 F1, 92.9% recall, 88.6% precision, 0.996 ROC-AUC
- **Issue**: Over-regularised. Val F1 plateaued at 0.831 from epoch 1 through 5 — the model converged to predicting the same 32/45 positives and never improved. Recall dropped 4.7% (97.6% → 92.9%) with 3 false negatives instead of 1. Precision improved marginally (+1.4%) but not enough to justify the recall loss.
- **Lesson**: weight_decay=0.15 + short warmup penalises the unfrozen layers too aggressively, preventing them from adapting to the domain. The v5 values (wd=0.1, warmup=250) give the last 2 layers enough room to learn coffee-crack features without memorising.

---

## Key Lessons

### 1. Freeze Strategy Matters More Than Learning Rate
- Full fine-tuning (86M params) on <1000 samples: always overfits, regardless of LR
- Fully frozen backbone (3K params): underfits — AudioSet features aren't close enough to coffee sounds
- Partial freeze (14M params): the sweet spot — enough capacity to adapt, not enough to memorise

### 2. Learning Rate Depends on What You're Training
- Full 86M model: 2e-5 works (standard for transformer fine-tuning)
- Frozen backbone, head only (3K params): needs 1e-3 to 5e-4 (training a fresh linear layer)
- Partial freeze (14M params): 5e-5 works — same order as full fine-tuning since the unfrozen layers are pre-trained, not random

### 3. Augmentation Solves Real Hardware Problems
- ±30% amplitude scaling simulates mic gain variation across sessions
- Without it, the model memorises the exact amplitude profile of each recording
- This directly addresses the mic2 detection failures — older mic2 recordings had ~12 dB lower gain than calibrated sessions
- Gaussian noise injection prevents overfitting to the specific background noise of each recording environment

### 4. Recall vs Precision Tradeoff
- v2 had 100% precision / 86% recall — never false-alarmed but missed 14% of cracks
- v5 has 87% precision / 98% recall — occasionally false-alarms but catches nearly every crack
- For a coffee roasting assistant, **recall is more important** — missing first crack ruins the roast, a false alarm just gets ignored
- The 6 false positives in v5 are acceptable; the 1 false negative is the concern

### 5. Early Stopping Patience
- patience=3 consistently works — the model peaks at epoch 2-3 and degrades after
- Lower patience (1-2) risks stopping before the model converges
- Higher patience (5+) wastes compute and risks selecting a worse checkpoint

---

## Current Config (configs/default.yaml)

```yaml
training:
  batch_size: 8
  learning_rate: 5.0e-5
  num_epochs: 20
  warmup_steps: 250
  weight_decay: 0.1
  max_grad_norm: 1.0
  early_stopping_patience: 3
  metric_for_best_model: f1
```

Note: warmup=50 / weight_decay=0.15 was tested in baseline_v6 — it over-regularised, dropping recall from 97.6% to 92.9%. Reverted to v5 values.

Freeze strategy and augmentation are implemented in `src/coffee_first_crack/train.py` (not configurable via YAML — these are architectural decisions, not hyperparameters to sweep).

---

## What to Try Next

1. **More data** — the single highest-impact improvement. Each new roast recording adds ~70 chunks. Aim for 40+ recordings to get >500 positive samples.
2. **SpecAugment** — mask random frequency bands and time steps in the mel spectrogram. Standard technique for audio models, would complement the waveform-level augmentation already in place.
3. **Unfreeze more layers** — if dataset grows past ~2000 chunks, try unfreezing the last 4 layers instead of 2.
4. **Threshold tuning** — the full-file inference uses threshold=0.6 and min_pops=5. These could be optimised per-mic or per-deployment (e.g. RPi5 uses threshold=0.90 already).
