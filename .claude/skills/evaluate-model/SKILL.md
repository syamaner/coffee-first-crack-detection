---
name: evaluate-model
description: Evaluate a trained checkpoint on the test set and produce a full metrics report. Use when asked to evaluate, benchmark, or test model performance.
---

## Evaluate Model — Coffee First Crack Detection

### Steps

**1. Run evaluation**
```bash
python -m coffee_first_crack.evaluate \
  --model-dir experiments/baseline_v1/checkpoint-best \
  --test-dir data/splits/test \
  --output-dir experiments/baseline_v1/evaluation
```

Or evaluate a HuggingFace Hub checkpoint:
```bash
python -m coffee_first_crack.evaluate \
  --model-dir syamaner/coffee-first-crack-detection \
  --test-dir data/splits/test
```

**2. Check outputs**
Results saved to `--output-dir`:
- `test_results.txt` — accuracy, precision, recall, F1, ROC-AUC
- `classification_report.txt` — per-class breakdown
- `confusion_matrix.png` — visual confusion matrix

**3. Acceptance thresholds**
| Metric | Target |
|--------|--------|
| Accuracy | ≥ 0.90 |
| F1 (binary) | ≥ 0.90 |
| Recall (first_crack) | ≥ 0.95 ← most important |
| Precision (first_crack) | ≥ 0.85 |

**4. Update epic state**
After evaluation, update `docs/state/epics/coffee-first-crack-detection.md`:
- Fill in the **Latest Results** table with the new run's metrics
- Note checkpoint path and date

### Interpreting results
- **High recall, lower precision**: acceptable — better to detect early than miss first crack
- **Low recall on first_crack**: re-train with higher class weight or lower threshold
- **Confusion between classes**: check if annotation quality is consistent across both mics
