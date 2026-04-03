---
name: train-model
description: End-to-end training of the coffee first crack detection model. Use when asked to train, fine-tune, or re-train the model.
---

## Train Model — Coffee First Crack Detection

### Prerequisites
1. `data/splits/` must exist with `train/`, `val/`, `test/` subdirectories — run data prep first if not
2. Virtual environment active: `source venv/bin/activate`
3. Check device: `python -c "from coffee_first_crack.utils.device import get_device; print(get_device())"`

### Steps

**1. Verify data splits exist**
```bash
ls data/splits/train/first_crack/ | wc -l
ls data/splits/train/no_first_crack/ | wc -l
```
Expect ~145+ files per class in train. If missing, run the data prep pipeline first.

**2. Review config**
Check `configs/default.yaml` — key values:
- `training.batch_size`: 8 (increase to 16–32 on CUDA)
- `training.learning_rate`: 5e-5
- `training.num_epochs`: 20
- `training.fp16`: set `true` for CUDA, leave `false` for MPS

**3. Run training**
```bash
python -m coffee_first_crack.train \
  --data-dir data/splits \
  --experiment-name baseline_v1
```

For CUDA with mixed precision:
```bash
python -m coffee_first_crack.train \
  --data-dir data/splits \
  --experiment-name baseline_v1 \
  --fp16
```

**4. Monitor**
```bash
tensorboard --logdir experiments/baseline_v1/logs
```

**5. Validate results**
Target metrics on validation set:
- Accuracy ≥ 0.90
- F1 ≥ 0.90
- First-crack recall ≥ 0.95 (safety critical — false negatives are worse than false positives)

**6. On completion**
- Best checkpoint saved to `experiments/{name}/checkpoint-best/`
- Run evaluation: `/evaluate-model`
- If metrics pass, optionally push to Hub: `/push-to-hub`

### Troubleshooting
- `MPS backend out of memory` → reduce batch_size to 4
- `num_workers` error on MPS → ensure `num_workers=0` in config
- Loss not decreasing → check class weights are applied (should print at start)
