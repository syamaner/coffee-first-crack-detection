---
name: push-to-hub
description: Publish the trained model and dataset to HuggingFace Hub. Use when asked to publish, upload, release, or push to HuggingFace.
---

## Push to HuggingFace Hub — Coffee First Crack Detection

### Prerequisites
- Model trained and evaluated (metrics pass thresholds)
- HuggingFace CLI logged in: `huggingface-cli whoami`
- If not logged in: `huggingface-cli login` (use token from https://huggingface.co/settings/tokens)

### Steps

**1. Verify login**
```bash
huggingface-cli whoami
# expect: syamaner
```

**2. Push model**
```bash
python scripts/push_to_hub.py \
  --model-dir experiments/baseline_v1/checkpoint-best \
  --repo-id syamaner/coffee-first-crack-detection
```

This calls `model.push_to_hub()` and `feature_extractor.push_to_hub()`, uploading:
- `model.safetensors`
- `config.json`
- `preprocessor_config.json`
- `README.md` (model card)

**3. Push dataset**
```bash
python scripts/push_to_hub.py \
  --dataset-dir data/splits \
  --recordings-csv data/recordings.csv \
  --dataset-repo-id syamaner/coffee-first-crack-audio
```

**4. Upload ONNX artifacts**
```bash
python scripts/push_to_hub.py \
  --onnx-dir exports/onnx \
  --repo-id syamaner/coffee-first-crack-detection
```

**5. Verify on Hub**
- Model: https://huggingface.co/syamaner/coffee-first-crack-detection
- Dataset: https://huggingface.co/datasets/syamaner/coffee-first-crack-audio

Check that:
- Model card renders correctly (YAML frontmatter is valid)
- `from_pretrained('syamaner/coffee-first-crack-detection')` works in a clean environment
- Dataset has correct splits and metadata columns

**6. Update epic state**
After successful publish:
- Update `docs/state/epics/coffee-first-crack-detection.md` — mark S11/S12 done
- Close GitHub issues #12 and #13
- Update epic #1 checklist
- Add HuggingFace model URL to `docs/state/registry.md`

### Rollback
If you need to delete and re-push:
```bash
huggingface-cli repo delete syamaner/coffee-first-crack-detection --type model
huggingface-cli repo create coffee-first-crack-detection --type model
```
