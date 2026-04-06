---
name: sync-hub-cards
description: Sync model card, dataset card, and Gradio Space files to HuggingFace Hub. Use when asked to update cards, sync to Hub, update the Space, or push card changes to HuggingFace.
---

## Sync Cards & Space to HuggingFace Hub

Uploads only card and UI files — does not re-push model weights, dataset audio, or ONNX artifacts. For full artifact uploads use `/push-to-hub`.

### Prerequisites
- HuggingFace CLI logged in: `huggingface-cli whoami`
- If not logged in: `huggingface-cli login`

### File mapping

| Local file | HF repo | Destination |
|---|---|---|
| `README.md` | `syamaner/coffee-first-crack-detection` (model) | `README.md` |
| `data/DATASET_CARD.md` | `syamaner/coffee-first-crack-audio` (dataset) | `README.md` |
| `spaces/README.md` | `syamaner/coffee-first-crack-detection` (space) | `README.md` |
| `spaces/app.py` | `syamaner/coffee-first-crack-detection` (space) | `app.py` |
| `spaces/requirements.txt` | `syamaner/coffee-first-crack-detection` (space) | `requirements.txt` |

### Steps

**1. Sync everything**
```bash
python scripts/sync_hub_cards.py
```

**2. Or sync selectively**
```bash
# Model card only
python scripts/sync_hub_cards.py --model-card

# Dataset card only
python scripts/sync_hub_cards.py --dataset-card

# Gradio Space only
python scripts/sync_hub_cards.py --space

# Combine flags
python scripts/sync_hub_cards.py --model-card --space
```

**3. Verify**
- Model card: https://huggingface.co/syamaner/coffee-first-crack-detection
- Dataset card: https://huggingface.co/datasets/syamaner/coffee-first-crack-audio
- Gradio Space: https://huggingface.co/spaces/syamaner/coffee-first-crack-detection

Check that cards render correctly and the Space rebuilds without errors.
