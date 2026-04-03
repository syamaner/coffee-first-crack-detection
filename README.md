---
pipeline_tag: audio-classification
library_name: transformers
language: en
license: apache-2.0
tags:
  - audio
  - coffee
  - first-crack-detection
  - ast
  - audio-spectrogram-transformer
datasets:
  - syamaner/coffee-first-crack-audio
base_model: MIT/ast-finetuned-audioset-10-10-0.4593
metrics:
  - accuracy
  - f1
  - precision
  - recall
---

# Coffee First Crack Detection

An Audio Spectrogram Transformer (AST) fine-tuned to detect **first crack** — the critical moment during coffee roasting when beans begin to pop — from raw audio.

> **Source code & training**: [github.com/syamaner/coffee-first-crack-detection](https://github.com/syamaner/coffee-first-crack-detection)

---

## Model Description

Fine-tuned from [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (~86M parameters) for binary audio classification:

| Label | ID | Description |
|-------|----|-------------|
| `no_first_crack` | 0 | Background roast noise, no cracking |
| `first_crack` | 1 | First crack popping/cracking sounds |

**Feature extractor**: `ASTFeatureExtractor` at 16 kHz mono, 128 mel bins, max_length=1024, mean=-4.2677, std=4.5689 (AudioSet calibration).

---

## Intended Use

- **Roasting automation**: trigger events (reduce heat, start timer) when first crack is detected
- **Roast logging**: timestamp first crack onset for reproducibility
- **MCP server integration**: embedded in the `coffee-roasting` roaster control system

**Not intended for**: other food processing sounds, non-coffee audio, commercial food safety systems.

---

## How to Use

### Python (transformers)

```python
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import torch, librosa

model = ASTForAudioClassification.from_pretrained("syamaner/coffee-first-crack-detection")
extractor = ASTFeatureExtractor.from_pretrained("syamaner/coffee-first-crack-detection")
model.eval()

audio, _ = librosa.load("roast.wav", sr=16000, mono=True)
inputs = extractor(audio.tolist(), sampling_rate=16000, return_tensors="pt")

with torch.inference_mode():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)

label = model.config.id2label[probs.argmax().item()]
print(f"{label}: {probs.max().item():.3f}")
```

### Sliding window (long audio files)

```python
from coffee_first_crack.inference import SlidingWindowInference

detector = SlidingWindowInference(
    model_name_or_path="syamaner/coffee-first-crack-detection",
    window_size=10.0,
    overlap=0.7,
    threshold=0.6,
    min_pops=5,
)
events = detector.process_file("roast.wav")
for event in events:
    print(f"First crack at {event.timestamp_str}")
```

### Live microphone

```python
from coffee_first_crack.inference import FirstCrackDetector

detector = FirstCrackDetector(
    use_microphone=True,
    model_name_or_path="syamaner/coffee-first-crack-detection",
)
detector.start()
# poll detector.is_first_crack() in your roast loop
```

### ONNX (Raspberry Pi 5)

```python
import onnxruntime as rt
import numpy as np
from transformers import ASTFeatureExtractor
import librosa

extractor = ASTFeatureExtractor.from_pretrained("syamaner/coffee-first-crack-detection")
sess = rt.InferenceSession("model_quantized.onnx")

audio, _ = librosa.load("roast.wav", sr=16000, mono=True)
inputs = extractor(audio.tolist(), sampling_rate=16000, return_tensors="np")
logits = sess.run(None, {"input_features": inputs["input_features"]})[0]
label_id = logits.argmax()
```

---

## Training

| Parameter | Value |
|-----------|-------|
| Base model | MIT/ast-finetuned-audioset-10-10-0.4593 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Epochs | up to 20 (early stop on val F1) |
| Loss | Class-weighted CrossEntropyLoss |
| Crop mode (train) | Random |
| Crop mode (eval) | Center |

Training hardware: Apple M3+ Mac (MPS).

---

## Evaluation

> Results will be updated after retraining on the expanded dataset (mic-1 + mic-2).

| Metric | Prototype (mic-1 only) |
|--------|----------------------|
| Accuracy | ~93% |
| F1 | ~0.93 |
| Recall (first_crack) | 100% |
| Dataset | 298 chunks, 208/45/45 train/val/test |

---

## Limitations

- Small dataset (~300 chunks from 6 roasts) — generalisation to very different roasters/environments is uncertain
- Trained primarily on Costa Rica Hermosa HP and Brazil origins — other origins may vary
- Microphone quality matters: model trained on two different microphones (mic-1-original, mic-2-new)
- No second crack detection — model is binary only
- Not validated in commercial roasting environments

---

## Hardware Requirements

| Platform | Inference | Notes |
|----------|-----------|-------|
| Apple M3+ Mac | PyTorch (MPS) | < 100ms / 10s window |
| NVIDIA RTX 4090 | PyTorch (CUDA) | < 30ms / 10s window |
| Raspberry Pi 5 (16GB) | ONNX Runtime (CPU) | < 500ms / 10s window, use `model_quantized.onnx` |

---

## Dataset

Training data: [`syamaner/coffee-first-crack-audio`](https://huggingface.co/datasets/syamaner/coffee-first-crack-audio)

10-second WAV chunks at 16 kHz mono, labelled `first_crack` / `no_first_crack`. Includes per-sample metadata: microphone, coffee origin, annotation source.

---

## Citation

```bibtex
@misc{yamaner2025coffeefc,
  author = {Yamaner, Sertan},
  title  = {Coffee First Crack Detection},
  year   = {2025},
  url    = {https://huggingface.co/syamaner/coffee-first-crack-detection}
}
```
