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
model-index:
- name: coffee-first-crack-detection
  results:
  - task:
      type: audio-classification
      name: Audio Classification
    dataset:
      name: Coffee First Crack Audio
      type: syamaner/coffee-first-crack-audio
      split: test
    metrics:
    - type: accuracy
      value: 0.911
      name: Test Accuracy
    - type: f1
      value: 0.913
      name: Test F1 (macro)
    - type: recall
      value: 0.955
      name: Test Recall (first_crack)
    - type: roc_auc
      value: 0.978
      name: Test ROC-AUC
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

# Thread-limit for power-constrained devices (RPi5 with standard PSU)
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 2  # default: all cores
sess_options.inter_op_num_threads = 1

extractor = ASTFeatureExtractor.from_pretrained("syamaner/coffee-first-crack-detection")
sess = rt.InferenceSession("model_quantized.onnx", sess_options=sess_options,
                           providers=["CPUExecutionProvider"])

audio, _ = librosa.load("roast.wav", sr=16000, mono=True)
inputs = extractor(audio.tolist(), sampling_rate=16000, return_tensors="np")
logits = sess.run(None, {sess.get_inputs()[0].name: inputs["input_values"]})[0]

# Softmax for probabilities
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
label_id = int(np.argmax(probs))
print(f"first_crack prob: {probs[0, 1]:.3f}")
```

> **Note**: RPi5 requires the official 27W (5V/5A) USB-C PSU and active cooling
> for stable multi-thread inference. See [Hardware Requirements](#hardware-requirements).

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

**baseline_v1** — mic-1 recordings only (Costa Rica Hermosa HP + Brazil), Apple M3 Mac (MPS).

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | 95.6% | 91.1% |
| F1 (macro) | 0.955 | 0.913 |
| Recall (`first_crack`) | — | 95.5% |
| ROC-AUC | — | 0.978 |
| Dataset split | 45 samples | 45 samples |

Full dataset: 298 × 10 s chunks, 208 / 45 / 45 train / val / test split.

---

## Limitations

- Small dataset (~300 chunks from 6 roasts) — generalisation to very different roasters/environments is uncertain
- Trained primarily on Costa Rica Hermosa HP and Brazil origins — other origins may vary
- Microphone quality matters: model trained on two different microphones (mic-1-original, mic-2-new)
- No second crack detection — model is binary only
- Not validated in commercial roasting environments
- AST model (87M params) is too large for real-time (<500ms) inference on Raspberry Pi 5 — achieves ~2.4s per 10s window with INT8 quantization at 2 threads

---

## Hardware Requirements

| Platform | Inference | Latency (10s window) | Notes |
|----------|-----------|---------------------|-------|
| Apple M3+ Mac | PyTorch (MPS) | ~100ms | Auto-detected device |
| Apple M3+ Mac | ONNX Runtime (CPU) | ~197ms (INT8) / ~375ms (FP32) | No GPU needed |
| NVIDIA RTX 4090 | PyTorch (CUDA) | ~30ms | fp16/bf16, num_workers=4 |
| Raspberry Pi 5 (16GB) | ONNX Runtime (CPU) | ~2.4s (INT8, 2 threads) | Requires 27W PSU + active cooler |

### Raspberry Pi 5 Notes

- Use `model_quantized.onnx` (INT8, 90MB) — 2x faster than FP32 with zero quality loss
- **Power**: requires official RPi5 27W (5V/5A) USB-C PSU. Standard chargers (5V/3A) cause under-voltage crashes under load
- **Cooling**: active cooler mandatory — sustained inference reaches 77°C+ and triggers thermal throttling
- **Threads**: limit ONNX Runtime to 2 threads (`--threads 2` or `sess_options.intra_op_num_threads = 2`) to balance speed vs power/thermal
- **Latency target**: current AST model (87M params) does not meet the <500ms target on RPi5. Consider a lighter model for real-time edge use
- Install: `pip install -r requirements-pi.txt` (also needs `torch` CPU-only for feature extraction)

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
