---
pipeline_tag: audio-classification
library_name: transformers
language: en
license: apache-2.0
widget:
  - src: https://huggingface.co/syamaner/coffee-first-crack-detection/resolve/main/audio_examples/first_crack_sample.wav
    example_title: "First crack (10s clip)"
  - src: https://huggingface.co/syamaner/coffee-first-crack-detection/resolve/main/audio_examples/no_first_crack_sample.wav
    example_title: "No first crack (10s clip)"
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
      value: 0.977
      name: Test Accuracy
    - type: f1
      value: 0.921
      name: Test F1 (macro)
    - type: precision
      value: 0.872
      name: Test Precision (first_crack)
    - type: recall
      value: 0.976
      name: Test Recall (first_crack)
    - type: roc_auc
      value: 0.997
      name: Test ROC-AUC
---

# Coffee First Crack Detection

An Audio Spectrogram Transformer (AST) fine-tuned to detect **first crack** — the critical moment during coffee roasting when beans begin to pop — from raw audio.

> **Source code & training**: [github.com/syamaner/coffee-first-crack-detection](https://github.com/syamaner/coffee-first-crack-detection)
>
> **How this was built:**
> - Part 1 — [The Architecture & The Agent — Spec-Driven ML Development with Warp/Oz](https://dev.to/syamaner/part-1-the-architecture-the-agent-spec-driven-ml-development-with-warpoz-3al6)
>
> **Original prototype:**
> - Part 1 — [Training a Neural Network to Detect Coffee First Crack from Audio](https://dev.to/syamaner/part-1-training-a-neural-network-to-detect-coffee-first-crack-from-audio-an-agentic-development-1jei)

---

## Model Description

Fine-tuned from [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) with partial backbone freeze (~14M trainable / 72M frozen) for binary audio classification:

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

ONNX models are published on HuggingFace Hub under `onnx/fp32/` and `onnx/int8/`:
- **INT8 (recommended)**: `onnx/int8/model_quantized.onnx` — 90MB, 2x faster, zero quality loss
- **FP32**: `onnx/fp32/model.onnx` — 345MB

```python
import onnxruntime as rt
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import ASTFeatureExtractor
import librosa

# Download INT8 model and preprocessor config from HF Hub
model_path = hf_hub_download(
    repo_id="syamaner/coffee-first-crack-detection",
    filename="onnx/int8/model_quantized.onnx",
)
extractor = ASTFeatureExtractor.from_pretrained(
    "syamaner/coffee-first-crack-detection", subfolder="onnx/int8",
)

# Thread-limit for RPi5 (2 threads — leaves cores free for MCP server + UI)
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1

sess = rt.InferenceSession(
    model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
)

audio, _ = librosa.load("roast.wav", sr=16000, mono=True)
inputs = extractor([audio.tolist()], sampling_rate=16000, return_tensors="np")
logits = sess.run(None, {sess.get_inputs()[0].name: inputs["input_values"]})[0]

# Softmax for probabilities
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
label_id = int(np.argmax(probs))
print(f"first_crack prob: {probs[0, 1]:.3f}")
```

> **Note**: RPi5 requires adequate PSU (5V/5A recommended) and active cooling.
> Default is 2 ONNX threads to leave CPU headroom for MCP server and agent UI.
> See [Hardware Requirements](#hardware-requirements).

---

## Training

| Parameter | Value |
|-----------|-------|
| Base model | MIT/ast-finetuned-audioset-10-10-0.4593 |
| Freeze strategy | Last 2 transformer layers + layernorm unfrozen (~14M / 86M params) |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Epochs | 5 (early stop patience=3, best epoch 2) |
| Weight decay | 0.1 |
| Loss | Class-weighted CrossEntropyLoss |
| Augmentation | Random amplitude scaling (±30%) + Gaussian noise injection |
| Crop mode (train) | Random |
| Crop mode (eval) | Center |

Training hardware: Apple M3+ Mac (MPS). Dataset: 1,435 fixed 10s chunks from 21 recordings (9 legacy mic-1 + 6 amplified mic-2 + 3 mic-1-panama + 3 mic-2-panama).

---

## Evaluation

**baseline_v5** — 21 recordings (9 legacy + 6 amplified mic-2 + 3 mic-1-panama + 3 mic-2-panama), Apple M3 Mac (MPS).

| Metric | Test |
|--------|------|
| Accuracy | **97.7%** |
| F1 (macro) | **0.921** |
| Precision (`first_crack`) | 87.2% |
| Recall (`first_crack`) | **97.6%** |
| ROC-AUC | **0.997** |
| Confusion matrix | 1 FN, 6 FP (303 test samples) |

Full dataset: 1,435 × 10s chunks (fixed sliding window), 922 / 210 / 303 train / val / test split (recording-level, no data leakage).

**Full-file detection on test recordings** (sliding window, threshold=0.6, min_pops=5):

| Recording | Mic | Ground Truth | Detected | Delta |
|-----------|-----|-------------|----------|-------|
| mic1-panama-roast2 | mic-1 | 13:09 | 13:03 | **-6s** |
| mic2-brazil-roast3-amplified | mic-2 | 10:39 | 10:33 | **-6s** |
| mic2-panama-roast1 | mic-2 | 11:05 | 10:57 | **-8s** |
| roast-3-costarica-hermosa-hp-a | mic-1 | 07:19 | MISSED | — |

---

## Limitations

- Dataset of 1,435 chunks from 21 roasts — generalisation to very different roasters/environments is uncertain
- Trained on Costa Rica Hermosa HP, Brazil, Brazil Santos, and Panama Hortigal Estate origins — other origins may vary
- Mic gain variation affects detection — older uncalibrated mic-2 recordings required amplification
- Microphone quality matters: model trained on two different microphones (FIFINE K669B condenser, Audio-Technica ATR2100x dynamic)
- No second crack detection — model is binary only
- Not validated in commercial roasting environments
- AST model (87M params) is too large for real-time (<500ms) inference on Raspberry Pi 5 — achieves ~2.07s per 10s window with INT8 quantization at 4 threads (with fan)

---

## Hardware Requirements

| Platform | Inference | Latency (10s window) | Model Size | Notes |
|----------|-----------|---------------------|------------|-------|
| Apple M3+ Mac | PyTorch (MPS) | ~100ms | 345MB | Auto-detected device |
| Apple M3+ Mac | ONNX Runtime (CPU) | ~197ms (INT8) / ~375ms (FP32) | 90MB / 345MB | No GPU needed |
| NVIDIA RTX 4090 | PyTorch (CUDA) | ~30ms | 345MB | fp16/bf16, num_workers=4 |
| Raspberry Pi 5 (16GB) | ONNX Runtime (CPU) | ~2.45s (INT8, 2 threads) | 90MB | ⭐ Recommended Pi config |

### Raspberry Pi 5 Notes

- Use `model_quantized.onnx` (INT8, 90MB) — 2x faster than FP32 with zero quality loss
- **Recommended config**: INT8, 2 threads, adequate PSU + active cooler → **p50 = 2,452ms**
- **Why 2 threads**: the Pi also runs an MCP server and agent UI — 2 ONNX threads leaves 2 cores free for those services
- **Detection threshold**: 0.90 (precision=0.952, recall=0.909, F1=0.930) — minimises false positives
- **Power**: adequate PSU (5V/5A recommended) required for multi-thread. Standard chargers (5V/3A) cause under-voltage crashes under load
- **Cooling**: active cooler recommended — sustained 2-thread load without fan reaches 77°C+ and triggers thermal throttling
- **Threads**: 2 threads with fan (2,452ms), 4 threads with fan (2,070ms), 1 thread on any PSU (4,441ms)
- **Latency target**: current AST model (87M params) does not meet the <500ms target on RPi5. Consider a lighter model for real-time edge use
- Install: `pip install -r requirements-pi.txt` then `pip install torch --index-url https://download.pytorch.org/whl/cpu`

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
