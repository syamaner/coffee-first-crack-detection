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
      value: 0.980
      name: Test Accuracy
    - type: f1
      value: 0.932
      name: Test F1 (macro)
    - type: precision
      value: 0.891
      name: Test Precision (first_crack)
    - type: recall
      value: 0.976
      name: Test Recall (first_crack)
    - type: roc_auc
      value: 0.998
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

### ONNX (Raspberry Pi 5) — torch-free

ONNX models are published on HuggingFace Hub under `onnx/fp32/` and `onnx/int8/`:
- **INT8 (recommended)**: `onnx/int8/model_quantized.onnx` — 90MB, 2x faster, no measurable quality loss. Re-benchmarked on baseline_v5's 303-sample test set (12 Jul 2026): 91.1% precision / 97.6% recall / F1 0.943 on `first_crack`, marginally ahead of fp32 and with no quality loss (see [Evaluation](#evaluation)). Reproduce with `python scripts/evaluate_onnx.py --onnx-dir exports/onnx/int8 --test-dir data/splits/test`.
- **FP32**: `onnx/fp32/model.onnx` — 345MB

Pi/ONNX inference uses `coffee_first_crack.inference_onnx.OnnxSlidingWindowInference`, which
loads the ONNX model and a `MelFrontend` (a hand-written numpy/scipy Kaldi-compatible mel
filterbank, added in D27) directly from HuggingFace Hub — no `torch` or `transformers`
required:

```python
from coffee_first_crack.inference_onnx import OnnxSlidingWindowInference

# Loads the INT8 model + MelFrontend feature extractor from HF Hub
inference = OnnxSlidingWindowInference(profile="pi_inference")
events = inference.process_file("roast.wav")
for event in events:
    print(f"First crack at {event.timestamp_str}")
```

`MelFrontend` reproduces `ASTFeatureExtractor`'s Kaldi fbank computation using only `numpy`
and `scipy.signal` — it is **not** `librosa`-based. `librosa`'s mel filterbank cannot
reproduce Kaldi-style features (a different filter construction entirely), so it is used
only for audio I/O (`librosa.load`) on this path, never for feature extraction.

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

> **Provenance note (12 Jul, #55):** the table below was re-measured from a clean run against the
> checkpoint and test split currently on disk, after two different FP counts had been reported
> from earlier runs ("1 FN, 6 FP" here vs "4 FP" in issue #55). Root cause: the committed
> `experiments/baseline_v5/evaluation/test_results.json` predates a regeneration of
> `data/splits/test/` from later the same session (checkpoint `model.safetensors` written
> 21:26, that eval run 21:43, test-split WAVs rewritten 21:56) — it was scored against a
> since-replaced version of the split, not the current one. The "4 FP" figure in #55 is real but
> is the **ONNX INT8** export's number, not the PyTorch/AST checkpoint's — two different,
> legitimate configurations, not a bug. Both rows below were reproduced twice (deterministic,
> `crop_mode="center"`) against `data/splits/test/` (303 samples, 42 `first_crack` / 261
> `no_first_crack`) as of this checkpoint.

| Metric | AST / PyTorch (fp32) | ONNX INT8 |
|--------|----------------------|-----------|
| Accuracy | **98.0%** | **98.3%** |
| F1 (macro) | **0.932** | **0.943** |
| Precision (`first_crack`) | 89.1% | 91.1% |
| Recall (`first_crack`) | **97.6%** | **97.6%** |
| ROC-AUC | **0.9979** | **0.9976** |
| Confusion matrix | 1 FN, 5 FP | 1 FN, 4 FP |

ROC-AUC shown to 4 decimals rather than the usual 3 — rounding to 3 makes both columns read
"0.998" and visually erases INT8's (tiny, expected) AUC dip relative to fp32.

The AST/PyTorch confusion matrix (256 TN / 5 FP / 1 FN / 41 TP) is not part of
`evaluate.py`'s JSON output (`MetricsCalculator.compute()` only returns scalar metrics) — it's
read from that run's own `test_results.txt` / `confusion_matrix.png`
(`experiments/baseline_v5/evaluation/`), not inferred from the ONNX fp32 run. See
`results/baseline_v5_303set/ast_fp32_eval.json` for the full JSON including this matrix.

Reproduce:
```bash
# AST / PyTorch checkpoint
python -m coffee_first_crack.evaluate \
  --model-dir experiments/baseline_v5/checkpoint-best --test-dir data/splits/test

# ONNX INT8 export
python scripts/evaluate_onnx.py \
  --onnx-dir exports/onnx/int8 --test-dir data/splits/test
```

Full dataset: 1,435 × 10s chunks (fixed sliding window), 922 / 210 / 303 train / val / test split (recording-level, no data leakage).

**Full-file detection on test recordings** (sliding window, threshold=0.6, min_pops=5):

| Recording | Mic | Ground Truth | Detected | Delta |
|-----------|-----|-------------|----------|-------|
| mic1-panama-roast2 | mic-1 | 13:09 | 13:03 | **-6s** |
| mic2-brazil-roast3-amplified | mic-2 | 10:39 | 10:33 | **-6s** |
| mic2-panama-roast1 | mic-2 | 11:05 | 10:57 | **-8s** |
| roast-3-costarica-hermosa-hp-a | mic-1 | 07:19 | MISSED | — |

---

## Retraining & updating metrics

When new recordings are added to the dataset, rebuild the splits (see
[docs/data_preparation.md](docs/data_preparation.md)), then retrain, evaluate,
export, and re-benchmark the ONNX export:

```bash
# 1. Train (reads data/splits, writes a checkpoint under experiments/)
python -m coffee_first_crack.train --data-dir data/splits --experiment-name baseline_vN

# 2. Evaluate the PyTorch/AST checkpoint on the test split
python -m coffee_first_crack.evaluate \
  --model-dir experiments/baseline_vN/checkpoint-best \
  --test-dir data/splits/test \
  --output-dir experiments/baseline_vN/evaluation

# 3. Export to ONNX (INT8 quantized by default; --no-quantize to skip)
python -m coffee_first_crack.export_onnx \
  --model-dir experiments/baseline_vN/checkpoint-best \
  --output-dir exports/onnx --quantize

# 4. Re-benchmark the ONNX INT8 export on the SAME test split
python scripts/evaluate_onnx.py \
  --onnx-dir exports/onnx/int8 \
  --test-dir data/splits/test \
  --output results/baseline_vN_int8_eval.json
```

The `train-first-crack`, `evaluate-first-crack`, and `export-onnx-first-crack`
console scripts (declared in `pyproject.toml`) are equivalent to the module
invocations above.

**After a retrain, update these docs so the numbers match reality:**

- the **Evaluation** table in this README (both fp32 and INT8 columns) and the
  INT8 note in the [ONNX](#onnx-raspberry-pi-5--torch-free) section;
- `data/DATASET_CARD.md` (split counts, source-recordings table, and the
  "Last updated" line) whenever the dataset itself changed.

**What is committed vs published vs ignored:**

- **Committed to git:** the evaluation artifacts under
  `experiments/<name>/evaluation/` (`test_results.txt` / `.json`,
  `confusion_matrix.png`) and any `results/*.json` you keep as provenance, plus
  the doc updates above. Note `data/raw/`, `data/processed/`, `data/splits/`,
  `experiments/`, `exports/`, and `*.onnx` are **gitignored** (large binaries) —
  only the small text/JSON eval outputs you explicitly `git add -f` are tracked.
- **Pushed to HuggingFace Hub:** the model weights and ONNX exports —
  `python -m coffee_first_crack.train ... --push-to-hub`, or `scripts/push_to_hub.py`.
- **Left gitignored (local only):** everything under `experiments/` and
  `exports/` except the small eval artifacts you force-add.

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
| Apple M3+ Mac | PyTorch (MPS) | ~56ms | 345MB | Auto-detected device |
| Apple M3+ Mac | ONNX Runtime (CPU) | ~216ms (INT8) / ~429ms (FP32) | 90MB / 345MB | No GPU needed |
| NVIDIA RTX 4090 | PyTorch (CUDA) | ~30ms | 345MB | fp16/bf16, num_workers=4 |
| Raspberry Pi 5 (16GB) | ONNX Runtime (CPU) | ~2.45s (INT8, 2 threads) | 90MB | ⭐ Recommended Pi config |

> Mac PyTorch/ONNX rows re-benchmarked 12 Jul on baseline_v5 (`scripts/benchmark_platforms.py
> --model-dir experiments/baseline_v5/checkpoint-best --onnx-dir exports/onnx --n-runs 30`,
> dummy 10s audio, p50 of 30 runs after 5 warmup). Absolute numbers will vary by machine/ONNX
> Runtime version — don't treat these as a hard SLA, only as the relative INT8-vs-FP32 shape.
> RTX 4090 and RPi5 rows are carried over from earlier hardware-specific validation, not
> re-run here.

### Raspberry Pi 5 Notes

- Use `model_quantized.onnx` (INT8, 90MB) — **on this machine, INT8 was ~2x faster than FP32
  AND scored marginally *better* on quality** (98.3% acc / 91.1% precision / 4 FP vs FP32's
  98.0% acc / 89.1% precision / 5 FP on the 303-sample test set, both against baseline_v5,
  12 Jul). That is a quantization-noise-sized difference on one 303-sample set, not a
  guarantee INT8 always matches or beats FP32 — but on the measured evidence there is no
  quality loss to trade against the latency win. Supersedes the earlier "no measurable
  quality loss" claim, which predated a re-benchmark on baseline_v5 (see `results/`
  eval JSONs for raw output; reproduce with `python scripts/evaluate_onnx.py --onnx-dir
  exports/onnx/{fp32,int8} --test-dir data/splits/test`).
- **Recommended config**: INT8, 2 threads, adequate PSU + active cooler → **p50 = 2,452ms**
- **Why 2 threads**: the Pi also runs an MCP server and agent UI — 2 ONNX threads leaves 2 cores free for those services
- **Detection threshold**: 0.90 (precision=0.952, recall=0.909, F1=0.930 — historical RPi5 threshold sweep on the earlier 45-sample test set, see `results/README.md`) — minimises false positives
- **Power**: adequate PSU (5V/5A recommended) required for multi-thread. Standard chargers (5V/3A) cause under-voltage crashes under load
- **Cooling**: active cooler recommended — sustained 2-thread load without fan reaches 77°C+ and triggers thermal throttling
- **Threads**: 2 threads with fan (2,452ms), 4 threads with fan (2,070ms), 1 thread on any PSU (4,441ms)
- **Latency target**: current AST model (87M params) does not meet the <500ms target on RPi5. Consider a lighter model for real-time edge use
- Install: `pip install -r requirements-pi.txt` — no `torch` install needed. Pi/ONNX inference
  uses a numpy/scipy Kaldi-compatible mel front-end (`MelFrontend`, D27) instead of the
  `transformers`/`torch`-based `ASTFeatureExtractor`

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
