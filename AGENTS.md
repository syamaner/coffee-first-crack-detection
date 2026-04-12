# AGENTS.md — Coffee First Crack Detection

Project rules and context for Warp/Oz agents and Claude Code working in this repository.

---

## Rules

- Python 3.11+ with full type hints on all public functions and methods
- Google-style docstrings
- `ruff check` and `ruff format` must pass before marking code complete
- `pyright` must pass with no errors on new code
- All dependencies declared in `pyproject.toml` — never install ad-hoc
- Large files (WAV, checkpoints, ONNX models) go to HuggingFace Hub — **never commit to git**
- `data/` and `experiments/` and `exports/` are `.gitignore`'d — keep them that way
- Seed all RNG (`torch.manual_seed`, `numpy.random.seed`) using `configs/default.yaml` seed value
- Privacy: raw audio may contain ambient conversation — never commit WAV files
- Commit messages must reference the story issue number and include:
  ```
  Co-Authored-By: Oz <oz-agent@warp.dev>
  ```
- One PR per story, branch: `feature/{issue-number}-{slug}`
- Before starting a task: read `docs/state/registry.md` → open epic file → check GitHub issue

---

## Quick Commands

### Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
```

### Data preparation
```bash
# Generate recordings manifest from data/raw/
python -c "from coffee_first_crack.dataset import generate_recordings_manifest; generate_recordings_manifest('data/raw', 'data/recordings.csv')"

# Convert Label Studio JSON export to per-file annotations
python -m coffee_first_crack.data_prep.convert_labelstudio_export \
  --input data/labels/export.json --output data/labels --data-root data/raw

# Chunk annotated audio into 10s windows
python -m coffee_first_crack.data_prep.chunk_audio \
  --labels-dir data/labels --audio-dir data/raw --output-dir data/processed

# Stratified train/val/test split (70/15/15)
python -m coffee_first_crack.data_prep.dataset_splitter \
  --input data/processed --output data/splits --train 0.7 --val 0.15 --test 0.15 --seed 42
```

See `docs/data_preparation.md` for the full annotated pipeline including Label Studio steps.

### Training
```bash
# Single run (auto-detects MPS/CUDA/CPU)
python -m coffee_first_crack.train \
  --data-dir data/splits --experiment-name baseline_v1

# With push to HuggingFace Hub after training
python -m coffee_first_crack.train \
  --data-dir data/splits --experiment-name baseline_v1 --push-to-hub

# Resume from checkpoint
python -m coffee_first_crack.train \
  --data-dir data/splits --resume experiments/baseline_v1/checkpoint-best
```

### Evaluation
```bash
python -m coffee_first_crack.evaluate \
  --model-dir experiments/baseline_v1/checkpoint-best \
  --test-dir data/splits/test \
  --output-dir experiments/baseline_v1/evaluation
```

### Inference
```bash
# File-based sliding window
python -m coffee_first_crack.inference \
  --audio data/raw/mic2-brazil-roast1.wav \
  --model-dir syamaner/coffee-first-crack-detection

# Live microphone
python -m coffee_first_crack.inference --microphone \
  --model-dir syamaner/coffee-first-crack-detection

# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### ONNX Export (for Raspberry Pi 5)
```bash
python -m coffee_first_crack.export_onnx \
  --model-dir experiments/baseline_v5/checkpoint-best \
  --output-dir exports/onnx --quantize --benchmark
```

### ONNX Evaluation (no PyTorch inference — works on RPi5)
```bash
# Evaluate INT8 model on test split
python scripts/evaluate_onnx.py \
  --onnx-dir exports/onnx/int8 \
  --test-dir data/splits/test \
  --output results/eval.json \
  --threads 2  # default: 2 on ARM64, auto on x86

# Latency benchmark with dummy audio
python scripts/benchmark_onnx_pi.py \
  --onnx-dir exports/onnx \
  --n-runs 30 \
  --output results/latency.json
```

### Benchmark (PyTorch + ONNX, Mac/GPU only)
```bash
python scripts/benchmark_platforms.py \
  --model-dir experiments/baseline_v5/checkpoint-best \
  --onnx-dir exports/onnx
```

### Push to HuggingFace Hub
```bash
python scripts/push_to_hub.py \
  --model-dir experiments/baseline_v5/checkpoint-best \
  --repo-id syamaner/coffee-first-crack-detection
```

### Tests
```bash
pytest tests/ -v
```

### Lint / type-check
```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
```

---

## Codebase Architecture

```
src/coffee_first_crack/
  __init__.py          — public API exports
  model.py             — build_model(), build_feature_extractor(), FirstCrackClassifier
  dataset.py           — FirstCrackDataset, create_dataloaders(), parse_filename_metadata()
  train.py             — WeightedLossTrainer (HF Trainer subclass), TrainingArguments, CLI
  evaluate.py          — test-set evaluation, MetricsCalculator, confusion matrix plot
  inference.py         — SlidingWindowInference, FirstCrackDetector (file + microphone)
  export_onnx.py       — ONNX export (FP32 + INT8) via optimum
  utils/
    device.py          — get_device() MPS→CUDA→CPU, get_dataloader_kwargs()
    metrics.py         — MetricsCalculator (accuracy, F1, ROC-AUC, confusion matrix)
  data_prep/
    convert_labelstudio_export.py  — Label Studio JSON → per-file annotation JSON
    chunk_audio.py                 — annotated WAV → 10s chunks under data/processed/
    dataset_splitter.py            — stratified train/val/test split
scripts/
  evaluate_onnx.py     — ONNX-only test-set evaluation (no PyTorch inference dep)
  benchmark_onnx_pi.py — ONNX-only latency benchmark (dummy audio, no I/O variance)
  benchmark_platforms.py — PyTorch + ONNX benchmark (Mac/GPU only)
  push_to_hub.py       — publish model + dataset to HuggingFace Hub
  rebuild_and_train.sh — full pipeline: clean → chunk → split → train → eval
```

**Key design decisions:**
- HF-native: `save_pretrained` / `from_pretrained` — no custom `.pt` packaging
- WeightedLossTrainer overrides `compute_loss()` for class-weighted CrossEntropyLoss
- ASTFeatureExtractor params fixed: mean=-4.2677393, std=4.5689974, 128 mel bins, 16kHz
- MPS: `num_workers=0`, `pin_memory=False`; CUDA: `num_workers=4`, `pin_memory=True`
- RPi5: ONNX Runtime for inference, INT8 quantized model (90MB). PyTorch CPU still needed for ASTFeatureExtractor filterbank computation.
- RPi5 thread limiting: `--threads 2` default on ARM64 to prevent under-voltage crashes with standard 5V/3A PSUs

**Pointers to key docs:**
- Data pipeline step-by-step: `docs/data_preparation.md`
- Hyperparameter tuning history: `docs/hyperparameter_tuning.md`
- Multi-mic hardware setup: `docs/multi_mic_setup.md`
- Epic progress: `docs/state/epics/coffee-first-crack-detection.md`
- Epic registry: `docs/state/registry.md`
- HuggingFace model: https://huggingface.co/syamaner/coffee-first-crack-detection
- HuggingFace dataset: https://huggingface.co/datasets/syamaner/coffee-first-crack-audio
- GitHub: https://github.com/syamaner/coffee-first-crack-detection

---

## Epic State Management

Before starting any task:
1. Read `docs/state/registry.md` to find the active epic
2. Open `docs/state/epics/coffee-first-crack-detection.md` — check story status
3. Open the GitHub story issue — read comments for latest requirements
4. Work on a branch: `feature/{issue-number}-{slug}`

After completing a story:
1. Check off the story in `docs/state/epics/coffee-first-crack-detection.md`
2. Update **Active Context** section with what was built
3. Comment on the GitHub story issue with a delivery summary, then close it
4. Tick the checkbox in GitHub epic issue #1
5. Open a PR referencing the story issue

---

## Platform Notes

| Platform | Device | Train | Infer | Notes |
|----------|--------|-------|-------|-------|
| Apple M3+ Mac | MPS | ✅ | ✅ | num_workers=0, pin_memory=False, ~197ms INT8 ONNX |
| Ubuntu + RTX 4090 | CUDA | ✅ | ✅ | fp16/bf16, num_workers=4, pin_memory=True |
| Raspberry Pi 5 (16GB) | CPU | ❌ | ✅ | ONNX Runtime, INT8 model, ~2.4s @ 2 threads |

### RPi5 Hardware Requirements
- **PSU**: Official RPi5 27W (5V/5A) USB-C — standard chargers (5V/3A incl. Apple 96W) cause under-voltage crashes
- **Cooling**: Active cooler mandatory — sustained inference hits 77°C+ without it
- **Storage**: NVMe (Gen 2 default) works fine
- **Threads**: Default 2 via `--threads` flag; 4 threads needs 27W PSU + active cooler
- **Python**: 3.11+ (tested with 3.13.5)
- **Install**: `pip install -r requirements-pi.txt` + `pip install torch --index-url https://download.pytorch.org/whl/cpu`
