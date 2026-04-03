---
name: export-onnx
description: Export a trained model to ONNX format (FP32 + INT8 quantized) for Raspberry Pi 5 deployment. Use when asked to export, convert to ONNX, or prepare the model for edge/RPi deployment.
---

## Export to ONNX — Coffee First Crack Detection

Target: Raspberry Pi 5 (ARM64, 16GB RAM, CPU-only inference via ONNX Runtime).

### Steps

**1. Export FP32 + INT8 ONNX**
```bash
python -m coffee_first_crack.export_onnx \
  --model-dir experiments/baseline_v1/checkpoint-best \
  --output-dir exports/onnx \
  --quantize
```

Produces:
```
exports/onnx/
  model.onnx              ← FP32 (full precision)
  model_quantized.onnx    ← INT8 dynamic quantization (smaller, faster on CPU)
```

**2. Verify ONNX model**
```bash
python -c "
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession('exports/onnx/model_quantized.onnx')
dummy = np.random.randn(1, 1024, 128).astype(np.float32)
out = sess.run(None, {'input_features': dummy})
print('Output shape:', out[0].shape)  # expect (1, 2)
"
```

**3. Run platform benchmark**
```bash
python scripts/benchmark_platforms.py \
  --model-dir experiments/baseline_v1/checkpoint-best \
  --onnx-dir exports/onnx
```

**4. Expected latency targets**
| Platform | Model | Target |
|----------|-------|--------|
| Apple M3+ | PyTorch MPS | < 100ms / 10s window |
| RTX 4090 | PyTorch CUDA | < 30ms / 10s window |
| RPi5 | ONNX INT8 CPU | < 500ms / 10s window |

**5. Deploy to RPi5**
Copy to RPi5:
```bash
scp exports/onnx/model_quantized.onnx pi@raspberrypi:~/coffee-first-crack/
scp requirements-pi.txt pi@raspberrypi:~/coffee-first-crack/
```
On RPi5:
```bash
pip install -r requirements-pi.txt
python -m coffee_first_crack.inference \
  --onnx exports/onnx/model_quantized.onnx \
  --microphone
```

### Notes
- Use `model_quantized.onnx` on RPi5 — ~3-5x faster than FP32 ONNX on CPU
- Do NOT install `torch` on RPi5 unless needed — ONNX Runtime is sufficient
- ONNX model takes `input_features` (shape `[batch, max_length, num_mel_bins]`) — feature extraction still runs on CPU via `ASTFeatureExtractor`
