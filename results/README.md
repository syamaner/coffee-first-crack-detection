# Evaluation Results

ONNX inference validation results from issue [#22](https://github.com/syamaner/coffee-first-crack-detection/issues/22).

Model: `baseline_v1` checkpoint exported to ONNX FP32 (345MB) and INT8 (90MB).
Test set: 45 samples (22 first_crack, 23 no_first_crack).

## Quality Summary

All variants produce **identical** results — zero quality loss from quantization or cross-platform differences.

| Metric | Value |
|--------|-------|
| Accuracy | 93.3% |
| F1 | 0.933 |
| Precision | 0.913 |
| Recall (first_crack) | 0.955 |

Confusion matrix (same across all runs):

|  | Predicted NFC | Predicted FC |
|--|---------------|--------------|
| Actual NFC | 21 | 2 |
| Actual FC | 1 | 21 |

## Latency Summary

| Run | Platform | Model | Threads | p50 (ms) | p95 (ms) | mean (ms) | Notes |
|-----|----------|-------|---------|----------|----------|-----------|-------|
| mac_int8_eval | Mac (M-series) | INT8 | auto | 197 | 200 | 198 | ✅ baseline |
| mac_fp32_eval | Mac (M-series) | FP32 | auto | 375 | 379 | 376 | ✅ |
| pi5_int8_4threads_eval | RPi5 | INT8 | 4 | 2,070 | 2,090 | 2,070 | ⭐ recommended Pi config |
| pi5_int8_2threads_eval | RPi5 | INT8 | 2 | 2,436 | 2,704 | 2,499 | thermal throttled (no fan) |
| pi5_int8_eval | RPi5 | INT8 | 1 | 4,441 | 4,464 | 4,443 | stable on any PSU |
| pi5_fp32_eval | RPi5 | FP32 | 1 | 9,412 | 9,484 | 9,424 | baseline comparison |

### Latency Breakdown (INT8, 4 threads, RPi5)

| Stage | Time (ms) | % |
|-------|-----------|---|
| Feature extraction (ASTFeatureExtractor) | 49 | 2% |
| ONNX model inference | 2,019 | 98% |
| **Total** | **2,068** | |

The bottleneck is the ONNX model forward pass (AST, 87M params). Feature extraction is negligible.

## Hardware

**Raspberry Pi 5 Model B Rev 1.1 (16GB)**
- aarch64, Python 3.13.5, ONNX Runtime 1.24.4
- NVMe boot (Gen 2)
- Recommended: adequate PSU + active cooler for 4-thread operation
- 5V/3A PSU causes under-voltage crashes at >1 thread (`throttled=0x50000`)
- Without fan: 77°C under 2-thread load, thermal throttling (`throttled=0xe0000`)
- With fan: 45°C under 4-thread load, no throttling

## Result Files

| File | Description |
|------|-------------|
| `mac_int8_eval.json` | Mac ONNX INT8 evaluation |
| `mac_fp32_eval.json` | Mac ONNX FP32 evaluation |
| `pi5_int8_4threads_eval.json` | RPi5 INT8, 4 threads, with fan ⭐ |
| `pi5_int8_2threads_eval.json` | RPi5 INT8, 2 threads, no fan |
| `pi5_int8_eval.json` | RPi5 INT8, 1 thread |
| `pi5_fp32_eval.json` | RPi5 FP32, 1 thread |
