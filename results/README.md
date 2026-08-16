# Evaluation Results

## baseline_v6_pair_aware / leakage-free candidate (16 Aug 2026)

The pair-safe 541-window test set reports 97.0% accuracy, 0.852 F1, 80.7%
precision, 90.2% recall, and 0.9764 ROC-AUC for the unpublished INT8 candidate
(TN/FP/FN/TP = 479/11/5/46). This remains a chunk-classification result, not a
whole-roast event test.

The same fixed candidate was re-evaluated using the production Mac setting of
8 ONNX threads (10 s inputs, all 541 test windows). Quality was unchanged and
end-to-end per-window latency was:

| Threads | p50 (ms) | p95 (ms) | mean (ms) | 500 ms target |
| ---: | ---: | ---: | ---: | --- |
| 2 | 641.5 | 656.8 | — | fail |
| **8** | **202.7** | **204.9** | **202.9** | **pass** |

Reproduce the production-thread result:

```bash
venv/bin/python scripts/evaluate_onnx.py \
  --onnx-dir exports/onnx-baseline-v6-pair-aware/int8 \
  --test-dir data/splits/test \
  --threads 8 \
  --output results/baseline_v6_pair_aware_int8_8threads_eval.json
```

Candidate INT8 SHA-256:
`324eb3d1b603c10d09759ad7e7019fcc2a7252d9712c2ec57946f9e81c2bd6e4`.

### Fresh full-roast holdout status

No decisive fresh full-roast holdout can be run from the current corpus. All 34
complete/stageable MCP sessions appear in train, validation, or the existing
541-window test set. The only four session IDs absent from every split are
aborted/fault captures lasting 5.9–7.25 seconds; each lacks an authoritative
recording-relative `beans_added` milestone and is shorter than the 10-second
detector window.

`scripts/evaluate_mcp_heldout.py` now runs the real `coffee-roaster-mcp` ONNX
backend, detector-paced WAV pipeline, and confirmation adapter against a fresh
cohort. It freezes the model/profile/cohort before inference and fails closed on
prior pair/checksum exposure, missing T0 alignment, incomplete mic pairs, short
recordings, or changed protocol inputs. Mic1 is primary and mic2 is reported as
paired robustness evidence. See `docs/data_preparation.md` for the exact resume
command after newly captured mic1 holdout labels are exported.

---

## baseline_v5 / 303-sample test set (12 Jul, issue #55)

Reconciliation of a headline-number discrepancy: the committed
`experiments/baseline_v5/evaluation/test_results.json` reported "1 FN, 6 FP" while issue #55
reported "4 FP" — both citing the same 303-sample `data/splits/test/`. Root cause: the
committed AST eval predates a same-session regeneration of `data/splits/test/`, so it scored
a since-replaced version of the split. The "4 FP" figure is real but is the **ONNX INT8**
export's number, not the AST/PyTorch checkpoint's — two different, legitimate
configurations. Fresh, reproducible numbers against the checkpoint and split currently on
disk (deterministic, `crop_mode="center"`, both runs reproduced twice):

| Config | Accuracy | F1 | Precision | Recall (FC) | ROC-AUC | Confusion (TN/FP/FN/TP) |
|--------|----------|----|-----------|-------------|---------|--------------------------|
| AST / PyTorch fp32 (`checkpoint-best`) | 98.02% | 0.932 | 89.13% | 97.62% | 0.9979 | 256/5/1/41 |
| ONNX fp32 (`exports/onnx/fp32`) | 98.02% | 0.932 | 89.13% | 97.62% | 0.9979 | 256/5/1/41 |
| ONNX int8 (`exports/onnx/int8`) | 98.35% | 0.943 | 91.11% | 97.62% | 0.9976 | 257/4/1/41 |

ONNX fp32 exactly matches the AST/PyTorch checkpoint (expected — fp32 export shouldn't lose
precision). INT8 quantization scored marginally *better* here (4 FP vs 5, +1.98pp precision) —
a quantization-noise-sized shift on one 303-sample set, not proof INT8 always matches or beats
fp32, but on this evidence there was no quality loss to trade against the latency win.

Latency (`scripts/benchmark_platforms.py`, this Mac, dummy 10s audio, p50 of 30 runs after 5
warmup):

| Backend | p50 (ms) | p95 (ms) | mean (ms) |
|---------|----------|----------|-----------|
| PyTorch / MPS | 56.2 | 58.2 | 56.4 |
| ONNX Runtime / CPU, fp32 | 428.6 | 437.8 | 424.1 |
| ONNX Runtime / CPU, int8 | 216.3 | 222.2 | 216.8 |

INT8 is ~2x faster than fp32 on ONNX Runtime CPU, consistent with the historical baseline_v1
figures below — same shape, different (newer) model and machine, so absolute ms are not
directly comparable across the two sections.

Reproduce:
```bash
python -m coffee_first_crack.evaluate --model-dir experiments/baseline_v5/checkpoint-best \
  --test-dir data/splits/test --output-dir results/baseline_v5_303set_ast_repro
python scripts/evaluate_onnx.py --onnx-dir exports/onnx/fp32 --test-dir data/splits/test \
  --output results/baseline_v5_303set_onnx_fp32_repro.json
python scripts/evaluate_onnx.py --onnx-dir exports/onnx/int8 --test-dir data/splits/test \
  --output results/baseline_v5_303set_onnx_int8_repro.json
python scripts/benchmark_platforms.py --model-dir experiments/baseline_v5/checkpoint-best \
  --onnx-dir exports/onnx --n-runs 30 --output results/baseline_v5_303set_latency_repro.json
```

Raw JSON: `results/baseline_v5_303set/{ast_fp32_eval,onnx_fp32_eval,onnx_int8_eval,latency_benchmark_mac}.json`.

---

## baseline_v1 / 45-sample test set (issue #22, historical)

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

## Threshold Sweep (INT8, RPi5)

Sweep run on Pi loading model from HF Hub (`syamaner/coffee-first-crack-detection`, `onnx/int8`).
ROC-AUC = 0.988 across all thresholds.

| Threshold | Accuracy | Precision | Recall | F1 | FP | FN |
|-----------|----------|-----------|--------|-------|----|----|  
| 0.50–0.65 | 93.3% | 0.913 | 0.955 | 0.933 | 2 | 1 |
| 0.70–0.75 | 91.1% | 0.909 | 0.909 | 0.909 | 2 | 2 |
| 0.80–0.90 | 93.3% | 0.952 | 0.909 | 0.930 | 1 | 2 |
| 0.95 | 88.9% | 1.000 | 0.773 | 0.872 | 0 | 5 |

**Chosen Pi threshold: 0.90** — reduces false positives to 1 while preserving 0.909 recall and 0.930 F1, matching the repository Pi inference config.

## Final HF Hub Evaluations (INT8, RPi5)

Models loaded from HuggingFace Hub (not local) — confirms end-to-end deployment path.

| Config | Threads | p50 (ms) | p95 (ms) | mean (ms) | Accuracy | F1 | Notes |
|--------|---------|----------|----------|-----------|----------|----|-------|
| 4 threads (fan) | 4 | 2,068 | 2,084 | 2,070 | 93.3% | 0.933 | ⭐ recommended |
| 2 threads (no fan) | 2 | 2,452 | 2,470 | 2,453 | 93.3% | 0.933 | minimal hardware |

Latency is consistent with earlier local-model runs — HF Hub caching works as expected.

## Result Files

| File | Description |
|------|-------------|
| `mac_int8_eval.json` | Mac ONNX INT8 evaluation |
| `mac_fp32_eval.json` | Mac ONNX FP32 evaluation |
| `pi5_int8_4threads_eval.json` | RPi5 INT8, 4 threads, with fan ⭐ |
| `pi5_int8_2threads_eval.json` | RPi5 INT8, 2 threads, no fan |
| `pi5_int8_eval.json` | RPi5 INT8, 1 thread |
| `pi5_fp32_eval.json` | RPi5 FP32, 1 thread |
| `pi5_int8_4t_optimised.json` | RPi5 INT8, 4 threads, HF Hub model ⭐ |
| `pi5_int8_2t_optimised.json` | RPi5 INT8, 2 threads, HF Hub model |
| `pi5_threshold_sweep.json` | RPi5 threshold sweep (0.50–0.95) |
| `threshold_sweep.json` | Mac threshold sweep (0.50–0.95) |
| `simulation.json` | Parameter space simulation (135 combinations) |
