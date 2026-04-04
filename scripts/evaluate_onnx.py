#!/usr/bin/env python3
"""Evaluate an ONNX model on the test split — no PyTorch dependency.

Designed to run on Raspberry Pi 5 with only ``requirements-pi.txt`` installed.
Walks ``data/splits/test/{first_crack,no_first_crack}/`` to infer ground-truth
labels from directory names, runs each WAV through the ONNX model, and reports
accuracy, F1, precision, recall, confusion matrix, and per-window latency.

Usage::

    python scripts/evaluate_onnx.py \
        --onnx-dir exports/onnx/int8 \
        --test-dir data/splits/test \
        --output results/pi5_int8_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import ASTFeatureExtractor

# Canonical label mapping — must stay in sync with configs/default.yaml
LABEL2ID: dict[str, int] = {"no_first_crack": 0, "first_crack": 1}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# ASTFeatureExtractor parameters (same as model.py)
FEATURE_EXTRACTOR_KWARGS: dict[str, object] = {
    "max_length": 1024,
    "num_mel_bins": 128,
    "sampling_rate": 16000,
    "do_normalize": True,
    "mean": -4.2677393,
    "std": 4.5689974,
}

SAMPLE_RATE = 16000


def _find_onnx_model(onnx_dir: Path) -> Path:
    """Find the ONNX model file in the given directory."""
    candidates = list(onnx_dir.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No .onnx files found in {onnx_dir}")
    if len(candidates) > 1:
        # Prefer quantized model if present
        quantized = [c for c in candidates if "quantized" in c.name]
        if quantized:
            return quantized[0]
    return candidates[0]


def _collect_test_samples(test_dir: Path) -> list[tuple[Path, int]]:
    """Walk test directory and return (wav_path, label_id) pairs."""
    samples: list[tuple[Path, int]] = []
    for label_name, label_id in LABEL2ID.items():
        label_dir = test_dir / label_name
        if not label_dir.exists():
            print(f"Warning: {label_dir} not found, skipping")
            continue
        wav_files = sorted(label_dir.glob("*.wav"))
        for wav_path in wav_files:
            samples.append((wav_path, label_id))
    print(f"Collected {len(samples)} test samples")
    for label_name, label_id in LABEL2ID.items():
        count = sum(1 for _, lid in samples if lid == label_id)
        print(f"  {label_name}: {count}")
    return samples


def evaluate(
    onnx_dir: Path,
    test_dir: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Run ONNX evaluation on the test split.

    Args:
        onnx_dir: Directory containing the ONNX model and preprocessor config.
        test_dir: Test split directory with ``first_crack/`` and ``no_first_crack/`` subdirs.
        output_path: Optional path to write JSON results.

    Returns:
        Dict with metrics, confusion matrix, and latency stats.
    """
    import onnxruntime as rt

    onnx_path = _find_onnx_model(onnx_dir)
    print(f"Model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Load feature extractor
    extractor = ASTFeatureExtractor(**FEATURE_EXTRACTOR_KWARGS)

    # Create ONNX Runtime session
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"ONNX Runtime: {rt.__version__}, provider: CPUExecutionProvider")

    # Collect test samples
    samples = _collect_test_samples(test_dir)
    if not samples:
        raise ValueError(f"No test samples found in {test_dir}")

    # Run inference
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    latencies_ms: list[float] = []

    print(f"\nRunning inference on {len(samples)} samples...")
    for i, (wav_path, label_id) in enumerate(samples):
        # Load audio
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

        # Feature extraction + inference (timed end-to-end)
        t0 = time.perf_counter()
        inputs = extractor(
            [audio.tolist()],
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
        )
        logits = sess.run(None, {input_name: inputs["input_values"]})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        pred_id = int(np.argmax(probs, axis=-1)[0])
        first_crack_prob = float(probs[0, 1])

        y_true.append(label_id)
        y_pred.append(pred_id)
        y_prob.append(first_crack_prob)

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [{i + 1}/{len(samples)}] {elapsed_ms:.0f}ms")

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred).tolist()

    latency_arr = np.array(latencies_ms)
    latency_stats = {
        "p50_ms": float(np.percentile(latency_arr, 50)),
        "p95_ms": float(np.percentile(latency_arr, 95)),
        "mean_ms": float(np.mean(latency_arr)),
        "min_ms": float(np.min(latency_arr)),
        "max_ms": float(np.max(latency_arr)),
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:       {onnx_path.name}")
    print(f"  Samples:     {len(samples)}")
    print(f"  Accuracy:    {acc:.3f} ({acc * 100:.1f}%)")
    print(f"  F1:          {f1:.3f}")
    print(f"  Precision:   {precision:.3f}")
    print(f"  Recall (FC): {recall:.3f}")
    print("\nConfusion Matrix:")
    print(f"  {'':20s} predicted_NFC  predicted_FC")
    print(f"  {'actual_no_first_crack':20s}  {cm[0][0]:>8d}  {cm[0][1]:>11d}")
    print(f"  {'actual_first_crack':20s}  {cm[1][0]:>8d}  {cm[1][1]:>11d}")
    print("\nLatency (per 10s window, end-to-end):")
    print(
        f"  p50={latency_stats['p50_ms']:.1f}ms  "
        f"p95={latency_stats['p95_ms']:.1f}ms  "
        f"mean={latency_stats['mean_ms']:.1f}ms"
    )

    target_ms = 500
    status = "✅ PASS" if latency_stats["p50_ms"] < target_ms else "⚠️  FAIL"
    print(f"  Target: <{target_ms}ms → {status}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL2ID.keys())))

    results: dict[str, object] = {
        "model": str(onnx_path),
        "model_size_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "n_samples": len(samples),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall_first_crack": round(recall, 4),
        "confusion_matrix": cm,
        "latency": latency_stats,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX model on test split (no PyTorch required)"
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        required=True,
        help="Directory containing the ONNX model (e.g. exports/onnx/int8)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/splits/test"),
        help="Test split directory (default: data/splits/test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results (optional)",
    )
    args = parser.parse_args()

    if not args.onnx_dir.exists():
        print(f"Error: ONNX directory not found: {args.onnx_dir}")
        raise SystemExit(1)
    if not args.test_dir.exists():
        print(f"Error: test directory not found: {args.test_dir}")
        raise SystemExit(1)

    evaluate(
        onnx_dir=args.onnx_dir,
        test_dir=args.test_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
