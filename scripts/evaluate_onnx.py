#!/usr/bin/env python3
"""Evaluate an ONNX model on the test split — no PyTorch dependency.

Designed to run on Raspberry Pi 5 with only ``requirements-pi.txt`` installed.
Walks ``data/splits/test/{first_crack,no_first_crack}/`` to infer ground-truth
labels from directory names, runs each WAV through the ONNX model, and reports
accuracy, F1, precision, recall, confusion matrix, and per-window latency.

Supports loading the model from a local directory or from HuggingFace Hub.

Usage::

    # Local model
    python scripts/evaluate_onnx.py \
        --onnx-dir exports/onnx/int8 \
        --test-dir data/splits/test \
        --output results/pi5_int8_eval.json

    # HuggingFace Hub model
    python scripts/evaluate_onnx.py \
        --repo-id syamaner/coffee-first-crack-detection \
        --subfolder onnx/int8 \
        --test-dir data/splits/test \
        --output results/eval.json

    # Threshold sweep from HF Hub
    python scripts/evaluate_onnx.py \
        --repo-id syamaner/coffee-first-crack-detection \
        --subfolder onnx/int8 \
        --threshold-sweep --output results/threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import platform
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
    roc_auc_score,
)
from transformers import ASTFeatureExtractor

# Canonical label mapping — must stay in sync with configs/default.yaml
LABEL2ID: dict[str, int] = {"no_first_crack": 0, "first_crack": 1}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

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


def _resolve_model(
    onnx_dir: Path | None = None,
    repo_id: str | None = None,
    subfolder: str = "onnx/int8",
) -> tuple[str, str]:
    """Resolve ONNX model path and feature extractor source.

    Supports two modes:
    - **Local**: ``onnx_dir`` points to a directory containing ``*.onnx`` +
      ``preprocessor_config.json``.
    - **HuggingFace Hub**: ``repo_id`` + ``subfolder`` are used to download the
      ONNX model via ``hf_hub_download`` and load the feature extractor via
      ``from_pretrained(repo_id, subfolder=...)``.

    Args:
        onnx_dir: Local directory with the ONNX model.
        repo_id: HuggingFace Hub repository ID.
        subfolder: Subfolder within the HF repo (default: ``onnx/int8``).

    Returns:
        Tuple of ``(onnx_model_path, extractor_source)`` where
        ``extractor_source`` is either a local path string or a tuple to pass
        to ``from_pretrained``.
    """
    if onnx_dir is not None:
        onnx_path = str(_find_onnx_model(onnx_dir))
        extractor_source = str(onnx_dir)
        return onnx_path, extractor_source

    if repo_id is None:
        raise ValueError("Either --onnx-dir or --repo-id must be specified")

    from huggingface_hub import hf_hub_download

    # Try quantized first, fall back to non-quantized
    for filename in ("model_quantized.onnx", "model.onnx"):
        try:
            onnx_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
            )
            print(f"Downloaded {filename} from {repo_id}/{subfolder}")
            break
        except Exception:  # noqa: BLE001
            continue
    else:
        raise FileNotFoundError(
            f"No ONNX model found in {repo_id}/{subfolder} (tried model_quantized.onnx, model.onnx)"
        )

    # extractor_source is passed to from_pretrained as (repo_id, {subfolder})
    extractor_source = f"{repo_id}:{subfolder}"
    return onnx_path, extractor_source


def _load_extractor(extractor_source: str) -> ASTFeatureExtractor:
    """Load the feature extractor from a local path or HF Hub.

    Args:
        extractor_source: Either a local directory path or
            ``"repo_id:subfolder"`` for HuggingFace Hub loading.

    Returns:
        An initialised ``ASTFeatureExtractor``.
    """
    if ":" in extractor_source and not Path(extractor_source).exists():
        repo_id, subfolder = extractor_source.split(":", 1)
        return ASTFeatureExtractor.from_pretrained(repo_id, subfolder=subfolder)
    return ASTFeatureExtractor.from_pretrained(extractor_source)


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


def _default_threads() -> int:
    """Return a safe default thread count for the current platform.

    ARM64 devices (e.g. Raspberry Pi 5) default to 2 threads to reduce peak
    power draw and avoid under-voltage crashes with standard 5V/3A supplies.
    """
    if platform.machine() in ("aarch64", "arm64"):
        return 2
    return 0  # 0 = let ONNX Runtime decide


def evaluate(
    onnx_dir: Path | None = None,
    test_dir: Path = Path("data/splits/test"),
    output_path: Path | None = None,
    threads: int | None = None,
    repo_id: str | None = None,
    subfolder: str = "onnx/int8",
) -> dict[str, object]:
    """Run ONNX evaluation on the test split.

    Args:
        onnx_dir: Local directory containing the ONNX model and preprocessor config.
        test_dir: Test split directory with ``first_crack/`` and ``no_first_crack/`` subdirs.
        output_path: Optional path to write JSON results.
        threads: Number of ONNX Runtime intra-op threads (0 = auto, None = platform default).
        repo_id: HuggingFace Hub repository ID (alternative to ``onnx_dir``).
        subfolder: HF repo subfolder for model files (default: ``onnx/int8``).

    Returns:
        Dict with metrics, confusion matrix, and latency stats.
    """
    import onnxruntime as rt

    if threads is None:
        threads = _default_threads()

    onnx_path_str, extractor_source = _resolve_model(onnx_dir, repo_id, subfolder)
    onnx_path = Path(onnx_path_str)
    print(f"Model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    extractor = _load_extractor(extractor_source)

    # Create ONNX Runtime session with thread limit
    sess_options = rt.SessionOptions()
    if threads > 0:
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1
    sess = rt.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    thread_info = f"{threads} threads" if threads > 0 else "auto threads"
    print(f"ONNX Runtime: {rt.__version__}, provider: CPUExecutionProvider, {thread_info}")

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
    roc_auc = roc_auc_score(y_true, y_prob)
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
    print(f"  ROC-AUC:     {roc_auc:.3f}")
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
        "model": onnx_path.name,
        "model_size_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "threads": threads,
        "n_samples": len(samples),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall_first_crack": round(recall, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "latency": latency_stats,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def threshold_sweep(
    onnx_dir: Path | None = None,
    test_dir: Path = Path("data/splits/test"),
    output_path: Path | None = None,
    threads: int | None = None,
    threshold_min: float = 0.50,
    threshold_max: float = 0.95,
    threshold_step: float = 0.05,
    repo_id: str | None = None,
    subfolder: str = "onnx/int8",
) -> dict[str, object]:
    """Sweep classification thresholds and report per-threshold metrics.

    Runs ONNX inference once to collect per-sample probabilities, then
    re-evaluates at each threshold without re-running the model.

    Args:
        onnx_dir: Local directory containing the ONNX model and preprocessor config.
        test_dir: Test split directory with ``first_crack/`` and ``no_first_crack/`` subdirs.
        output_path: Optional path to write JSON results.
        threads: Number of ONNX Runtime intra-op threads.
        threshold_min: Lowest threshold to evaluate (inclusive).
        threshold_max: Highest threshold to evaluate (inclusive).
        threshold_step: Step size between thresholds.
        repo_id: HuggingFace Hub repository ID (alternative to ``onnx_dir``).
        subfolder: HF repo subfolder for model files (default: ``onnx/int8``).

    Returns:
        Dict with per-threshold metrics and a recommended threshold.
    """
    import onnxruntime as rt

    if threads is None:
        threads = _default_threads()

    onnx_path_str, extractor_source = _resolve_model(onnx_dir, repo_id, subfolder)
    onnx_path = Path(onnx_path_str)
    print(f"Model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    extractor = _load_extractor(extractor_source)

    sess_options = rt.SessionOptions()
    if threads > 0:
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1
    sess = rt.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    samples = _collect_test_samples(test_dir)
    if not samples:
        raise ValueError(f"No test samples found in {test_dir}")

    # Single inference pass to collect probabilities
    y_true: list[int] = []
    y_prob: list[float] = []
    print(f"\nRunning inference on {len(samples)} samples...")
    for i, (wav_path, label_id) in enumerate(samples):
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        inputs = extractor([audio.tolist()], sampling_rate=SAMPLE_RATE, return_tensors="np")
        logits = sess.run(None, {input_name: inputs["input_values"]})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        y_true.append(label_id)
        y_prob.append(float(probs[0, 1]))
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [{i + 1}/{len(samples)}]")

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    # Sweep thresholds
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    sweep_results: list[dict[str, object]] = []

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 70)
    print(
        f"{'Thresh':>7s}  {'Acc':>6s}  {'Prec':>6s}  {'Recall':>6s}  "
        f"{'F1':>6s}  {'FP':>4s}  {'FN':>4s}  {'AUC':>6s}"
    )
    print("-" * 70)

    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    for thresh in thresholds:
        y_pred = (y_prob_arr >= thresh).astype(int)
        acc = float(accuracy_score(y_true_arr, y_pred))
        prec = float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_true_arr, y_pred).tolist()
        # FP = predicted positive but actually negative (cm[0][1])
        # FN = predicted negative but actually positive (cm[1][0])
        fp = cm[0][1] if len(cm) > 1 and len(cm[0]) > 1 else 0
        fn = cm[1][0] if len(cm) > 1 else 0

        print(
            f"  {thresh:5.2f}  {acc:6.3f}  {prec:6.3f}  {rec:6.3f}  "
            f"{f1:6.3f}  {fp:4d}  {fn:4d}  {roc_auc:6.3f}"
        )

        sweep_results.append(
            {
                "threshold": round(float(thresh), 2),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall_first_crack": round(rec, 4),
                "f1": round(f1, 4),
                "false_positives": fp,
                "false_negatives": fn,
                "confusion_matrix": cm,
            }
        )

    # Find best threshold: highest F1 among those with zero FPs
    zero_fp = [r for r in sweep_results if r["false_positives"] == 0]
    if zero_fp:
        best = max(zero_fp, key=lambda r: r["f1"])  # type: ignore[arg-type]
        recommendation = (
            f"Threshold {best['threshold']} achieves F1={best['f1']} with zero false positives"
        )
    else:
        # Fall back to highest F1 overall
        best = max(sweep_results, key=lambda r: r["f1"])  # type: ignore[arg-type]
        recommendation = (
            f"No threshold eliminates all FPs. "
            f"Best F1={best['f1']} at threshold={best['threshold']}"
        )

    print(f"\nRecommendation: {recommendation}")

    result: dict[str, object] = {
        "model": onnx_path.name,
        "n_samples": len(samples),
        "roc_auc": round(roc_auc, 4),
        "thresholds": sweep_results,
        "recommendation": recommendation,
        "recommended_threshold": best["threshold"],
        "per_sample_probabilities": [
            {"file": samples[i][0].name, "label_id": y_true[i], "prob": round(y_prob[i], 4)}
            for i in range(len(samples))
        ],
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX model on test split (no PyTorch required)"
    )
    # Model source: either local --onnx-dir or HF Hub --repo-id
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--onnx-dir",
        type=Path,
        help="Local directory containing the ONNX model (e.g. exports/onnx/int8)",
    )
    model_group.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace Hub repo ID (e.g. syamaner/coffee-first-crack-detection)",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="onnx/int8",
        help="HF repo subfolder for ONNX model (default: onnx/int8)",
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
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="ONNX Runtime intra-op threads (default: 2 on ARM64, auto otherwise)",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Run threshold sensitivity sweep instead of standard evaluation",
    )
    args = parser.parse_args()

    if args.onnx_dir and not args.onnx_dir.exists():
        print(f"Error: ONNX directory not found: {args.onnx_dir}")
        raise SystemExit(1)
    if not args.test_dir.exists():
        print(f"Error: test directory not found: {args.test_dir}")
        raise SystemExit(1)

    if args.threshold_sweep:
        threshold_sweep(
            onnx_dir=args.onnx_dir,
            test_dir=args.test_dir,
            output_path=args.output,
            threads=args.threads,
            repo_id=args.repo_id,
            subfolder=args.subfolder,
        )
    else:
        evaluate(
            onnx_dir=args.onnx_dir,
            test_dir=args.test_dir,
            output_path=args.output,
            threads=args.threads,
            repo_id=args.repo_id,
            subfolder=args.subfolder,
        )


if __name__ == "__main__":
    main()
