#!/usr/bin/env python3
"""Random Forest + MFCC comparator baseline for first-crack detection.

A classical, non-transformer comparator for the AST-based detector, following the
pre-registered protocol in ``results/rf_mfcc_baseline/PROTOCOL.md``. Trains a
``RandomForestClassifier`` on standard MFCC summary statistics (librosa, not the
Kaldi-compatible :class:`~coffee_first_crack.mel_frontend.MelFrontend` the AST
torch-free path needs) using ``data/splits/train/``, and evaluates on
``data/splits/test/`` — the same 303-sample set every other comparator in this repo
uses. Never touches the test set until final evaluation.

Usage::

    python scripts/rf_mfcc_baseline.py \\
        --train-dir data/splits/train \\
        --test-dir data/splits/test \\
        --output results/rf_mfcc_baseline/RESULTS.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Canonical label mapping — must stay in sync with configs/default.yaml
LABEL2ID: dict[str, int] = {"no_first_crack": 0, "first_crack": 1}

SAMPLE_RATE = 16000
N_MFCC = 20


def _collect_samples(split_dir: Path) -> list[tuple[Path, int]]:
    """Walk a split directory and return ``(wav_path, label_id)`` pairs.

    Args:
        split_dir: Directory with ``first_crack/`` and ``no_first_crack/`` subdirs.

    Returns:
        List of ``(wav_path, label_id)`` tuples, sorted for determinism.
    """
    samples: list[tuple[Path, int]] = []
    for label_name, label_id in LABEL2ID.items():
        label_dir = split_dir / label_name
        if not label_dir.exists():
            print(f"Warning: {label_dir} not found, skipping")
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            samples.append((wav_path, label_id))
    print(f"Collected {len(samples)} samples from {split_dir}")
    for label_name, label_id in LABEL2ID.items():
        count = sum(1 for _, lid in samples if lid == label_id)
        print(f"  {label_name}: {count}")
    return samples


def extract_mfcc_features(audio: np.ndarray) -> np.ndarray:
    """Extract a fixed-length MFCC summary feature vector for one audio clip.

    Per the pre-registered protocol: mean + standard deviation of ``N_MFCC``
    MFCC coefficients across time frames, using librosa's standard (non-Kaldi)
    implementation — this is a comparator baseline, not a numerical-equivalence
    target, so it does not need the AST frontend's Kaldi-fbank compatibility.

    Args:
        audio: 1-D mono waveform sampled at :data:`SAMPLE_RATE`.

    Returns:
        A ``(2 * N_MFCC,)`` feature vector: ``N_MFCC`` means followed by
        ``N_MFCC`` standard deviations.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def _load_features(samples: list[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Load audio and extract MFCC features for a list of samples.

    Args:
        samples: ``(wav_path, label_id)`` pairs.

    Returns:
        Tuple of ``(X, y)`` — feature matrix and integer label array.
    """
    features: list[np.ndarray] = []
    labels: list[int] = []
    for wav_path, label_id in samples:
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        features.append(extract_mfcc_features(audio))
        labels.append(label_id)
    return np.stack(features), np.array(labels)


def run(
    train_dir: Path,
    test_dir: Path,
    output_path: Path | None = None,
    seed: int = 42,
    class_weight: str = "balanced",
) -> dict[str, object]:
    """Train the RF+MFCC comparator and evaluate on the test split.

    Args:
        train_dir: Training split directory (``data/splits/train``).
        test_dir: Test split directory (``data/splits/test``) — used only for
            final evaluation, never for feature selection or tuning.
        output_path: Optional path to write JSON results.
        seed: Random seed, matching the repo's ``configs/default.yaml`` convention.
        class_weight: ``"balanced"`` (protocol default — data-blind inverse-frequency
            rebalancing for the ~15% positive training split, 136:786) or ``"none"``
            (bare ``RandomForestClassifier`` default, no rebalancing). Exposed as a CLI
            flag so the PROTOCOL.md transparency check comparing the two is itself
            reproducible from the committed script, not just a one-off interactive run.

    Returns:
        Dict with metrics, confusion matrix, and per-window inference latency.
    """
    print(f"\n=== Training data: {train_dir} ===")
    train_samples = _collect_samples(train_dir)
    if not train_samples:
        raise ValueError(f"No training samples found in {train_dir}")

    print("\nExtracting MFCC features (train)...")
    train_features, train_labels = _load_features(train_samples)

    # class_weight="balanced" is the PROTOCOL.md default — data-blind inverse-frequency
    # rebalancing for the ~15% positive training split, not a hyperparameter search.
    # class_weight="none" reproduces the bare-default transparency check. All other
    # RandomForestClassifier args are sklearn defaults either way.
    sklearn_class_weight = None if class_weight == "none" else class_weight
    clf = RandomForestClassifier(random_state=seed, class_weight=sklearn_class_weight)
    print(f"\nFitting RandomForestClassifier (seed={seed}, class_weight={class_weight})...")
    clf.fit(train_features, train_labels)

    print(f"\n=== Test data: {test_dir} ===")
    test_samples = _collect_samples(test_dir)
    if not test_samples:
        raise ValueError(f"No test samples found in {test_dir}")

    print("\nExtracting MFCC features + predicting (test, timed end-to-end)...")
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    latencies_ms: list[float] = []

    for i, (wav_path, label_id) in enumerate(test_samples):
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

        t0 = time.perf_counter()
        feats = extract_mfcc_features(audio).reshape(1, -1)
        pred_id = int(clf.predict(feats)[0])
        prob_fc = float(clf.predict_proba(feats)[0, 1])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        y_true.append(label_id)
        y_pred.append(pred_id)
        y_prob.append(prob_fc)
        latencies_ms.append(elapsed_ms)

        if (i + 1) % 50 == 0 or (i + 1) == len(test_samples):
            print(f"  [{i + 1}/{len(test_samples)}]")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    latency_arr = np.array(latencies_ms)
    latency_stats = {
        "p50_ms": float(np.percentile(latency_arr, 50)),
        "p95_ms": float(np.percentile(latency_arr, 95)),
        "mean_ms": float(np.mean(latency_arr)),
    }

    print("\n" + "=" * 60)
    print("RF + MFCC — TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Confusion matrix: {cm}")
    print(
        f"  Latency: p50={latency_stats['p50_ms']:.2f}ms "
        f"p95={latency_stats['p95_ms']:.2f}ms mean={latency_stats['mean_ms']:.2f}ms"
    )
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL2ID.keys())))

    results: dict[str, object] = {
        "model": "RandomForestClassifier+MFCC",
        "seed": seed,
        "class_weight": class_weight,
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "n_mfcc": N_MFCC,
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
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
        description="Random Forest + MFCC comparator baseline for first-crack detection"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/splits/train"),
        help="Training split directory (default: data/splits/train)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42, matches configs/default.yaml)",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help=(
            "RandomForestClassifier class_weight: 'balanced' (default, protocol-specified "
            "inverse-frequency rebalancing) or 'none' (bare sklearn default, reproduces the "
            "PROTOCOL.md transparency check)"
        ),
    )
    args = parser.parse_args()

    if not args.train_dir.exists():
        print(f"Error: training directory not found: {args.train_dir}")
        raise SystemExit(1)
    if not args.test_dir.exists():
        print(f"Error: test directory not found: {args.test_dir}")
        raise SystemExit(1)

    run(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_path=args.output,
        seed=args.seed,
        class_weight=args.class_weight,
    )


if __name__ == "__main__":
    main()
