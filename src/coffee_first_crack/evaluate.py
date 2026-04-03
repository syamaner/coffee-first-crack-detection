"""Evaluate a fine-tuned checkpoint on the test set.

Produces a full metrics report (accuracy, F1, ROC-AUC, per-class
precision/recall), classification report, and confusion matrix plot.

Usage::

    python -m coffee_first_crack.evaluate \\
        --model-dir experiments/baseline_v1/checkpoint-best \\
        --test-dir data/splits/test \\
        --output-dir experiments/baseline_v1/evaluation

    # Evaluate a HuggingFace Hub checkpoint
    python -m coffee_first_crack.evaluate \\
        --model-dir syamaner/coffee-first-crack-detection \\
        --test-dir data/splits/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from coffee_first_crack.dataset import FirstCrackDataset
from coffee_first_crack.model import FirstCrackClassifier, build_feature_extractor
from coffee_first_crack.utils.device import get_dataloader_kwargs, get_device
from coffee_first_crack.utils.metrics import MetricsCalculator


def evaluate_model(
    model: FirstCrackClassifier,
    test_dir: Path,
    batch_size: int = 8,
    sample_rate: int = 16000,
    target_length: int = 10,
) -> MetricsCalculator:
    """Run evaluation on the test set.

    Args:
        model: Loaded ``FirstCrackClassifier`` (already on correct device).
        test_dir: Directory with ``first_crack/`` and ``no_first_crack/`` subdirs.
        batch_size: Evaluation batch size.
        sample_rate: Target sample rate in Hz.
        target_length: Audio window length in seconds.

    Returns:
        Populated ``MetricsCalculator`` ready to call ``.compute()``.
    """
    dl_kwargs = get_dataloader_kwargs(model.device_str)
    dataset = FirstCrackDataset(
        test_dir,
        sample_rate=sample_rate,
        target_length=target_length,
        crop_mode="center",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    print(f"\nTest set: {len(dataset)} samples")
    stats = dataset.get_statistics()
    print(f"  first_crack:    {stats['first_crack']}")
    print(f"  no_first_crack: {stats['no_first_crack']}")

    metrics = MetricsCalculator()
    model.model.eval()

    print("\nEvaluating...")
    with torch.inference_mode():
        for audio, labels in tqdm(loader, desc="Testing"):
            audio = audio.to(model.device_str)
            labels = labels.to(model.device_str)

            logits = model(audio)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            metrics.update(preds, labels, probs)

    return metrics


def plot_confusion_matrix(cm: object, output_path: Path) -> None:
    """Save a labelled confusion matrix heatmap.

    Args:
        cm: Numpy confusion matrix array.
        output_path: Where to save the PNG.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["no_first_crack", "first_crack"],
        yticklabels=["no_first_crack", "first_crack"],
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the first crack detection model")
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to save_pretrained checkpoint dir or HuggingFace model ID",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("data/splits/test"),
        help="Test set directory (default: data/splits/test)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to save results (default: <model-dir>/evaluation)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Evaluation batch size (default: 8)",
    )
    args = parser.parse_args()

    if not args.test_dir.exists():
        print(f"Error: test directory not found: {args.test_dir}")
        sys.exit(1)

    output_dir = args.output_dir or Path(args.model_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model from: {args.model_dir}")

    model = FirstCrackClassifier(
        model_name_or_path=args.model_dir,
        device=device,
    )

    metrics = evaluate_model(
        model=model,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
    )

    results = metrics.compute()
    cm = metrics.compute_confusion_matrix()
    report = metrics.get_classification_report()

    # Print results
    sep = "=" * 60
    print(f"\n{sep}")
    print("TEST SET RESULTS")
    print(sep)
    print(f"Accuracy:              {results['accuracy']:.4f}")
    print(f"Precision:             {results['precision']:.4f}")
    print(f"Recall:                {results['recall']:.4f}")
    print(f"F1:                    {results['f1']:.4f}")
    if "roc_auc" in results:
        print(f"ROC-AUC:               {results['roc_auc']:.4f}")
    print(f"\nFirst-crack recall:    {results['recall_first_crack']:.4f}  ← key metric")
    print(f"First-crack precision: {results['precision_first_crack']:.4f}")
    print(f"\n{sep}\nCLASSIFICATION REPORT\n{sep}")
    print(report)
    print(f"{sep}\nCONFUSION MATRIX\n{sep}")
    print(cm)

    # Save text results
    results_path = output_dir / "test_results.txt"
    with results_path.open("w") as f:
        f.write(f"Model: {args.model_dir}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"\n{report}\n\nConfusion Matrix:\n{cm}\n")
    print(f"\nResults saved to {results_path}")

    # Save JSON for programmatic use
    json_path = output_dir / "test_results.json"
    with json_path.open("w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    # Save classification report
    (output_dir / "classification_report.txt").write_text(report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
