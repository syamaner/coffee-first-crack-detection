"""Evaluation metrics for binary first crack detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Accumulate and compute classification metrics across batches.

    Tracks predictions, labels, and probabilities from multiple batches
    and computes aggregate metrics at the end of an epoch.

    Example::

        metrics = MetricsCalculator()
        for preds, labels, probs in dataloader_results:
            metrics.update(preds, labels, probs)
        results = metrics.compute()
    """

    def __init__(self, num_classes: int = 2) -> None:
        """Initialise calculator.

        Args:
            num_classes: Number of output classes (default: 2).
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated values."""
        self.all_preds: list[int] = []
        self.all_labels: list[int] = []
        self.all_probs: list[list[float]] = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor | None = None,
    ) -> None:
        """Accumulate a batch of predictions.

        Args:
            predictions: Predicted class indices, shape ``(batch,)``.
            labels: True labels, shape ``(batch,)``.
            probabilities: Optional class probabilities, shape ``(batch, num_classes)``.
        """
        self.all_preds.extend(predictions.detach().cpu().numpy().tolist())
        self.all_labels.extend(labels.detach().cpu().numpy().tolist())
        if probabilities is not None:
            self.all_probs.extend(probabilities.detach().cpu().numpy().tolist())

    def compute(self) -> dict[str, float]:
        """Compute all aggregate metrics.

        Returns:
            Dictionary mapping metric name to scalar value. Includes accuracy,
            precision, recall, F1, per-class precision/recall, and ROC-AUC (if
            probabilities were provided).
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        metrics: dict[str, Any] = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="binary", zero_division=0),
            "recall": recall_score(labels, preds, average="binary", zero_division=0),
            "f1": f1_score(labels, preds, average="binary", zero_division=0),
        }

        # Per-class precision and recall
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        metrics["precision_no_first_crack"] = float(precision_per_class[0])
        metrics["precision_first_crack"] = float(precision_per_class[1])
        metrics["recall_no_first_crack"] = float(recall_per_class[0])
        metrics["recall_first_crack"] = float(recall_per_class[1])

        # ROC-AUC only when probabilities are available
        if self.all_probs:
            probs = np.array(self.all_probs)
            metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])

        return metrics

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix.

        Returns:
            2×2 confusion matrix as a numpy array.
        """
        return confusion_matrix(self.all_labels, self.all_preds)

    def get_classification_report(
        self, target_names: list[str] | None = None
    ) -> str:
        """Return a detailed sklearn classification report string.

        Args:
            target_names: Class names. Defaults to ``["no_first_crack", "first_crack"]``.

        Returns:
            Formatted classification report string.
        """
        if target_names is None:
            target_names = ["no_first_crack", "first_crack"]
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=target_names,
        )


def calculate_batch_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute accuracy for a single batch.

    Args:
        predictions: Predicted class indices, shape ``(batch,)``.
        labels: True labels, shape ``(batch,)``.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0
