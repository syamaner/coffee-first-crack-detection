"""Training script for coffee first crack detection using HuggingFace Trainer API.

Uses a custom ``WeightedLossTrainer`` subclass that overrides ``compute_loss``
to apply class-weighted CrossEntropyLoss for handling dataset imbalance.

Usage::

    python -m coffee_first_crack.train \\
        --data-dir data/splits \\
        --experiment-name baseline_v1

    # With push to HuggingFace Hub after training
    python -m coffee_first_crack.train \\
        --data-dir data/splits \\
        --experiment-name baseline_v1 \\
        --push-to-hub

    # Resume from a checkpoint directory
    python -m coffee_first_crack.train \\
        --data-dir data/splits \\
        --resume experiments/baseline_v1/checkpoint-best
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import (
    ASTForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from coffee_first_crack.dataset import FirstCrackDataset, create_dataloaders
from coffee_first_crack.model import (
    DEFAULT_BASE_MODEL,
    build_feature_extractor,
    build_model,
)
from coffee_first_crack.utils.device import get_dataloader_kwargs, get_device
from coffee_first_crack.utils.metrics import MetricsCalculator


# ── Weighted-loss trainer ─────────────────────────────────────────────────────


class WeightedLossTrainer(Trainer):
    """HuggingFace ``Trainer`` subclass with class-weighted CrossEntropyLoss.

    Pass ``class_weights`` as a float tensor of shape ``(num_classes,)``
    when constructing. The weights are moved to the correct device automatically.

    Args:
        class_weights: Per-class loss weights. Passed to ``nn.CrossEntropyLoss``.
        *args: Positional arguments forwarded to ``Trainer``.
        **kwargs: Keyword arguments forwarded to ``Trainer``.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: ASTForAudioClassification,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Override loss with class-weighted CrossEntropyLoss.

        Args:
            model: The model being trained.
            inputs: Batch dict containing ``labels`` and model inputs.
            return_outputs: If ``True``, also return model outputs.

        Returns:
            Loss tensor, or ``(loss, outputs)`` if ``return_outputs`` is ``True``.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── HF Dataset adapter ────────────────────────────────────────────────────────


class _HFDatasetAdapter(torch.utils.data.Dataset):
    """Wraps ``FirstCrackDataset`` to return dicts compatible with HF Trainer.

    The Trainer expects batches with ``input_values`` (or ``input_features``)
    and ``labels``. We use the ``ASTFeatureExtractor`` to convert raw waveforms
    to ``input_features`` on-the-fly.
    """

    def __init__(self, base_dataset: FirstCrackDataset) -> None:
        self._base = base_dataset
        self._extractor = build_feature_extractor()

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_tensor, label = self._base[idx]
        audio_list: list[float] = audio_tensor.tolist()
        inputs = self._extractor(
            [audio_list],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_features": inputs["input_features"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ── compute_metrics callback ──────────────────────────────────────────────────


def _make_compute_metrics() -> Any:
    """Return a ``compute_metrics`` function for the HF Trainer.

    Uses ``MetricsCalculator`` to compute accuracy, F1, precision, recall,
    and first-crack recall.
    """
    import evaluate as hf_evaluate  # lazy import — only needed during training

    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric = hf_evaluate.load("f1")

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

        calc = MetricsCalculator()
        calc.all_preds = preds.tolist()
        calc.all_labels = labels.tolist()
        calc.all_probs = probs.tolist()
        results = calc.compute()

        # Also use HF evaluate for standard metrics
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")

        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"],
            "precision": results["precision"],
            "recall": results["recall"],
            "recall_first_crack": results["recall_first_crack"],
            "roc_auc": results.get("roc_auc", 0.0),
        }

    return compute_metrics


# ── main training function ────────────────────────────────────────────────────


def train(
    data_dir: Path,
    experiment_name: str,
    config_path: Path = Path("configs/default.yaml"),
    resume_from: Path | None = None,
    push_to_hub: bool = False,
    fp16: bool = False,
    bf16: bool = False,
) -> Path:
    """Run the full training pipeline.

    Args:
        data_dir: Root directory containing ``train/``, ``val/``, ``test/`` splits.
        experiment_name: Name for the experiment directory under ``experiments/``.
        config_path: Path to YAML config file.
        resume_from: Optional path to a checkpoint directory to resume from.
        push_to_hub: If ``True``, push the final model to HuggingFace Hub.
        fp16: Enable FP16 mixed precision (CUDA only).
        bf16: Enable BF16 mixed precision (CUDA only).

    Returns:
        Path to the best checkpoint directory.
    """
    # Load config
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    ac = cfg["audio"]
    device = get_device()
    dl_kwargs = get_dataloader_kwargs(device)

    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])

    print(f"\nDevice: {device}")
    print(f"Experiment: {experiment_name}")

    # Paths
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    print("Loading datasets...")
    train_base = FirstCrackDataset(
        train_dir,
        sample_rate=ac["sample_rate"],
        target_length=ac["target_length_sec"],
        crop_mode=tc["train_crop_mode"],
    )
    val_base = FirstCrackDataset(
        val_dir,
        sample_rate=ac["sample_rate"],
        target_length=ac["target_length_sec"],
        crop_mode=tc["eval_crop_mode"],
    )
    train_stats = train_base.get_statistics()
    val_stats = val_base.get_statistics()
    print(
        f"Train: {train_stats['total_samples']} samples "
        f"({train_stats['first_crack']} first_crack, "
        f"{train_stats['no_first_crack']} no_first_crack)"
    )
    print(
        f"Val:   {val_stats['total_samples']} samples "
        f"({val_stats['first_crack']} first_crack, "
        f"{val_stats['no_first_crack']} no_first_crack)"
    )

    class_weights = train_base.get_class_weights()
    print(f"Class weights: no_first_crack={class_weights[0]:.3f}, first_crack={class_weights[1]:.3f}")

    train_dataset = _HFDatasetAdapter(train_base)
    val_dataset = _HFDatasetAdapter(val_base)

    # Model
    print(f"\nLoading base model: {DEFAULT_BASE_MODEL}")
    model = build_model(device=device)

    feature_extractor = build_feature_extractor()

    # TrainingArguments
    hub_kwargs: dict[str, Any] = {}
    if push_to_hub:
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": "syamaner/coffee-first-crack-detection",
        }

    training_args = TrainingArguments(
        output_dir=str(experiment_dir),
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        learning_rate=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        warmup_steps=tc["warmup_steps"],
        eval_strategy=tc["evaluation_strategy"],
        save_strategy=tc["save_strategy"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=True,
        logging_dir=str(experiment_dir / "logs"),
        logging_steps=10,
        save_total_limit=3,
        fp16=fp16 and device == "cuda",
        bf16=bf16 and device == "cuda",
        dataloader_num_workers=dl_kwargs["num_workers"],
        seed=tc["seed"],
        report_to=["tensorboard"],
        **hub_kwargs,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=feature_extractor,
        compute_metrics=_make_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=tc["early_stopping_patience"]
            )
        ],
    )

    # Train
    print("\nStarting training...\n")
    if resume_from:
        trainer.train(resume_from_checkpoint=str(resume_from))
    else:
        trainer.train()

    # Save best model using HF save_pretrained
    best_dir = experiment_dir / "checkpoint-best"
    trainer.save_model(str(best_dir))
    feature_extractor.save_pretrained(str(best_dir))
    print(f"\nBest model saved to: {best_dir}")

    if push_to_hub:
        trainer.push_to_hub()
        print("Model pushed to HuggingFace Hub")

    return best_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train the coffee first crack detection model")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/splits"),
        help="Directory containing train/val/test splits (default: data/splits)",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name (default: exp_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push model to HuggingFace Hub after training",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 (CUDA only)")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 (CUDA only)")
    args = parser.parse_args()

    if not (args.data_dir / "train").exists():
        print(f"Error: train split not found under {args.data_dir}")
        print("Run data preparation first — see docs/data_preparation.md")
        sys.exit(1)

    experiment_name = args.experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train(
        data_dir=args.data_dir,
        experiment_name=experiment_name,
        config_path=args.config,
        resume_from=args.resume,
        push_to_hub=args.push_to_hub,
        fp16=args.fp16,
        bf16=args.bf16,
    )


if __name__ == "__main__":
    main()
