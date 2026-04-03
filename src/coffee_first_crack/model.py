"""HuggingFace-native model for coffee first crack audio classification.

Wraps ``ASTForAudioClassification`` with label mappings and a pre-configured
``ASTFeatureExtractor`` so the model can be saved and loaded via
``save_pretrained`` / ``from_pretrained``.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from coffee_first_crack.utils.device import get_device

# Canonical label mapping — must stay in sync with configs/default.yaml
LABEL2ID: dict[str, int] = {"no_first_crack": 0, "first_crack": 1}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Default base model from HuggingFace Hub
DEFAULT_BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# ASTFeatureExtractor parameters calibrated on AudioSet (same as base model)
FEATURE_EXTRACTOR_KWARGS: dict[str, Any] = {
    "max_length": 1024,
    "num_mel_bins": 128,
    "sampling_rate": 16000,
    "do_normalize": True,
    "mean": -4.2677393,
    "std": 4.5689974,
}


def build_feature_extractor() -> ASTFeatureExtractor:
    """Build the ``ASTFeatureExtractor`` with project-specific parameters.

    Returns:
        Configured ``ASTFeatureExtractor`` instance.
    """
    return ASTFeatureExtractor(**FEATURE_EXTRACTOR_KWARGS)


def build_model(
    base_model: str = DEFAULT_BASE_MODEL,
    device: str | None = None,
) -> ASTForAudioClassification:
    """Load ``ASTForAudioClassification`` configured for first crack detection.

    Args:
        base_model: HuggingFace model identifier to initialise from.
        device: Target device (``"mps"``, ``"cuda"``, or ``"cpu"``).
                Defaults to auto-detected device.

    Returns:
        Model moved to the target device.
    """
    if device is None:
        device = get_device()

    model = ASTForAudioClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL2ID),
        id2label={str(k): v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


class FirstCrackClassifier(torch.nn.Module):
    """Thin inference wrapper around ``ASTForAudioClassification``.

    Handles raw waveform input end-to-end: feature extraction → model →
    logits / probabilities / class prediction.

    This class is used for inference and direct integration with the streaming
    detector. For training, use the HuggingFace ``Trainer`` API with
    :func:`build_model` directly.

    Args:
        model_name_or_path: HuggingFace model ID or local path.  Pass a
            locally saved ``save_pretrained`` directory to load a fine-tuned
            checkpoint.
        device: Target device. Defaults to auto-detected.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_BASE_MODEL,
        device: str | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = get_device()
        self.device_str = device

        self.feature_extractor = build_feature_extractor()
        self.model = ASTForAudioClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(LABEL2ID),
            id2label={str(k): v for k, v in ID2LABEL.items()},
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(device)
        self.sampling_rate: int = self.feature_extractor.sampling_rate  # type: ignore[assignment]

    def forward(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """Run a forward pass from raw waveform to logits.

        Args:
            audio_batch: Float tensor of shape ``(batch, samples)`` sampled at
                ``self.sampling_rate`` (16 kHz).

        Returns:
            Logit tensor of shape ``(batch, num_labels)``.
        """
        if audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)

        # Feature extraction runs on CPU (HF processors don't support MPS/CUDA)
        audio_list: list[list[float]] = [
            x.detach().cpu().float().tolist() for x in audio_batch
        ]
        inputs: dict[str, Any] = self.feature_extractor(
            audio_list,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        return self.model(**inputs).logits

    @torch.inference_mode()
    def predict_proba(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax.

        Args:
            audio_batch: Raw waveform tensor, shape ``(batch, samples)``.

        Returns:
            Probability tensor, shape ``(batch, num_labels)``.
        """
        return torch.softmax(self.forward(audio_batch), dim=-1)

    @torch.inference_mode()
    def predict(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices.

        Args:
            audio_batch: Raw waveform tensor, shape ``(batch, samples)``.

        Returns:
            Class index tensor, shape ``(batch,)``.
        """
        return torch.argmax(self.predict_proba(audio_batch), dim=-1)
