"""Multi-platform device detection for MPS, CUDA, and CPU (RPi5)."""

from __future__ import annotations

import torch


def get_device() -> str:
    """Detect the best available compute device.

    Priority:
        1. MPS — Apple M-series (M3+/MPS, training + inference)
        2. CUDA — NVIDIA GPU (RTX 4090, training + inference)
        3. CPU — fallback, used for RPi5 inference via ONNX

    Returns:
        Device string: ``"mps"``, ``"cuda"``, or ``"cpu"``.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dataloader_kwargs(device: str | None = None) -> dict:
    """Return DataLoader keyword arguments appropriate for the target device.

    MPS requires ``num_workers=0`` and ``pin_memory=False``.
    CUDA benefits from ``num_workers=4`` and ``pin_memory=True``.
    CPU (RPi5) uses ``num_workers=0`` and ``pin_memory=False``.

    Args:
        device: Device string. If ``None``, auto-detects via :func:`get_device`.

    Returns:
        Dict with ``num_workers`` and ``pin_memory`` keys.
    """
    if device is None:
        device = get_device()

    if device == "cuda":
        return {"num_workers": 4, "pin_memory": True}
    # MPS and CPU both work best with num_workers=0
    return {"num_workers": 0, "pin_memory": False}


def is_training_supported(device: str | None = None) -> bool:
    """Return ``True`` if training is supported on the given device.

    Training is supported on MPS and CUDA. The RPi5 (CPU) is inference-only.

    Args:
        device: Device string. If ``None``, auto-detects.

    Returns:
        ``True`` for MPS and CUDA; ``False`` for CPU.
    """
    if device is None:
        device = get_device()
    return device in ("mps", "cuda")
