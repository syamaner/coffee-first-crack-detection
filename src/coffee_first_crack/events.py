"""Shared detection event types for first crack detection.

This module is intentionally torch-free so that
``coffee_first_crack.inference_onnx`` can import it on a Raspberry Pi without
``torch`` or ``transformers`` installed.  Both ``inference.py`` (PyTorch path)
and ``inference_onnx.py`` (ONNX path) import from here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionEvent:
    """A confirmed first-crack detection event.

    Attributes:
        timestamp_sec: Time (in seconds) of first confirmed pop.
        timestamp_str: Human-readable ``"MM:SS"`` string.
        confidence: Number of positive pops within the confirmation window.
    """

    timestamp_sec: float
    timestamp_str: str
    confidence: int


def _format_time(seconds: float) -> str:
    """Format seconds as ``MM:SS``.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        String of the form ``"MM:SS"``.
    """
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"
