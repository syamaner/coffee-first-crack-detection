"""Coffee First Crack Detection — audio ML model for detecting first crack during roasting."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__author__ = "Sertan Yamaner"

# Lazy-import torch-dependent submodules so that
# ``import coffee_first_crack`` (e.g. from inference_onnx on the Pi) does not
# pull torch or transformers.  The public names are still importable as
# ``from coffee_first_crack import FirstCrackClassifier`` — Python resolves them
# via __getattr__ on first access.
#
# The TYPE_CHECKING block lets pyright see the names statically while keeping
# the runtime import lazy (TYPE_CHECKING is False at runtime).
if TYPE_CHECKING:
    from coffee_first_crack.dataset import FirstCrackDataset, create_dataloaders
    from coffee_first_crack.model import (
        FirstCrackClassifier,
        build_feature_extractor,
        build_model,
    )

_LAZY: dict[str, str] = {
    "FirstCrackClassifier": "coffee_first_crack.model",
    "build_model": "coffee_first_crack.model",
    "build_feature_extractor": "coffee_first_crack.model",
    "FirstCrackDataset": "coffee_first_crack.dataset",
    "create_dataloaders": "coffee_first_crack.dataset",
}

__all__ = [
    "FirstCrackClassifier",
    "build_model",
    "build_feature_extractor",
    "FirstCrackDataset",
    "create_dataloaders",
]


def __getattr__(name: str) -> object:
    """Lazily import torch-dependent names on first access.

    Args:
        name: Attribute name being requested.

    Returns:
        The resolved attribute from the appropriate submodule.

    Raises:
        AttributeError: If the name is not in the lazy registry.
    """
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        obj = getattr(module, name)
        # Cache on the package namespace so subsequent accesses are free
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'coffee_first_crack' has no attribute {name!r}")
