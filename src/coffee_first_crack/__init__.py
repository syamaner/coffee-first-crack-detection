"""Coffee First Crack Detection — audio ML model for detecting first crack during coffee roasting."""

__version__ = "0.1.0"
__author__ = "Sertan Yamaner"

from coffee_first_crack.model import FirstCrackClassifier, build_model, build_feature_extractor
from coffee_first_crack.dataset import FirstCrackDataset, create_dataloaders

__all__ = [
    "FirstCrackClassifier",
    "build_model",
    "build_feature_extractor",
    "FirstCrackDataset",
    "create_dataloaders",
]
