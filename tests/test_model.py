"""Tests for model.py — build_model, build_feature_extractor, FirstCrackClassifier.

Tests that load the full AST model (~86M params) are marked ``@pytest.mark.slow``
and skipped by default. Run with ``pytest -m slow`` to include them.
"""

from __future__ import annotations

import pytest
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from coffee_first_crack.model import (
    LABEL2ID,
    ID2LABEL,
    FirstCrackClassifier,
    build_feature_extractor,
    build_model,
)

pytestmark_slow = pytest.mark.slow


# ── Module-scoped fixtures — model loaded once per test session ───────────────


@pytest.fixture(scope="module")
def ast_model() -> ASTForAudioClassification:
    """Load the AST model once for all TestBuildModel / TestFirstCrackClassifier tests."""
    return build_model(device="cpu")


@pytest.fixture(scope="module")
def classifier() -> FirstCrackClassifier:
    """Load FirstCrackClassifier once for all forward-pass tests."""
    return FirstCrackClassifier(device="cpu")


# ── Label mapping (no model load) ─────────────────────────────────────────────


class TestLabelMappings:
    def test_label2id_keys(self) -> None:
        assert set(LABEL2ID.keys()) == {"first_crack", "no_first_crack"}

    def test_id2label_is_inverse(self) -> None:
        for idx, label in ID2LABEL.items():
            assert LABEL2ID[label] == idx


# ── Feature extractor (no model load) ────────────────────────────────────────


class TestBuildFeatureExtractor:
    def test_returns_ast_extractor(self) -> None:
        fe = build_feature_extractor()
        assert isinstance(fe, ASTFeatureExtractor)

    def test_sampling_rate(self) -> None:
        fe = build_feature_extractor()
        assert fe.sampling_rate == 16000

    def test_num_mel_bins(self) -> None:
        fe = build_feature_extractor()
        assert fe    def test_forward_single(self, classifier: FirstCrackClassifier) -> None:
        audio = torch.randn(1, 16000 * 10)
        logits = classifier(audio)
        assert logits.shape == (1, 2)

    def test_forward_batch(self, classifier: FirstCrackClassifier) -> None:
        audio = torch.randn(4, 16000 * 10)
        logits = classifier(audio)
        assert logits.shape == (4, 2)

    def test_predict_proba_sums_to_one(self, classifier: FirstCrackClassifier) -> None:
        audio = torch.randn(2, 16000 * 10)
        probs = classifier.predict_proba(audio)
        assert probs.shape == (2, 2)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2))

    def test_predict_returns_class_index(self, classifier: FirstCrackClassifier) -> None:
        audio = torch.randn(3, 16000 * 10)
        preds = classifier.predict(audio)
        assert preds.shape == (3,)
        assert preds.max().item() <= 1
        assert preds.min().item() >= 0

    def test_1d_input_unsqueezed(self, classifier: FirstCrackClassifier) -> None:
        audio = torch.randn(16000 * 10)
        logits = classifier(audio)
        assert logits.shape == (1, 2)
