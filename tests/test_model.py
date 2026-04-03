"""Tests for model.py — build_model, build_feature_extractor, FirstCrackClassifier."""

from __future__ import annotations

import torch
import pytest
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from coffee_first_crack.model import (
    LABEL2ID,
    ID2LABEL,
    build_feature_extractor,
    build_model,
    FirstCrackClassifier,
)


class TestLabelMappings:
    def test_label2id_keys(self) -> None:
        assert set(LABEL2ID.keys()) == {"first_crack", "no_first_crack"}

    def test_id2label_is_inverse(self) -> None:
        for idx, label in ID2LABEL.items():
            assert LABEL2ID[label] == idx


class TestBuildFeatureExtractor:
    def test_returns_ast_extractor(self) -> None:
        fe = build_feature_extractor()
        assert isinstance(fe, ASTFeatureExtractor)

    def test_sampling_rate(self) -> None:
        fe = build_feature_extractor()
        assert fe.sampling_rate == 16000

    def test_num_mel_bins(self) -> None:
        fe = build_feature_extractor()
        assert fe.num_mel_bins == 128

    def test_do_normalize(self) -> None:
        fe = build_feature_extractor()
        assert fe.do_normalize is True


class TestBuildModel:
    def test_returns_ast_model(self) -> None:
        model = build_model(device="cpu")
        assert isinstance(model, ASTForAudioClassification)

    def test_num_labels(self) -> None:
        model = build_model(device="cpu")
        assert model.config.num_labels == 2

    def test_label_mapping(self) -> None:
        model = build_model(device="cpu")
        assert model.config.label2id["first_crack"] == 1
        assert model.config.label2id["no_first_crack"] == 0

    def test_on_cpu(self) -> None:
        model = build_model(device="cpu")
        param = next(model.parameters())
        assert param.device.type == "cpu"


class TestFirstCrackClassifier:
    @pytest.fixture()
    def classifier(self) -> FirstCrackClassifier:
        return FirstCrackClassifier(device="cpu")

    def test_forward_single(self, classifier: FirstCrackClassifier) -> None:
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
