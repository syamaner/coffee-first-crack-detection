"""Tests for ONNX export provenance binding."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from coffee_first_crack import export_onnx as export_module


def test_export_binds_each_onnx_variant_to_training_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provenance-aware export emits lineage beside the deployable artifact."""

    class FakeOrtModel:
        """Minimal Optimum export stand-in."""

        def save_pretrained(self, output: str) -> None:
            Path(output, "model.onnx").write_bytes(b"onnx")

    class FakeOrtFactory:
        """Minimal dynamic import target."""

        @classmethod
        def from_pretrained(cls, model_dir: str, *, export: bool) -> FakeOrtModel:
            assert model_dir.endswith("checkpoint-best")
            assert export is True
            return FakeOrtModel()

    class FakeFeatureExtractor:
        """Write the file consumed by provenance binding."""

        def save_pretrained(self, output: str) -> None:
            Path(output, "preprocessor_config.json").write_text("{}", encoding="utf-8")

    fake_optimum = ModuleType("optimum.onnxruntime")
    fake_optimum.ORTModelForAudioClassification = FakeOrtFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", fake_optimum)
    monkeypatch.setattr(export_module, "build_feature_extractor", FakeFeatureExtractor)
    calls: list[dict[str, Path]] = []

    def fake_bind(
        *,
        training_data_snapshot: Path,
        model_dir: Path,
        onnx_model: Path,
        preprocessor_config: Path,
        output: Path,
    ) -> None:
        calls.append(
            {
                "training_data_snapshot": training_data_snapshot,
                "model_dir": model_dir,
                "onnx_model": onnx_model,
                "preprocessor_config": preprocessor_config,
                "output": output,
            }
        )
        output.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(export_module, "bind_onnx_artifact", fake_bind)
    checkpoint = tmp_path / "experiments" / "candidate" / "checkpoint-best"
    checkpoint.mkdir(parents=True)
    snapshot = checkpoint.parent / "training_data_provenance.json"
    snapshot.write_text("{}", encoding="utf-8")

    results = export_module.export_onnx(
        model_dir=checkpoint,
        output_dir=tmp_path / "exports",
        quantize=False,
        training_data_provenance=snapshot,
    )

    assert results == {"fp32": tmp_path / "exports" / "fp32" / "model.onnx"}
    assert len(calls) == 1
    assert calls[0]["training_data_snapshot"] == snapshot
    assert calls[0]["model_dir"] == checkpoint
    assert calls[0]["output"].name == "training_provenance.json"
    assert calls[0]["output"].is_file()
