"""Tests for immutable training-data and ONNX lineage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from coffee_first_crack import training_provenance
from coffee_first_crack.training_provenance import (
    bind_onnx_artifact,
    sha256_file,
    snapshot_training_data,
    validate_onnx_training_provenance,
)


def _provenance_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create one local experiment and its exact dataset evidence."""
    evidence = {
        "split_integrity": tmp_path / "data" / "split_integrity.json",
        "chunk_manifest": tmp_path / "data" / "chunk_manifest.jsonl",
        "dataset_capture_manifest": tmp_path / "data" / "capture_manifest.json",
    }
    for name, path in evidence.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    experiment = tmp_path / "experiments" / "candidate"
    snapshot_path = experiment / "training_data_provenance.json"
    snapshot_training_data(
        experiment_name="candidate",
        split_integrity=evidence["split_integrity"],
        chunk_manifest=evidence["chunk_manifest"],
        dataset_capture_manifest=evidence["dataset_capture_manifest"],
        output=snapshot_path,
    )
    checkpoint = experiment / "checkpoint-best"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    artifact_dir = tmp_path / "exports" / "int8"
    artifact_dir.mkdir(parents=True)
    onnx_model = artifact_dir / "model_quantized.onnx"
    onnx_model.write_bytes(b"onnx")
    preprocessor = artifact_dir / "preprocessor_config.json"
    preprocessor.write_text("{}", encoding="utf-8")
    bound_path = artifact_dir / "training_provenance.json"
    bind_onnx_artifact(
        training_data_snapshot=snapshot_path,
        model_dir=checkpoint,
        onnx_model=onnx_model,
        preprocessor_config=preprocessor,
        output=bound_path,
    )
    return {
        **evidence,
        "snapshot": snapshot_path,
        "checkpoint": checkpoint,
        "onnx_model": onnx_model,
        "preprocessor": preprocessor,
        "bound": bound_path,
    }


def test_bound_artifact_validates_against_exact_dataset_evidence(tmp_path: Path) -> None:
    """A local checkpoint and ONNX digest retain the pre-training data snapshot."""
    paths = _provenance_fixture(tmp_path)
    evidence_hashes = {
        name: sha256_file(paths[name])
        for name in ("split_integrity", "chunk_manifest", "dataset_capture_manifest")
    }

    provenance = validate_onnx_training_provenance(
        provenance_path=paths["bound"],
        onnx_sha256=sha256_file(paths["onnx_model"]),
        preprocessor_sha256=sha256_file(paths["preprocessor"]),
        dataset_evidence_sha256=evidence_hashes,
    )

    assert provenance["experiment_name"] == "candidate"
    assert provenance["checkpoint"]["files"]["model.safetensors"]["sha256"] == sha256_file(
        paths["checkpoint"] / "model.safetensors"
    )


def test_training_snapshot_cannot_be_overwritten(tmp_path: Path) -> None:
    """A rerun cannot silently retarget an existing experiment to new data."""
    paths = _provenance_fixture(tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite provenance"):
        snapshot_training_data(
            experiment_name="candidate",
            split_integrity=paths["split_integrity"],
            chunk_manifest=paths["chunk_manifest"],
            dataset_capture_manifest=paths["dataset_capture_manifest"],
            output=paths["snapshot"],
        )


@pytest.mark.parametrize("changed", ["onnx_model", "preprocessor", "split_integrity"])
def test_validation_rejects_artifact_or_dataset_mismatch(tmp_path: Path, changed: str) -> None:
    """Neither replacement model bytes nor different split evidence can pass replay admission."""
    paths = _provenance_fixture(tmp_path)
    onnx_sha = sha256_file(paths["onnx_model"])
    preprocessor_sha = sha256_file(paths["preprocessor"])
    evidence_hashes = {
        name: sha256_file(paths[name])
        for name in ("split_integrity", "chunk_manifest", "dataset_capture_manifest")
    }
    if changed == "onnx_model":
        onnx_sha = "0" * 64
    elif changed == "preprocessor":
        preprocessor_sha = "0" * 64
    else:
        evidence_hashes["split_integrity"] = "0" * 64

    with pytest.raises(ValueError, match="does not match"):
        validate_onnx_training_provenance(
            provenance_path=paths["bound"],
            onnx_sha256=onnx_sha,
            preprocessor_sha256=preprocessor_sha,
            dataset_evidence_sha256=evidence_hashes,
        )


def test_bound_provenance_records_exact_snapshot_digest(tmp_path: Path) -> None:
    """The artifact declaration itself identifies the immutable pre-training snapshot."""
    paths = _provenance_fixture(tmp_path)
    value = json.loads(paths["bound"].read_text(encoding="utf-8"))

    assert value["training_data_snapshot"]["sha256"] == sha256_file(paths["snapshot"])


def test_snapshot_rejects_invalid_name_or_missing_input(tmp_path: Path) -> None:
    """The pre-training boundary fails closed before writing partial provenance."""
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="trimmed"):
        snapshot_training_data(
            experiment_name=" candidate ",
            split_integrity=missing,
            chunk_manifest=missing,
            dataset_capture_manifest=missing,
            output=tmp_path / "snapshot.json",
        )
    with pytest.raises(ValueError, match="not a regular file"):
        sha256_file(missing)


@pytest.mark.parametrize("mutation", ["kind", "evidence", "changed", "checkpoint"])
def test_binding_rejects_untrusted_snapshot_or_checkpoint(tmp_path: Path, mutation: str) -> None:
    """Malformed, stale, or cross-experiment inputs cannot be bound to ONNX."""
    paths = _provenance_fixture(tmp_path)
    snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
    model_dir = paths["checkpoint"]
    if mutation == "kind":
        snapshot["kind"] = "other"
    elif mutation == "evidence":
        snapshot["dataset_evidence"].pop("chunk_manifest")
    elif mutation == "changed":
        paths["chunk_manifest"].write_text("changed\n", encoding="utf-8")
    else:
        model_dir = tmp_path / "outside-checkpoint"
        model_dir.mkdir()
    paths["snapshot"].write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError):
        bind_onnx_artifact(
            training_data_snapshot=paths["snapshot"],
            model_dir=model_dir,
            onnx_model=paths["onnx_model"],
            preprocessor_config=paths["preprocessor"],
            output=tmp_path / "rebound.json",
        )


@pytest.mark.parametrize("mutation", ["kind", "artifact", "evidence", "evaluator"])
def test_validation_rejects_malformed_or_incomplete_provenance(
    tmp_path: Path, mutation: str
) -> None:
    """Replay admission validates both provenance schema and evaluator evidence shape."""
    paths = _provenance_fixture(tmp_path)
    bound = json.loads(paths["bound"].read_text(encoding="utf-8"))
    evidence_hashes = {
        name: sha256_file(paths[name])
        for name in ("split_integrity", "chunk_manifest", "dataset_capture_manifest")
    }
    if mutation == "kind":
        bound["kind"] = "other"
    elif mutation == "artifact":
        bound["artifact"] = "malformed"
    elif mutation == "evidence":
        bound["dataset_evidence"].pop("chunk_manifest")
    else:
        evidence_hashes.pop("chunk_manifest")
    paths["bound"].write_text(json.dumps(bound), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_onnx_training_provenance(
            provenance_path=paths["bound"],
            onnx_sha256=sha256_file(paths["onnx_model"]),
            preprocessor_sha256=sha256_file(paths["preprocessor"]),
            dataset_evidence_sha256=evidence_hashes,
        )


def test_training_snapshot_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The rebuild script's module command writes the requested immutable snapshot."""
    inputs = [tmp_path / name for name in ("split.json", "chunks.jsonl", "captures.json")]
    for path in inputs:
        path.write_text(path.name, encoding="utf-8")
    output = tmp_path / "experiment" / "training_data_provenance.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_provenance",
            "--experiment-name",
            "candidate",
            "--split-integrity",
            str(inputs[0]),
            "--chunk-manifest",
            str(inputs[1]),
            "--dataset-capture-manifest",
            str(inputs[2]),
            "--output",
            str(output),
        ],
    )

    training_provenance.main()

    assert json.loads(output.read_text(encoding="utf-8"))["experiment_name"] == "candidate"
