"""Immutable dataset and model lineage for decisive held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_EVIDENCE_NAMES = (
    "split_integrity",
    "chunk_manifest",
    "dataset_capture_manifest",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Provenance input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with provenance-specific error context."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read provenance JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Provenance JSON must contain an object: {path}")
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Write a new immutable-intent provenance object without overwriting."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite provenance: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def snapshot_training_data(
    *,
    experiment_name: str,
    split_integrity: Path,
    chunk_manifest: Path,
    dataset_capture_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze the exact dataset evidence immediately before model training.

    Args:
        experiment_name: Training run name whose directory owns the snapshot.
        split_integrity: Machine-checkable pair-level split report.
        chunk_manifest: Per-window source and pair manifest.
        dataset_capture_manifest: Staged MCP capture manifest.
        output: New snapshot JSON path inside the experiment directory.

    Returns:
        The written training-data snapshot object.
    """
    if not experiment_name or experiment_name != experiment_name.strip():
        raise ValueError("experiment_name must be a non-empty trimmed string")
    inputs = {
        "split_integrity": split_integrity,
        "chunk_manifest": chunk_manifest,
        "dataset_capture_manifest": dataset_capture_manifest,
    }
    evidence = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for name, path in inputs.items()
    }
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "kind": "training_data_snapshot",
        "experiment_name": experiment_name,
        "dataset_evidence": evidence,
    }
    _write_new_json(output, snapshot)
    return snapshot


def bind_onnx_artifact(
    *,
    training_data_snapshot: Path,
    model_dir: Path,
    onnx_model: Path,
    preprocessor_config: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind exported ONNX bytes to a pre-training dataset snapshot.

    Args:
        training_data_snapshot: Snapshot created before the training run.
        model_dir: Local checkpoint directory exported to ONNX.
        onnx_model: Exported ONNX model file.
        preprocessor_config: Exported preprocessing configuration.
        output: New bound provenance JSON beside the ONNX artifact.

    Returns:
        The written artifact-provenance object.
    """
    snapshot_path = training_data_snapshot.resolve()
    snapshot = _read_json(snapshot_path)
    if snapshot.get("schema_version") != 1 or snapshot.get("kind") != "training_data_snapshot":
        raise ValueError(f"Invalid training-data snapshot: {snapshot_path}")
    evidence = snapshot.get("dataset_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(DATASET_EVIDENCE_NAMES):
        raise ValueError(f"Incomplete training-data snapshot: {snapshot_path}")
    for name in DATASET_EVIDENCE_NAMES:
        item = evidence[name]
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or sha256_file(Path(item["path"])) != item["sha256"]
        ):
            raise ValueError(f"Training-data evidence changed before export: {name}")

    resolved_model_dir = model_dir.resolve()
    experiment_root = snapshot_path.parent.resolve()
    if not resolved_model_dir.is_dir() or not resolved_model_dir.is_relative_to(experiment_root):
        raise ValueError(
            f"Model checkpoint is not owned by the snapshotted experiment: {resolved_model_dir}"
        )
    checkpoint_files = {}
    for filename in ("config.json", "model.safetensors"):
        path = resolved_model_dir / filename
        checkpoint_files[filename] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "kind": "onnx_training_provenance",
        "experiment_name": snapshot.get("experiment_name"),
        "training_data_snapshot": {
            "path": str(snapshot_path),
            "sha256": sha256_file(snapshot_path),
        },
        "dataset_evidence": evidence,
        "checkpoint": {
            "path": str(resolved_model_dir),
            "files": checkpoint_files,
        },
        "artifact": {
            "onnx_sha256": sha256_file(onnx_model.resolve()),
            "preprocessor_sha256": sha256_file(preprocessor_config.resolve()),
        },
    }
    _write_new_json(output, provenance)
    return provenance


def validate_onnx_training_provenance(
    *,
    provenance_path: Path,
    onnx_sha256: str,
    preprocessor_sha256: str,
    dataset_evidence_sha256: dict[str, str],
) -> dict[str, Any]:
    """Validate an ONNX artifact against the evaluator's exact dataset evidence."""
    if set(dataset_evidence_sha256) != set(DATASET_EVIDENCE_NAMES):
        raise ValueError("Evaluator supplied incomplete dataset evidence")
    provenance = _read_json(provenance_path)
    if (
        provenance.get("schema_version") != 1
        or provenance.get("kind") != "onnx_training_provenance"
    ):
        raise ValueError(f"Invalid ONNX training provenance: {provenance_path}")
    artifact = provenance.get("artifact")
    evidence = provenance.get("dataset_evidence")
    if not isinstance(artifact, dict) or not isinstance(evidence, dict):
        raise ValueError(f"Malformed ONNX training provenance: {provenance_path}")
    if artifact.get("onnx_sha256") != onnx_sha256:
        raise ValueError("ONNX model does not match its training provenance")
    if artifact.get("preprocessor_sha256") != preprocessor_sha256:
        raise ValueError("Preprocessor does not match its training provenance")
    if set(evidence) != set(DATASET_EVIDENCE_NAMES):
        raise ValueError("Training provenance has incomplete dataset evidence")
    for name in DATASET_EVIDENCE_NAMES:
        item = evidence[name]
        if not isinstance(item, dict) or item.get("sha256") != dataset_evidence_sha256[name]:
            raise ValueError(f"Training provenance does not match evaluator evidence: {name}")
    return provenance


def main() -> None:
    """Create the pre-training snapshot used by the rebuild pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split-integrity", type=Path, required=True)
    parser.add_argument("--chunk-manifest", type=Path, required=True)
    parser.add_argument("--dataset-capture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot_training_data(
        experiment_name=args.experiment_name,
        split_integrity=args.split_integrity,
        chunk_manifest=args.chunk_manifest,
        dataset_capture_manifest=args.dataset_capture_manifest,
        output=args.output,
    )
    print(f"Frozen pre-training dataset provenance: {args.output}")


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    main()
