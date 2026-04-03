#!/usr/bin/env python3
"""Push model, dataset, and ONNX artifacts to HuggingFace Hub.

Usage::

    # Push model only
    python scripts/push_to_hub.py \\
        --model-dir experiments/baseline_v1/checkpoint-best \\
        --repo-id syamaner/coffee-first-crack-detection

    # Push dataset
    python scripts/push_to_hub.py \\
        --dataset-dir data/splits \\
        --recordings-csv data/recordings.csv \\
        --dataset-repo-id syamaner/coffee-first-crack-audio

    # Push ONNX artifacts
    python scripts/push_to_hub.py \\
        --onnx-dir exports/onnx \\
        --repo-id syamaner/coffee-first-crack-detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _validate_model_card(model_dir: Path) -> None:
    """Validate README.md YAML frontmatter before pushing.

    Args:
        model_dir: Checkpoint directory expected to contain a README.md.

    Raises:
        ValueError: If the README is missing required frontmatter fields.
    """
    import yaml

    readme = model_dir / "README.md"
    # Fall back to repo-root README
    if not readme.exists():
        readme = Path("README.md")
    if not readme.exists():
        print("Warning: README.md not found — skipping model card validation")
        return

    content = readme.read_text()
    if not content.startswith("---"):
        raise ValueError("README.md is missing YAML frontmatter (expected '---' at start)")

    # Extract YAML block between first pair of ---
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("README.md frontmatter is malformed (missing closing '---')")

    metadata = yaml.safe_load(parts[1])
    required = {"pipeline_tag", "license", "base_model"}
    missing = required - set(metadata or {})
    if missing:
        raise ValueError(f"README.md frontmatter missing required fields: {missing}")

    print("Model card YAML frontmatter is valid.")


def push_model(model_dir: Path, repo_id: str) -> None:
    """Push model and feature extractor to HuggingFace Hub."""
    from transformers import ASTForAudioClassification, ASTFeatureExtractor

    _validate_model_card(model_dir)
    print(f"Loading model from {model_dir}...")
    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    extractor = ASTFeatureExtractor.from_pretrained(str(model_dir))

    print(f"Pushing model to {repo_id}...")
    model.push_to_hub(repo_id)
    extractor.push_to_hub(repo_id)
    print(f"Model pushed to https://huggingface.co/{repo_id}")


def push_dataset(
    dataset_dir: Path,
    recordings_csv: Path,
    dataset_repo_id: str,
) -> None:
    """Push audio dataset with metadata to HuggingFace Datasets Hub."""
    import csv
    from datasets import Audio, Dataset, DatasetDict

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    meta_by_stem: dict[str, dict] = {}

    # Load recordings metadata
    if recordings_csv.exists():
        with recordings_csv.open() as f:
            for row in csv.DictReader(f):
                stem = Path(row["filename"]).stem
                meta_by_stem[stem] = row

    # Collect samples per split
    for split_name in splits:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue
        for label in ("first_crack", "no_first_crack"):
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for wav in sorted(label_dir.glob("*.wav")):
                # Attempt to find original recording stem from chunk name
                # Chunk names like: roast-1-costarica_chunk_001.wav
                orig_stem = wav.stem.rsplit("_chunk_", 1)[0] if "_chunk_" in wav.stem else wav.stem
                meta = meta_by_stem.get(orig_stem, {})
                splits[split_name].append({
                    "audio": str(wav),
                    "label": label,
                    "label_id": 1 if label == "first_crack" else 0,
                    "microphone": meta.get("microphone", "unknown"),
                    "coffee_origin": meta.get("coffee_origin", "unknown"),
                })

    dataset_dict = DatasetDict({
        split: Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
        for split, rows in splits.items()
        if rows
    })

    print(f"Pushing dataset to {dataset_repo_id}...")
    dataset_dict.push_to_hub(dataset_repo_id)
    print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_repo_id}")


def push_onnx(onnx_dir: Path, repo_id: str) -> None:
    """Upload ONNX files as model card attachments."""
    from huggingface_hub import HfApi

    api = HfApi()
    onnx_files = list(onnx_dir.rglob("*.onnx"))
    if not onnx_files:
        print(f"No .onnx files found in {onnx_dir}")
        return

    for onnx_file in onnx_files:
        path_in_repo = f"onnx/{onnx_file.name}"
        print(f"Uploading {onnx_file.name} → {repo_id}/{path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(onnx_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
    print("ONNX artifacts uploaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model/dataset/ONNX to HuggingFace Hub")
    parser.add_argument("--model-dir", type=Path, help="Checkpoint directory")
    parser.add_argument("--repo-id", type=str, default="syamaner/coffee-first-crack-detection")
    parser.add_argument("--dataset-dir", type=Path, help="data/splits directory")
    parser.add_argument("--recordings-csv", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--dataset-repo-id", type=str, default="syamaner/coffee-first-crack-audio")
    parser.add_argument("--onnx-dir", type=Path, help="exports/onnx directory")
    args = parser.parse_args()

    if not any([args.model_dir, args.dataset_dir, args.onnx_dir]):
        parser.print_help()
        sys.exit(1)

    if args.model_dir:
        push_model(args.model_dir, args.repo_id)
    if args.dataset_dir:
        push_dataset(args.dataset_dir, args.recordings_csv, args.dataset_repo_id)
    if args.onnx_dir:
        push_onnx(args.onnx_dir, args.repo_id)


if __name__ == "__main__":
    main()
