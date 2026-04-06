#!/usr/bin/env python3
"""Sync model card, dataset card, and Gradio Space files to HuggingFace Hub.

Uploads only the card/UI files — does not re-push model weights, dataset audio,
or ONNX artifacts.  Use ``scripts/push_to_hub.py`` for full artifact uploads.

Usage::

    # Sync everything (model card + dataset card + Space)
    python scripts/sync_hub_cards.py

    # Sync only the model card
    python scripts/sync_hub_cards.py --model-card

    # Sync only the dataset card
    python scripts/sync_hub_cards.py --dataset-card

    # Sync only the Gradio Space files
    python scripts/sync_hub_cards.py --space

    # Combine flags
    python scripts/sync_hub_cards.py --model-card --space
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODEL_REPO = "syamaner/coffee-first-crack-detection"
_DATASET_REPO = "syamaner/coffee-first-crack-audio"

# Local paths → (repo_id, repo_type, path_in_repo)
_MODEL_CARD = ("README.md", _MODEL_REPO, "model", "README.md")
_DATASET_CARD = ("data/DATASET_CARD.md", _DATASET_REPO, "dataset", "README.md")
_SPACE_FILES = [
    ("spaces/README.md", _MODEL_REPO, "space", "README.md"),
    ("spaces/app.py", _MODEL_REPO, "space", "app.py"),
    ("spaces/requirements.txt", _MODEL_REPO, "space", "requirements.txt"),
]


def _upload(
    api: object,
    local_path: str,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    commit_message: str,
) -> bool:
    """Upload a single file to HuggingFace Hub.

    Args:
        api: ``HfApi`` instance.
        local_path: Path to local file.
        repo_id: Target HF repository.
        repo_type: One of ``model``, ``dataset``, ``space``.
        path_in_repo: Destination filename in the repo.
        commit_message: Commit message for the upload.

    Returns:
        True if the upload succeeded, False otherwise.
    """
    p = Path(local_path)
    if not p.exists():
        print(f"  SKIP {local_path} — file not found")
        return False

    api.upload_file(  # type: ignore[attr-defined]
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    print(f"  OK   {local_path} → {repo_type}:{repo_id}/{path_in_repo}")
    return True


def sync_model_card(api: object) -> None:
    """Upload the model card README.md to the model repo."""
    local, repo_id, repo_type, dest = _MODEL_CARD
    _upload(api, local, repo_id, repo_type, dest, "Sync model card from GitHub")


def sync_dataset_card(api: object) -> None:
    """Upload the dataset card to the dataset repo (renamed to README.md)."""
    local, repo_id, repo_type, dest = _DATASET_CARD
    _upload(api, local, repo_id, repo_type, dest, "Sync dataset card from GitHub")


def sync_space(api: object) -> None:
    """Upload Gradio Space files (README, app.py, requirements.txt)."""
    for local, repo_id, repo_type, dest in _SPACE_FILES:
        _upload(api, local, repo_id, repo_type, dest, "Sync Space files from GitHub")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync card and Space files to HuggingFace Hub",
    )
    parser.add_argument("--model-card", action="store_true", help="Sync model card only")
    parser.add_argument("--dataset-card", action="store_true", help="Sync dataset card only")
    parser.add_argument("--space", action="store_true", help="Sync Gradio Space files only")
    args = parser.parse_args()

    # If no flags specified, sync everything
    sync_all = not any([args.model_card, args.dataset_card, args.space])

    from huggingface_hub import HfApi

    api = HfApi()

    # Verify auth
    try:
        user = api.whoami()["name"]
        print(f"Authenticated as: {user}")
    except Exception:
        print("ERROR: Not logged in to HuggingFace Hub.")
        print("  Run: huggingface-cli login")
        sys.exit(1)

    if sync_all or args.model_card:
        print("\n[Model card]")
        sync_model_card(api)

    if sync_all or args.dataset_card:
        print("\n[Dataset card]")
        sync_dataset_card(api)

    if sync_all or args.space:
        print("\n[Gradio Space]")
        sync_space(api)

    print("\nDone.")


if __name__ == "__main__":
    main()
