"""PyTorch Dataset and HuggingFace Datasets integration for first crack detection.

Supports:
- Local audio chunks organised as ``data/splits/{train,val,test}/{first_crack,no_first_crack}/``
- HuggingFace dataset loading via ``load_dataset("syamaner/coffee-first-crack-audio")``
- Filename-based metadata extraction (microphone, coffee origin, roast number)
- Auto-generation of ``data/recordings.csv`` manifest
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from coffee_first_crack.data_prep.corpus_manifest import (
    load_capture_manifest,
    resolve_within,
)

# ── Filename metadata ────────────────────────────────────────────────────────

# New convention:  {mic}-{origin}-{roast-number}.wav
#   e.g.  mic2-brazil-roast1.wav   mic1-costarica-hermosa-roast3.wav
_NEW_PATTERN = re.compile(
    r"^(?P<mic>mic\d+)-(?P<origin>[a-z0-9\-]+?)-roast(?P<num>\d+)",
    re.IGNORECASE,
)

# Legacy mapping for old filenames that don't follow the new convention
_LEGACY_METADATA: dict[str, dict[str, str]] = {
    # roast-1-costarica-hermosa-hp-a.wav  etc.
    "roast-1-costarica": {"microphone": "mic-1-original", "coffee_origin": "costarica-hermosa"},
    "roast-2-costarica": {"microphone": "mic-1-original", "coffee_origin": "costarica-hermosa"},
    "roast-3-costarica": {"microphone": "mic-1-original", "coffee_origin": "costarica-hermosa"},
    "roast-4-costarica": {"microphone": "mic-1-original", "coffee_origin": "costarica-hermosa"},
    "roast1-19-10-2025-brazil": {"microphone": "mic-1-original", "coffee_origin": "brazil"},
    "roast2-19-10-2025-brazil": {"microphone": "mic-1-original", "coffee_origin": "brazil"},
}


def parse_filename_metadata(stem: str) -> dict[str, str]:
    """Extract microphone and coffee-origin metadata from a WAV filename stem.

    Supports the new convention (``mic{n}-{origin}-roast{n}``) and falls back
    to the legacy mapping table for older filenames.

    Args:
        stem: Filename without extension, e.g. ``"mic2-brazil-roast1"``.

    Returns:
        Dict with keys ``microphone`` and ``coffee_origin``. Returns
        ``"unknown"`` values when the filename cannot be parsed.
    """
    m = _NEW_PATTERN.match(stem)
    if m:
        mic_num = m.group("mic").lower().replace("mic", "")
        label = "mic-2-new" if mic_num != "1" else "mic-1-original"
        return {
            "microphone": label,
            "coffee_origin": m.group("origin").lower(),
        }

    # Try legacy table (prefix match)
    for prefix, meta in _LEGACY_METADATA.items():
        if stem.startswith(prefix):
            return dict(meta)

    # Default fallback — assume legacy mic-1
    is_at_prefix = stem.lower().startswith("at-roast")
    return {
        "microphone": "mic-2-new" if is_at_prefix else "mic-1-original",
        "coffee_origin": "unknown",
    }


# ── Dataset ───────────────────────────────────────────────────────────────────


class FirstCrackDataset(Dataset):
    """PyTorch Dataset for binary first crack audio classification.

    Loads 10-second WAV chunks, resamples to 16 kHz mono, and applies
    optional cropping strategies.

    Directory layout expected under ``data_dir``::

        data_dir/
          first_crack/       ← positive class (label 1)
          no_first_crack/    ← negative class (label 0)

    Args:
        data_dir: Directory containing ``first_crack/`` and ``no_first_crack/`` subdirs.
        sample_rate: Target sample rate in Hz (default: 16 000).
        target_length: Target window length in seconds (default: 10).
        transform: Optional transform applied to the raw waveform tensor.
        crop_mode: Cropping strategy when audio > ``target_length``.
            One of ``"random"`` (training), ``"center"`` (evaluation), ``"start"``.
    """

    LABEL2IDX: dict[str, int] = {"no_first_crack": 0, "first_crack": 1}
    IDX2LABEL: dict[int, str] = {v: k for k, v in LABEL2IDX.items()}

    def __init__(
        self,
        data_dir: Path | str,
        sample_rate: int = 16000,
        target_length: int = 10,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        crop_mode: str = "start",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.target_samples = int(sample_rate * target_length)
        self.transform = transform
        self.crop_mode = crop_mode

        self.samples: list[tuple[Path, int]] = []
        for label_name, label_idx in self.LABEL2IDX.items():
            label_dir = self.data_dir / label_name
            if label_dir.exists():
                for audio_file in sorted(label_dir.glob("*.wav")):
                    self.samples.append((audio_file, label_idx))

        if not self.samples:
            raise ValueError(f"No WAV files found under {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return a single (waveform, label) pair.

        Returns:
            Tuple of float32 waveform tensor of shape ``(target_samples,)``
            and integer label.
        """
        audio_path, label = self.samples[idx]
        audio, _sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        audio = self._pad_or_crop(audio)
        tensor = torch.FloatTensor(audio)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label

    def _pad_or_crop(self, audio: np.ndarray) -> np.ndarray:
        """Pad (with zeros) or crop audio to exactly ``target_samples`` frames."""
        n, t = len(audio), self.target_samples
        if n < t:
            return np.pad(audio, (0, t - n), mode="constant")
        if n == t:
            return audio
        # Crop
        if self.crop_mode == "start":
            start = 0
        elif self.crop_mode == "center":
            start = (n - t) // 2
        elif self.crop_mode == "random":
            max_start = n - t
            start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        else:
            start = 0
        return audio[start : start + t]

    # ── Helper methods ────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced data.

        Returns:
            Float tensor of shape ``(2,)`` with weights for
            ``[no_first_crack, first_crack]``.
        """
        counts: dict[int, int] = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        total = len(self.samples)
        return torch.FloatTensor(
            [
                total / (2 * counts[0]),
                total / (2 * counts[1]),
            ]
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return basic dataset statistics."""
        counts: dict[int, int] = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return {
            "total_samples": len(self.samples),
            "no_first_crack": counts[0],
            "first_crack": counts[1],
            "class_ratio": counts[1] / counts[0] if counts[0] > 0 else 0.0,
            "sample_rate": self.sample_rate,
            "target_length_sec": self.target_length,
            "target_samples": self.target_samples,
        }

    def get_label_name(self, idx: int) -> str:
        """Convert a label index to its string name."""
        return self.IDX2LABEL[idx]


# ── DataLoader factory ────────────────────────────────────────────────────────


def create_dataloaders(
    train_dir: Path | str,
    val_dir: Path | str,
    test_dir: Path | str | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    sample_rate: int = 16000,
    target_length: int = 10,
    train_crop_mode: str = "random",
    eval_crop_mode: str = "center",
) -> tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and optionally test DataLoaders.

    Args:
        train_dir: Training data directory.
        val_dir: Validation data directory.
        test_dir: Optional test data directory.
        batch_size: Batch size (default: 8).
        num_workers: DataLoader worker count (use 0 for MPS).
        pin_memory: Enable pin_memory (use ``False`` for MPS/CPU).
        sample_rate: Target sample rate in Hz.
        target_length: Window length in seconds.
        train_crop_mode: Crop strategy for training (``"random"``).
        eval_crop_mode: Crop strategy for val/test (``"center"``).

    Returns:
        ``(train_loader, val_loader)`` or ``(train_loader, val_loader, test_loader)``.
    """
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}

    train_dataset = FirstCrackDataset(
        train_dir, sample_rate, target_length, crop_mode=train_crop_mode
    )
    val_dataset = FirstCrackDataset(val_dir, sample_rate, target_length, crop_mode=eval_crop_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    if test_dir is not None:
        test_dataset = FirstCrackDataset(
            test_dir, sample_rate, target_length, crop_mode=eval_crop_mode
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


# ── recordings.csv manifest ───────────────────────────────────────────────────


def generate_recordings_manifest(raw_dir: Path | str, output_path: Path | str) -> Path:
    """Generate ``recordings.csv`` from legacy files and staged MCP manifests.

    MCP rows use their capture-manifest metadata rather than parsing the
    UUID-prefixed staged filename. Their ``filename`` is relative to
    ``raw_dir`` so collision-free staged identities remain resolvable.

    Args:
        raw_dir: Directory containing raw ``.wav`` files.
        output_path: Path to write the CSV file.

    Returns:
        Path to the written CSV file.
    """
    raw_dir = Path(raw_dir).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen_filenames: set[str] = set()
    for wav_file in sorted(raw_dir.glob("*.wav")):
        meta = parse_filename_metadata(wav_file.stem)
        try:
            duration = librosa.get_duration(path=str(wav_file))
        except Exception:
            duration = 0.0
        filename = wav_file.name
        seen_filenames.add(filename)
        rows.append(
            {
                "filename": filename,
                "microphone": meta["microphone"],
                "coffee_origin": meta["coffee_origin"],
                "duration_sec": f"{duration:.2f}",
                "pair_id": "",
                "roast_num": "",
                "mic_num": "",
                "original_filename": filename,
                "notes": "legacy_or_single_stream",
            }
        )

    for manifest_path in sorted(raw_dir.rglob("capture_manifest.json")):
        if manifest_path.is_symlink():
            raise ValueError(f"Capture manifest may not be a symlink: {manifest_path}")
        manifest = load_capture_manifest(manifest_path)
        staging_root = Path(manifest["staging_root"]).resolve()
        if staging_root != manifest_path.parent.resolve():
            raise ValueError(
                f"Capture manifest staging_root does not match its directory: {manifest_path}"
            )
        for session in manifest["sessions"]:
            for stream in session["streams"]:
                wav_file = resolve_within(staging_root, stream["staged_relative_path"])
                if not wav_file.is_file() or wav_file.is_symlink():
                    raise FileNotFoundError(
                        f"Staged manifest WAV must be a regular non-symlink file: {wav_file}"
                    )
                try:
                    filename = wav_file.relative_to(raw_dir).as_posix()
                except ValueError as exc:
                    raise ValueError(f"Staged MCP WAV escapes raw directory: {wav_file}") from exc
                if filename in seen_filenames:
                    raise ValueError(f"Duplicate recording filename in catalog: {filename}")
                seen_filenames.add(filename)
                rows.append(
                    {
                        "filename": filename,
                        "microphone": stream["label"],
                        "coffee_origin": session["origin"],
                        "duration_sec": f"{stream['duration_seconds']:.2f}",
                        "pair_id": session["pair_id"],
                        "roast_num": str(session["roast_num"]),
                        "mic_num": str(stream["mic_num"]),
                        "original_filename": stream["original_filename"],
                        "notes": "staged_mcp_capture",
                    }
                )

    fieldnames = [
        "filename",
        "microphone",
        "coffee_origin",
        "duration_sec",
        "pair_id",
        "roast_num",
        "mic_num",
        "original_filename",
        "notes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {output_path}")
    return output_path
