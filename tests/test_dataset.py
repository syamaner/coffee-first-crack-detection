"""Tests for dataset.py — FirstCrackDataset and filename metadata parsing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from coffee_first_crack.dataset import (
    FirstCrackDataset,
    generate_recordings_manifest,
    parse_filename_metadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_wav(path: Path, duration_sec: float = 10.0, sample_rate: int = 16000) -> None:
    """Write a silent WAV file at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    sf.write(str(path), samples, sample_rate)


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    """Create a minimal split directory with 3 first_crack and 3 no_first_crack WAVs."""
    for label in ("first_crack", "no_first_crack"):
        for i in range(3):
            _write_wav(tmp_path / label / f"chunk_{i:03d}.wav")
    return tmp_path


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Create a raw directory with mic-1 and mic-2 WAVs."""
    _write_wav(tmp_path / "mic2-brazil-roast1-03-04-26.wav")
    _write_wav(tmp_path / "mic1-costarica-hermosa-roast1.wav")
    _write_wav(tmp_path / "roast-1-costarica-hermosa-hp-a.wav")
    return tmp_path


# ── parse_filename_metadata ────────────────────────────────────────────────────


class TestParseFilenameMetadata:
    def test_new_convention_mic2_brazil(self) -> None:
        meta = parse_filename_metadata("mic2-brazil-roast1-03-04-26")
        assert meta["microphone"] == "mic-2-new"
        assert meta["coffee_origin"] == "brazil"

    def test_new_convention_mic1(self) -> None:
        meta = parse_filename_metadata("mic1-costarica-hermosa-roast3")
        assert meta["microphone"] == "mic-1-original"
        assert meta["coffee_origin"] == "costarica-hermosa"

    def test_legacy_costarica(self) -> None:
        meta = parse_filename_metadata("roast-1-costarica-hermosa-hp-a")
        assert meta["microphone"] == "mic-1-original"
        assert "costarica" in meta["coffee_origin"]

    def test_legacy_brazil(self) -> None:
        meta = parse_filename_metadata("roast1-19-10-2025-brazil")
        assert meta["microphone"] == "mic-1-original"
        assert meta["coffee_origin"] == "brazil"

    def test_at_roast_fallback(self) -> None:
        meta = parse_filename_metadata("at-roast1")
        assert meta["microphone"] == "mic-2-new"

    def test_unknown_fallback(self) -> None:
        meta = parse_filename_metadata("random-file-name")
        assert "microphone" in meta
        assert "coffee_origin" in meta


# ── FirstCrackDataset ─────────────────────────────────────────────────────────


class TestFirstCrackDataset:
    def test_loads_correct_count(self, split_dir: Path) -> None:
        ds = FirstCrackDataset(split_dir)
        assert len(ds) == 6

    def test_returns_tensor_and_label(self, split_dir: Path) -> None:
        ds = FirstCrackDataset(split_dir)
        audio, label = ds[0]
        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 1
        assert label in (0, 1)

    def test_target_length_padding(self, tmp_path: Path) -> None:
        # Write a 5s file (shorter than target 10s)
        _write_wav(tmp_path / "first_crack" / "short.wav", duration_sec=5.0)
        ds = FirstCrackDataset(tmp_path, target_length=10)
        audio, _ = ds[0]
        assert audio.shape[0] == 16000 * 10

    def test_crop_modes(self, split_dir: Path) -> None:
        for mode in ("start", "center", "random"):
            ds = FirstCrackDataset(split_dir, crop_mode=mode)
            audio, _ = ds[0]
            assert audio.shape[0] == 16000 * 10

    def test_class_weights_shape(self, split_dir: Path) -> None:
        ds = FirstCrackDataset(split_dir)
        weights = ds.get_class_weights()
        assert weights.shape == (2,)
        assert (weights > 0).all()

    def test_statistics_keys(self, split_dir: Path) -> None:
        ds = FirstCrackDataset(split_dir)
        stats = ds.get_statistics()
        assert "total_samples" in stats
        assert stats["total_samples"] == 6

    def test_raises_on_empty_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No WAV files"):
            FirstCrackDataset(tmp_path / "empty")


# ── generate_recordings_manifest ──────────────────────────────────────────────


class TestGenerateRecordingsManifest:
    def test_creates_csv(self, raw_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "recordings.csv"
        generate_recordings_manifest(raw_dir, out)
        assert out.exists()
        content = out.read_text()
        assert "filename" in content
        assert "microphone" in content

    def test_row_count(self, raw_dir: Path, tmp_path: Path) -> None:
        import csv
        out = tmp_path / "recordings.csv"
        generate_recordings_manifest(raw_dir, out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3  # 3 WAV files in raw_dir fixture
