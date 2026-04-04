"""Tests for the data_prep module — chunking logic and dataset splitting utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import coffee_first_crack.data_prep.convert_labelstudio_export as convert_labelstudio_export
from coffee_first_crack.data_prep.chunk_audio import (
    chunk_recording,
    compute_overlap,
    label_window,
)
from coffee_first_crack.data_prep.convert_labelstudio_export import (
    convert_task,
    strip_hash_prefix,
)
from coffee_first_crack.data_prep.dataset_splitter import (
    extract_recording_stem,
    recording_level_split,
)

# ---------------------------------------------------------------------------
# compute_overlap
# ---------------------------------------------------------------------------


class TestComputeOverlap:
    """Tests for the overlap calculation between a window and annotated regions."""

    def test_full_overlap(self) -> None:
        """Window entirely inside a first_crack region."""
        regions = [{"start_time": 0.0, "end_time": 20.0, "label": "first_crack"}]
        assert compute_overlap(5.0, 15.0, regions) == pytest.approx(10.0)

    def test_no_overlap(self) -> None:
        """Window completely outside any first_crack region."""
        regions = [{"start_time": 100.0, "end_time": 200.0, "label": "first_crack"}]
        assert compute_overlap(0.0, 10.0, regions) == pytest.approx(0.0)

    def test_partial_overlap_start(self) -> None:
        """Window starts before the region and ends inside it."""
        regions = [{"start_time": 5.0, "end_time": 20.0, "label": "first_crack"}]
        assert compute_overlap(0.0, 10.0, regions) == pytest.approx(5.0)

    def test_partial_overlap_end(self) -> None:
        """Window starts inside the region and ends after it."""
        regions = [{"start_time": 0.0, "end_time": 7.0, "label": "first_crack"}]
        assert compute_overlap(5.0, 15.0, regions) == pytest.approx(2.0)

    def test_multiple_regions(self) -> None:
        """Window overlaps with two separate first_crack regions."""
        regions = [
            {"start_time": 1.0, "end_time": 3.0, "label": "first_crack"},
            {"start_time": 7.0, "end_time": 12.0, "label": "first_crack"},
        ]
        # 1-3 gives 2s overlap, 7-10 gives 3s overlap = 5s total
        assert compute_overlap(0.0, 10.0, regions) == pytest.approx(5.0)

    def test_ignores_wrong_label(self) -> None:
        """Regions with non-matching labels are ignored."""
        regions = [{"start_time": 0.0, "end_time": 10.0, "label": "no_first_crack"}]
        assert compute_overlap(0.0, 10.0, regions) == pytest.approx(0.0)

    def test_empty_regions(self) -> None:
        """No regions at all."""
        assert compute_overlap(0.0, 10.0, []) == pytest.approx(0.0)

    def test_overlapping_regions_union(self) -> None:
        """Overlapping regions compute union, not sum."""
        regions = [
            {"start_time": 0.0, "end_time": 10.0, "label": "first_crack"},
            {"start_time": 5.0, "end_time": 15.0, "label": "first_crack"},
        ]
        # Union is 0-15, clipped to window 0-10 = 10s
        assert compute_overlap(0.0, 10.0, regions) == pytest.approx(10.0)

    def test_overlapping_regions_partial(self) -> None:
        """Partially overlapping regions in the middle of a window."""
        regions = [
            {"start_time": 2.0, "end_time": 6.0, "label": "first_crack"},
            {"start_time": 4.0, "end_time": 8.0, "label": "first_crack"},
        ]
        # Union is 2-8 = 6s within a 0-10 window
        assert compute_overlap(0.0, 10.0, regions) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# label_window
# ---------------------------------------------------------------------------


class TestLabelWindow:
    """Tests for the window labelling logic."""

    def test_above_threshold(self) -> None:
        """Window with >50% overlap is labelled first_crack."""
        regions = [{"start_time": 0.0, "end_time": 8.0, "label": "first_crack"}]
        assert label_window(0.0, 10.0, regions, overlap_threshold=0.5) == "first_crack"

    def test_below_threshold(self) -> None:
        """Window with <50% overlap is labelled no_first_crack."""
        regions = [{"start_time": 0.0, "end_time": 4.0, "label": "first_crack"}]
        assert label_window(0.0, 10.0, regions, overlap_threshold=0.5) == "no_first_crack"

    def test_exactly_at_threshold(self) -> None:
        """Window with exactly 50% overlap meets the threshold."""
        regions = [{"start_time": 0.0, "end_time": 5.0, "label": "first_crack"}]
        assert label_window(0.0, 10.0, regions, overlap_threshold=0.5) == "first_crack"

    def test_zero_threshold(self) -> None:
        """With threshold=0, any overlap is sufficient."""
        regions = [{"start_time": 9.0, "end_time": 10.5, "label": "first_crack"}]
        assert label_window(0.0, 10.0, regions, overlap_threshold=0.0) == "first_crack"

    def test_no_regions(self) -> None:
        """No annotated regions -> no_first_crack."""
        assert label_window(0.0, 10.0, [], overlap_threshold=0.5) == "no_first_crack"


# ---------------------------------------------------------------------------
# chunk_recording
# ---------------------------------------------------------------------------


class TestChunkRecording:
    """Tests for the full chunking pipeline on a synthetic recording."""

    def _make_audio(self, duration_sec: float, sr: int = 16000) -> np.ndarray:
        """Create a silent audio array of the given duration."""
        return np.zeros(int(duration_sec * sr), dtype=np.float32)

    def test_basic_chunking(self) -> None:
        """30s recording with 10s windows produces 3 chunks."""
        audio = self._make_audio(30.0)
        regions = [{"start_time": 20.0, "end_time": 30.0, "label": "first_crack"}]
        chunks = chunk_recording(audio, 16000, regions, window_size=10.0)
        assert len(chunks) == 3
        assert chunks[0]["label"] == "no_first_crack"
        assert chunks[1]["label"] == "no_first_crack"
        assert chunks[2]["label"] == "first_crack"

    def test_discards_partial_window(self) -> None:
        """25s recording with 10s windows produces 2 chunks (last 5s dropped)."""
        audio = self._make_audio(25.0)
        chunks = chunk_recording(audio, 16000, [], window_size=10.0)
        assert len(chunks) == 2

    def test_hop_size_overlap(self) -> None:
        """With hop=5s on 20s recording, get 3 chunks (0-10, 5-15, 10-20)."""
        audio = self._make_audio(20.0)
        chunks = chunk_recording(audio, 16000, [], window_size=10.0, hop_size=5.0)
        assert len(chunks) == 3
        assert chunks[0]["start_sec"] == pytest.approx(0.0)
        assert chunks[1]["start_sec"] == pytest.approx(5.0)
        assert chunks[2]["start_sec"] == pytest.approx(10.0)

    def test_chunk_sample_length(self) -> None:
        """Each chunk has exactly window_size * sr samples."""
        sr = 16000
        audio = self._make_audio(30.0, sr=sr)
        chunks = chunk_recording(audio, sr, [], window_size=10.0)
        for chunk in chunks:
            assert len(chunk["samples"]) == 10 * sr

    def test_single_region_annotation(self) -> None:
        """One big first_crack region labels overlapping windows correctly."""
        audio = self._make_audio(60.0)
        # First crack from 30s to 50s
        regions = [{"start_time": 30.0, "end_time": 50.0, "label": "first_crack"}]
        chunks = chunk_recording(audio, 16000, regions, window_size=10.0)
        labels = [c["label"] for c in chunks]
        # Windows: 0-10(nfc), 10-20(nfc), 20-30(nfc), 30-40(fc), 40-50(fc), 50-60(nfc)
        assert labels == [
            "no_first_crack",
            "no_first_crack",
            "no_first_crack",
            "first_crack",
            "first_crack",
            "no_first_crack",
        ]

    def test_zero_window_size_raises(self) -> None:
        """Window size must be positive."""
        audio = self._make_audio(10.0)
        with pytest.raises(ValueError, match="window_size must be > 0 seconds"):
            chunk_recording(audio, 16000, [], window_size=0.0)

    def test_zero_hop_size_raises(self) -> None:
        """Hop size must be positive."""
        audio = self._make_audio(10.0)
        with pytest.raises(ValueError, match="hop_size must be > 0 seconds"):
            chunk_recording(audio, 16000, [], window_size=1.0, hop_size=0.0)

    def test_zero_sample_rate_raises(self) -> None:
        """Sample rate must be positive."""
        audio = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="sr must be > 0"):
            chunk_recording(audio, 0, [], window_size=1.0)

    def test_window_size_rounds_to_zero_samples_raises(self) -> None:
        """Reject window sizes that round to zero samples."""
        audio = np.zeros(1, dtype=np.float32)
        with pytest.raises(ValueError, match="window_size is too small for the sample rate"):
            chunk_recording(audio, 100, [], window_size=0.001, hop_size=0.01)

    def test_hop_size_rounds_to_zero_samples_raises(self) -> None:
        """Reject hop sizes that round to zero samples."""
        audio = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="hop_size is too small for the sample rate"):
            chunk_recording(audio, 100, [], window_size=0.01, hop_size=0.001)


# ---------------------------------------------------------------------------
# extract_recording_stem
# ---------------------------------------------------------------------------


class TestExtractRecordingStem:
    """Tests for parsing recording stem from chunk filenames."""

    def test_standard_chunk_name(self) -> None:
        assert (
            extract_recording_stem("roast-1-costarica-hermosa-hp-a_w0530.0.wav")
            == "roast-1-costarica-hermosa-hp-a"
        )

    def test_mic2_chunk_name(self) -> None:
        assert (
            extract_recording_stem("mic2-brazil-roast1-21-02-26-10-37_w0060.0.wav")
            == "mic2-brazil-roast1-21-02-26-10-37"
        )

    def test_zero_start(self) -> None:
        assert extract_recording_stem("mic2-brazil-roast1_w0000.0.wav") == "mic2-brazil-roast1"

    def test_no_window_suffix(self) -> None:
        """Filenames without the _w suffix return the full stem."""
        assert extract_recording_stem("some-file.wav") == "some-file"


# ---------------------------------------------------------------------------
# strip_hash_prefix
# ---------------------------------------------------------------------------


class TestStripHashPrefix:
    """Tests for Label Studio hash prefix stripping."""

    def test_hex_hash_prefix(self) -> None:
        """8-char hex hash is stripped."""
        assert strip_hash_prefix("0d93a737-roast-1.wav") == "roast-1.wav"

    def test_normal_filename_preserved(self) -> None:
        """Normal hyphenated filenames are NOT stripped."""
        assert strip_hash_prefix("roast-1-costarica-hermosa-hp-a.wav") == (
            "roast-1-costarica-hermosa-hp-a.wav"
        )

    def test_mic2_filename_preserved(self) -> None:
        """mic2 filenames are NOT stripped."""
        assert strip_hash_prefix("mic2-brazil-roast1.wav") == "mic2-brazil-roast1.wav"

    def test_no_hyphens(self) -> None:
        """Filename without hyphens returned as-is."""
        assert strip_hash_prefix("recording.wav") == "recording.wav"


# ---------------------------------------------------------------------------
# convert_task
# ---------------------------------------------------------------------------


class TestConvertTask:
    """Tests for converting Label Studio tasks to per-file annotations."""

    def test_successful_conversion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid task resolves the audio file, duration, and labels."""
        audio_file = tmp_path / "roast-1.wav"
        audio_file.write_bytes(b"stub")
        monkeypatch.setattr(convert_labelstudio_export.librosa, "get_duration", lambda path: 12.5)

        task = {
            "file_upload": "0d93a737-roast-1.wav",
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {
                                "start": 1.0,
                                "end": 2.5,
                                "labels": ["first_crack"],
                            },
                        }
                    ]
                }
            ],
        }

        converted = convert_task(task, tmp_path)

        assert converted["audio_file"] == "roast-1.wav"
        assert converted["duration"] == pytest.approx(12.5)
        assert converted["sample_rate"] == 44100
        assert converted["annotations"] == [
            {
                "id": "chunk_000",
                "start_time": 1.0,
                "end_time": 2.5,
                "label": "first_crack",
                "confidence": "high",
            }
        ]

    def test_missing_audio_source_raises(self, tmp_path: Path) -> None:
        """Tasks without file_upload or data.audio fail fast."""
        with pytest.raises(
            ValueError,
            match="Task is missing both 'file_upload' and 'data.audio'",
        ):
            convert_task({"annotations": []}, tmp_path)

    def test_missing_local_audio_file_raises(self, tmp_path: Path) -> None:
        """Resolved audio files must exist on disk."""
        task = {"file_upload": "0d93a737-roast-1.wav", "annotations": []}

        with pytest.raises(
            FileNotFoundError, match="Resolved audio file does not exist or is not a file"
        ):
            convert_task(task, tmp_path)

    def test_duration_read_failure_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Duration read errors are surfaced instead of silently zeroing metadata."""
        audio_file = tmp_path / "roast-1.wav"
        audio_file.write_bytes(b"stub")

        def _raise_duration_error(path: str) -> float:
            raise ValueError(f"bad audio: {path}")

        monkeypatch.setattr(
            convert_labelstudio_export.librosa, "get_duration", _raise_duration_error
        )

        task = {"file_upload": "0d93a737-roast-1.wav", "annotations": []}

        with pytest.raises(RuntimeError, match="Failed to read duration for audio file"):
            convert_task(task, tmp_path)


# ---------------------------------------------------------------------------
# recording_level_split
# ---------------------------------------------------------------------------


class TestRecordingLevelSplit:
    """Tests for recording-level dataset splitting edge cases."""

    def test_single_recording_raises_clear_error(self) -> None:
        """A single recording cannot satisfy the configured split ratios."""
        groups = {"rec1": {"first_crack": [Path("rec1_w0000.0.wav")]}}

        with pytest.raises(
            ValueError,
            match="Unable to split recordings with the requested test_size=0.15",
        ):
            recording_level_split(groups, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
