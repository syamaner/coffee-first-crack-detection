"""Tests for the data_prep module — chunking logic and dataset splitting utilities."""

from __future__ import annotations

import numpy as np
import pytest

from coffee_first_crack.data_prep.chunk_audio import (
    chunk_recording,
    compute_overlap,
    label_window,
)
from coffee_first_crack.data_prep.convert_labelstudio_export import strip_hash_prefix
from coffee_first_crack.data_prep.dataset_splitter import extract_recording_stem

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
