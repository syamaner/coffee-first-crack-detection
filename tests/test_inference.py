"""Tests for inference.py — SlidingWindowInference and FirstCrackDetector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from coffee_first_crack.inference import (
    DetectionEvent,
    FirstCrackDetector,
    SlidingWindowInference,
    _format_time,
)


# ── _format_time ──────────────────────────────────────────────────────────────


class TestFormatTime:
    def test_zero(self) -> None:
        assert _format_time(0) == "00:00"

    def test_one_minute(self) -> None:
        assert _format_time(60) == "01:00"

    def test_mixed(self) -> None:
        assert _format_time(95) == "01:35"

    def test_long(self) -> None:
        assert _format_time(3600) == "60:00"


# ── SlidingWindowInference ────────────────────────────────────────────────────


@pytest.fixture()
def silent_wav(tmp_path: Path) -> Path:
    """Write a 60-second silent WAV file (no first crack)."""
    path = tmp_path / "silent_roast.wav"
    samples = np.zeros(16000 * 60, dtype=np.float32)
    sf.write(str(path), samples, 16000)
    return path


class TestSlidingWindowInference:
    def test_no_detection_on_silence(self, silent_wav: Path) -> None:
        """Silent audio should produce no first crack events (noise gate triggers)."""
        detector = SlidingWindowInference(
            model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593",
            device="cpu",
        )
        events = detector.process_file(silent_wav)
        # Silent audio is blocked by the noise gate (RMS < 0.01)
        assert isinstance(events, list)
        assert len(events) == 0

    def test_returns_detection_event_type(self, tmp_path: Path) -> None:
        """Verify DetectionEvent fields are correctly typed."""
        event = DetectionEvent(
            timestamp_sec=42.0,
            timestamp_str="00:42",
            confidence=7,
        )
        assert event.timestamp_sec == 42.0
        assert event.timestamp_str == "00:42"
        assert event.confidence == 7


# ── FirstCrackDetector ────────────────────────────────────────────────────────


class TestFirstCrackDetector:
    def test_raises_when_both_inputs_given(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not both"):
            FirstCrackDetector(
                audio_file=tmp_path / "foo.wav",
                use_microphone=True,
            )

    def test_raises_when_no_input_given(self) -> None:
        with pytest.raises(ValueError, match="Must specify"):
            FirstCrackDetector()

    def test_is_first_crack_returns_false_before_start(self, silent_wav: Path) -> None:
        detector = FirstCrackDetector(
            audio_file=silent_wav,
            model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593",
        )
        assert detector.is_first_crack() is False

    def test_not_running_before_start(self, silent_wav: Path) -> None:
        detector = FirstCrackDetector(audio_file=silent_wav)
        assert not detector.is_running

    def test_start_stop_lifecycle(self, silent_wav: Path) -> None:
        """Detector starts, runs briefly, stops cleanly without errors."""
        detector = FirstCrackDetector(
            audio_file=silent_wav,
            model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593",
        )
        detector.start()
        assert detector.is_running
        import time
        time.sleep(0.5)
        detector.stop()
        assert not detector.is_running
