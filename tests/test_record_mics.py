"""Tests for scripts/record_mics.py audio-statistics helpers."""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.record_mics as rm

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SR = 44100  # sample rate used throughout tests


def _sine(
    amplitude: float = 0.1,
    duration: float = 1.0,
    freq: float = 440.0,
) -> np.ndarray:
    """Return a 1-D mono float32 sine wave."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _stereo_chunks(
    ch0: np.ndarray,
    ch1: np.ndarray,
    chunk_size: int = 1024,
) -> list[np.ndarray]:
    """Interleave two mono arrays into a list of (chunk_size, 2) chunks."""
    n = min(len(ch0), len(ch1))
    stereo = np.stack([ch0[:n], ch1[:n]], axis=1)
    return [stereo[i : i + chunk_size] for i in range(0, n, chunk_size)]


# ---------------------------------------------------------------------------
# _dbfs
# ---------------------------------------------------------------------------


class TestDbfs:
    """Tests for the _dbfs helper."""

    def test_full_scale_sine_peak_near_zero(self) -> None:
        """Full-scale sine should have peak ≈ 0 dBFS."""
        sig = _sine(amplitude=1.0)
        peak, _ = rm._dbfs(sig)
        assert peak == pytest.approx(0.0, abs=0.5)

    def test_half_amplitude_peak_near_minus6(self) -> None:
        """0.5-amplitude sine should have peak ≈ -6 dBFS."""
        sig = _sine(amplitude=0.5)
        peak, _ = rm._dbfs(sig)
        assert peak == pytest.approx(-6.0, abs=0.5)

    def test_rms_below_peak(self) -> None:
        """RMS of a sine wave is 3 dB below peak."""
        sig = _sine(amplitude=0.5)
        peak, rms = rm._dbfs(sig)
        assert rms < peak
        assert rms == pytest.approx(peak - 3.0, abs=0.2)

    def test_empty_array_returns_minus120(self) -> None:
        """Empty array should return -120 for both values."""
        peak, rms = rm._dbfs(np.array([], dtype=np.float32))
        assert peak == -120.0
        assert rms == -120.0

    def test_silent_array_near_floor(self) -> None:
        """All-zeros array should return values near the floor (-120 dBFS)."""
        sig = np.zeros(SR, dtype=np.float32)
        peak, rms = rm._dbfs(sig)
        assert peak < -100.0
        assert rms < -100.0


# ---------------------------------------------------------------------------
# _mic_stats_from_chunks
# ---------------------------------------------------------------------------


class TestMicStatsFromChunks:
    """Tests for _mic_stats_from_chunks."""

    def test_single_mic_peak_level(self) -> None:
        """Peak matches known input amplitude."""
        sig = _sine(amplitude=0.1)  # ≈ -20 dBFS peak
        chunks = _stereo_chunks(sig, np.zeros_like(sig))
        stats = rm._mic_stats_from_chunks(chunks, mics=[1], gains=[1.0])
        assert len(stats) == 1
        assert stats[0]["peak"] == pytest.approx(-20.0, abs=1.0)

    def test_second_channel(self) -> None:
        """mic=2 reads from channel index 1."""
        silent = np.zeros(SR, dtype=np.float32)
        loud = _sine(amplitude=0.5)
        chunks = _stereo_chunks(silent, loud)
        stats = rm._mic_stats_from_chunks(chunks, mics=[2], gains=[1.0])
        assert stats[0]["peak"] == pytest.approx(-6.0, abs=0.5)

    def test_gain_doubles_amplitude(self) -> None:
        """2× gain should add ≈ +6 dB to peak."""
        sig = _sine(amplitude=0.1)
        chunks = _stereo_chunks(sig, np.zeros_like(sig))
        no_gain = rm._mic_stats_from_chunks(chunks, mics=[1], gains=[1.0])
        with_gain = rm._mic_stats_from_chunks(chunks, mics=[1], gains=[2.0])
        assert with_gain[0]["peak"] == pytest.approx(no_gain[0]["peak"] + 6.0, abs=0.5)

    def test_gain_clips_at_unity(self) -> None:
        """Signal amplified above 1.0 is clipped; peak stays at 0 dBFS."""
        sig = _sine(amplitude=0.9)
        chunks = _stereo_chunks(sig, np.zeros_like(sig))
        stats = rm._mic_stats_from_chunks(chunks, mics=[1], gains=[5.0])
        assert stats[0]["peak"] == pytest.approx(0.0, abs=0.2)

    def test_start_chunk_windows_to_quiet_section(self) -> None:
        """start_chunk offsets into the chunk list; windowed RMS should be lower."""
        loud = _sine(amplitude=0.5)
        quiet = _sine(amplitude=0.01)
        loud_chunks = _stereo_chunks(loud, np.zeros_like(loud))
        quiet_chunks = _stereo_chunks(quiet, np.zeros_like(quiet))
        all_chunks = loud_chunks + quiet_chunks

        full_stats = rm._mic_stats_from_chunks(all_chunks, mics=[1], gains=[1.0])
        window_stats = rm._mic_stats_from_chunks(
            all_chunks, mics=[1], gains=[1.0], start_chunk=len(loud_chunks)
        )
        assert window_stats[0]["rms"] < full_stats[0]["rms"] - 10.0

    def test_empty_chunks_returns_floor(self) -> None:
        """Empty chunk list returns -120 for all mics."""
        stats = rm._mic_stats_from_chunks([], mics=[1, 2], gains=[1.0, 1.0])
        assert stats[0]["peak"] == -120.0
        assert stats[1]["rms"] == -120.0

    def test_start_chunk_beyond_end_returns_floor(self) -> None:
        """start_chunk beyond chunk list returns -120."""
        sig = _sine(amplitude=0.1)
        chunks = _stereo_chunks(sig, np.zeros_like(sig))
        stats = rm._mic_stats_from_chunks(
            chunks, mics=[1], gains=[1.0], start_chunk=len(chunks) + 10
        )
        assert stats[0]["peak"] == -120.0

    def test_two_mics_independent(self) -> None:
        """Stats for mic1 and mic2 reflect their respective channels."""
        ch1 = _sine(amplitude=0.5)
        ch2 = _sine(amplitude=0.1)
        chunks = _stereo_chunks(ch1, ch2)
        stats = rm._mic_stats_from_chunks(chunks, mics=[1, 2], gains=[1.0, 1.0])
        assert stats[0]["peak"] > stats[1]["peak"]


# ---------------------------------------------------------------------------
# _format_heartbeat
# ---------------------------------------------------------------------------


class TestFormatHeartbeat:
    """Tests for _format_heartbeat."""

    def test_timestamp_in_output(self) -> None:
        """Elapsed time formats as [MM:SS]."""
        stats = [{"peak": -20.0, "rms": -37.0}, {"peak": -22.0, "rms": -40.0}]
        line = rm._format_heartbeat(130.0, stats, mics=[1, 2], labels=["fifine", "atr"])
        assert "[02:10]" in line

    def test_per_mic_stats_in_output(self) -> None:
        """Both mic labels and their level values appear in the line."""
        stats = [{"peak": -20.0, "rms": -37.0}, {"peak": -22.0, "rms": -40.0}]
        line = rm._format_heartbeat(60.0, stats, mics=[1, 2], labels=["fifine", "atr"])
        assert "mic1" in line
        assert "mic2" in line
        assert "-20.0" in line
        assert "-37.0" in line

    def test_balance_warning_above_threshold(self) -> None:
        """RMS difference > 6 dB triggers the warning symbol."""
        stats = [{"peak": -20.0, "rms": -30.0}, {"peak": -35.0, "rms": -45.0}]
        line = rm._format_heartbeat(60.0, stats, mics=[1, 2], labels=["a", "b"])
        assert "⚠️" in line
        assert "balance=15.0dB" in line

    def test_balance_ok_below_threshold(self) -> None:
        """RMS difference ≤ 6 dB shows the OK symbol."""
        stats = [{"peak": -20.0, "rms": -37.0}, {"peak": -22.0, "rms": -40.0}]
        line = rm._format_heartbeat(60.0, stats, mics=[1, 2], labels=["a", "b"])
        assert "✅" in line

    def test_single_mic_no_balance_field(self) -> None:
        """With only one mic, no balance field is included."""
        stats = [{"peak": -20.0, "rms": -37.0}]
        line = rm._format_heartbeat(60.0, stats, mics=[1], labels=["fifine"])
        assert "balance" not in line

    def test_zero_elapsed(self) -> None:
        """Zero elapsed time renders as [00:00]."""
        stats = [{"peak": -20.0, "rms": -37.0}]
        line = rm._format_heartbeat(0.0, stats, mics=[1], labels=["fifine"])
        assert "[00:00]" in line


# ---------------------------------------------------------------------------
# _check_silent_mics
# ---------------------------------------------------------------------------


class TestCheckSilentMics:
    """Tests for _check_silent_mics."""

    def test_warns_for_silent_mic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A mic with RMS below threshold triggers a stderr warning."""
        stats = [{"peak": -80.0, "rms": -90.0}]
        warned = rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=set())
        captured = capsys.readouterr()
        assert "mic1" in captured.err
        assert "no signal" in captured.err
        assert 1 in warned

    def test_does_not_double_warn(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A mic already in the warned set is not warned again."""
        stats = [{"peak": -80.0, "rms": -90.0}]
        warned = rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=set())
        capsys.readouterr()  # consume first warning
        rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=warned)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_no_warn_for_active_mic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A mic with sufficient RMS produces no warning."""
        stats = [{"peak": -20.0, "rms": -37.0}]
        warned = rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=set())
        captured = capsys.readouterr()
        assert captured.err == ""
        assert warned == set()

    def test_returns_updated_set_with_silent_mic(self) -> None:
        """Only the silent mic number is added to the returned set."""
        stats = [{"peak": -80.0, "rms": -90.0}, {"peak": -20.0, "rms": -37.0}]
        warned = rm._check_silent_mics(stats, mics=[1, 2], labels=["fifine", "atr"], warned=set())
        assert 1 in warned
        assert 2 not in warned

    def test_does_not_mutate_input_set(self) -> None:
        """The input warned set is not mutated in place."""
        stats = [{"peak": -80.0, "rms": -90.0}]
        original: set[int] = set()
        rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=original)
        assert original == set()

    def test_boundary_exactly_at_threshold_is_not_silent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """RMS exactly equal to the threshold is NOT silent (condition is strict <)."""
        stats = [{"peak": -70.0, "rms": rm._SILENCE_THRESHOLD_DBFS}]
        warned = rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=set())
        assert 1 not in warned

    def test_boundary_just_above_threshold_is_not_silent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """RMS just above the threshold produces no warning."""
        stats = [{"peak": -25.0, "rms": rm._SILENCE_THRESHOLD_DBFS + 0.1}]
        warned = rm._check_silent_mics(stats, mics=[1], labels=["fifine"], warned=set())
        assert 1 not in warned


# ---------------------------------------------------------------------------
# _run_initial_silence_check
# ---------------------------------------------------------------------------


class TestRunInitialSilenceCheck:
    """Tests for _run_initial_silence_check."""

    def test_empty_chunks_keeps_retrying(self) -> None:
        """No audio yet should leave the check incomplete and warnings unchanged."""
        warned, completed = rm._run_initial_silence_check(
            chunks=[],
            mics=[1],
            labels=["fifine"],
            gains=[1.0],
            warned={2},
        )
        assert warned == {2}
        assert completed is False

    def test_available_audio_completes_check(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Captured audio should run the check and mark it complete."""
        silent = np.zeros(SR, dtype=np.float32)
        chunks = _stereo_chunks(silent, silent)
        warned, completed = rm._run_initial_silence_check(
            chunks=chunks,
            mics=[1],
            labels=["fifine"],
            gains=[1.0],
            warned=set(),
        )
        captured = capsys.readouterr()
        assert "mic1" in captured.err
        assert 1 in warned
        assert completed is True


# ---------------------------------------------------------------------------
# _print_session_summary
# ---------------------------------------------------------------------------


class TestPrintSessionSummary:
    """Tests for _print_session_summary."""

    def test_prints_per_mic_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Both mic numbers and their level values appear in the output."""
        stats = [{"peak": -20.0, "rms": -37.0}, {"peak": -22.0, "rms": -40.0}]
        rm._print_session_summary(stats, mics=[1, 2], labels=["fifine", "atr"])
        out = capsys.readouterr().out
        assert "mic1" in out
        assert "mic2" in out
        assert "-20.0" in out
        assert "-37.0" in out

    def test_prints_balance_for_two_mics(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Balance line is printed when there are two mics."""
        stats = [{"peak": -20.0, "rms": -37.0}, {"peak": -22.0, "rms": -40.0}]
        rm._print_session_summary(stats, mics=[1, 2], labels=["fifine", "atr"])
        out = capsys.readouterr().out
        assert "Balance" in out

    def test_no_balance_for_single_mic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No balance line when only one mic is present."""
        stats = [{"peak": -20.0, "rms": -37.0}]
        rm._print_session_summary(stats, mics=[1], labels=["fifine"])
        out = capsys.readouterr().out
        assert "Balance" not in out

    def test_clipping_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Peak ≥ −0.5 dBFS triggers CLIPPING warning."""
        stats = [{"peak": -0.1, "rms": -3.0}]
        rm._print_session_summary(stats, mics=[1], labels=["fifine"])
        out = capsys.readouterr().out
        assert "CLIPPING" in out

    def test_too_quiet_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Peak < −30 dBFS triggers TOO QUIET warning."""
        stats = [{"peak": -45.0, "rms": -60.0}]
        rm._print_session_summary(stats, mics=[1], labels=["fifine"])
        out = capsys.readouterr().out
        assert "TOO QUIET" in out

    def test_normal_levels_no_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Normal peak level produces no flag."""
        stats = [{"peak": -15.0, "rms": -32.0}]
        rm._print_session_summary(stats, mics=[1], labels=["fifine"])
        out = capsys.readouterr().out
        assert "CLIPPING" not in out
        assert "TOO QUIET" not in out

    def test_balance_unbalanced_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """RMS difference > 6 dB shows UNBALANCED."""
        stats = [{"peak": -20.0, "rms": -30.0}, {"peak": -35.0, "rms": -50.0}]
        rm._print_session_summary(stats, mics=[1, 2], labels=["a", "b"])
        out = capsys.readouterr().out
        assert "UNBALANCED" in out

    def test_balance_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        """RMS difference ≤ 6 dB shows balanced."""
        stats = [{"peak": -20.0, "rms": -37.0}, {"peak": -22.0, "rms": -40.0}]
        rm._print_session_summary(stats, mics=[1, 2], labels=["a", "b"])
        out = capsys.readouterr().out
        assert "balanced" in out
        assert "UNBALANCED" not in out


# ---------------------------------------------------------------------------
# StreamingWriter (#49 Part 1: streaming disk writes)
# ---------------------------------------------------------------------------


def _random_blocks(
    n_blocks: int,
    n_channels: int,
    block_size: int = 1024,
    seed: int = 42,
    amplitude: float = 0.3,
) -> list[np.ndarray]:
    """Return synthetic multi-channel float32 blocks for streaming-writer tests."""
    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(-amplitude, amplitude, size=(block_size, n_channels))).astype(np.float32)
        for _ in range(n_blocks)
    ]


def _write_via_buffered_reference(
    blocks: list[np.ndarray],
    mics: list[int],
    gains: list[float],
    sample_rate: int,
    paths: list[Path],
) -> None:
    """Reproduce the pre-#49 buffered write path: concatenate, gain, clip, sf.write.

    Mirrors exactly what ``cmd_record`` did before streaming writes (accumulate
    all chunks, then write once at the end) so the parity test has a ground
    truth independent of :class:`rm.StreamingWriter`.
    """
    full = np.concatenate(blocks, axis=0)
    for path, m, gain in zip(paths, mics, gains, strict=True):
        ch_idx = m - 1
        audio: np.ndarray = np.clip(full[:, ch_idx] * np.float32(gain), -1.0, 1.0).astype(
            np.float32, copy=False
        )
        sf.write(str(path), audio, sample_rate, subtype="FLOAT")


class TestStreamingWriter:
    """Tests for rm.StreamingWriter: correctness, parity, and lifecycle."""

    def test_frames_written_matches_input(self, tmp_path: Path) -> None:
        """Total frames written equals the sum of all enqueued block lengths."""
        blocks = _random_blocks(n_blocks=10, n_channels=2, block_size=512)
        paths = [tmp_path / "mic1.wav", tmp_path / "mic2.wav"]
        writer = rm.StreamingWriter(
            output_paths=paths, mics=[1, 2], gains=[1.0, 1.0], sample_rate=44100
        )
        writer.start()
        for block in blocks:
            writer.put(block)
        writer.stop_and_join()

        assert writer.frames_written == 10 * 512
        for path in paths:
            info = sf.info(str(path))
            assert info.frames == 10 * 512
            assert info.samplerate == 44100
            assert info.channels == 1
            assert info.subtype == "FLOAT"

    def test_byte_identical_to_buffered_reference(self, tmp_path: Path) -> None:
        """Streaming writer output is byte-for-byte identical to the old buffered path.

        This is the parity/determinism proof required by #49: for the same
        synthetic input, the new incremental writer must produce exactly the
        same PCM bytes (and WAV header) as the pre-#49 accumulate-then-write
        approach, for every mic and with per-mic gain applied.
        """
        blocks = _random_blocks(n_blocks=20, n_channels=2, block_size=1024)
        mics = [1, 2]
        gains = [1.0, 1.5]
        sample_rate = 44100

        reference_paths = [tmp_path / "ref_mic1.wav", tmp_path / "ref_mic2.wav"]
        _write_via_buffered_reference(blocks, mics, gains, sample_rate, reference_paths)

        streaming_paths = [tmp_path / "stream_mic1.wav", tmp_path / "stream_mic2.wav"]
        writer = rm.StreamingWriter(
            output_paths=streaming_paths, mics=mics, gains=gains, sample_rate=sample_rate
        )
        writer.start()
        for block in blocks:
            writer.put(block)
        writer.stop_and_join()

        for ref_path, stream_path in zip(reference_paths, streaming_paths, strict=True):
            assert ref_path.read_bytes() == stream_path.read_bytes(), (
                f"{stream_path.name} is not byte-identical to the buffered reference"
            )

    def test_streaming_twice_is_deterministic(self, tmp_path: Path) -> None:
        """Running the streaming writer twice on identical input yields identical bytes."""
        blocks = _random_blocks(n_blocks=15, n_channels=1, block_size=800, seed=7)
        sample_rate = 22050

        paths_a = [tmp_path / "a.wav"]
        writer_a = rm.StreamingWriter(
            output_paths=paths_a, mics=[1], gains=[1.0], sample_rate=sample_rate
        )
        writer_a.start()
        for block in blocks:
            writer_a.put(block)
        writer_a.stop_and_join()

        paths_b = [tmp_path / "b.wav"]
        writer_b = rm.StreamingWriter(
            output_paths=paths_b, mics=[1], gains=[1.0], sample_rate=sample_rate
        )
        writer_b.start()
        for block in blocks:
            writer_b.put(block)
        writer_b.stop_and_join()

        assert paths_a[0].read_bytes() == paths_b[0].read_bytes()

    def test_gain_and_clipping_applied_in_writer_thread(self, tmp_path: Path) -> None:
        """Gain is applied and the result clipped to [-1, 1] before writing."""
        block = np.full((100, 1), 0.9, dtype=np.float32)
        path = tmp_path / "clipped.wav"
        writer = rm.StreamingWriter(output_paths=[path], mics=[1], gains=[5.0], sample_rate=44100)
        writer.start()
        writer.put(block)
        writer.stop_and_join()

        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        assert np.allclose(audio, 1.0)

    def test_partial_write_before_close_is_still_valid_wav(self, tmp_path: Path) -> None:
        """A file with only some blocks written (simulating a mid-recording crash)
        is a valid, playable WAV once its handle is closed.

        This is the crash-safety property #49 asks for: closing after only
        part of the intended recording must still finalize the RIFF header
        correctly for the frames actually flushed.
        """
        blocks = _random_blocks(n_blocks=5, n_channels=1, block_size=256)
        path = tmp_path / "partial.wav"
        writer = rm.StreamingWriter(output_paths=[path], mics=[1], gains=[1.0], sample_rate=44100)
        writer.start()
        for block in blocks[:3]:  # simulate a crash after only 3 of 5 blocks
            writer.put(block)
        writer.stop_and_join()  # close() must still patch a consistent header

        info = sf.info(str(path))
        assert info.frames == 3 * 256
        audio, _ = sf.read(str(path))
        assert len(audio) == 3 * 256

    def test_stats_window_drains_and_resets(self, tmp_path: Path) -> None:
        """stats_window() returns blocks written since the last call, then resets."""
        blocks = _random_blocks(n_blocks=4, n_channels=1, block_size=100)
        path = tmp_path / "mic1.wav"
        writer = rm.StreamingWriter(output_paths=[path], mics=[1], gains=[1.0], sample_rate=44100)
        writer.start()
        for block in blocks[:2]:
            writer.put(block)
        writer.stop_and_join()  # join guarantees both blocks are consumed before draining

        first_window = writer.stats_window()
        assert len(first_window) == 2

        second_window_before_more = writer.stats_window()
        assert second_window_before_more == []  # drained already

    def test_stats_window_accumulates_across_multiple_puts(self, tmp_path: Path) -> None:
        """stats_window() reflects every block written since the previous drain,
        even across multiple put() calls, once the writer has finished."""
        blocks = _random_blocks(n_blocks=4, n_channels=1, block_size=100)
        path = tmp_path / "mic1.wav"
        writer = rm.StreamingWriter(output_paths=[path], mics=[1], gains=[1.0], sample_rate=44100)
        writer.start()
        for block in blocks:
            writer.put(block)
        writer.stop_and_join()

        window = writer.stats_window()
        assert len(window) == 4

    def test_multi_mic_independent_gain(self, tmp_path: Path) -> None:
        """Each mic's own gain is applied to its own channel, independently."""
        block = np.stack(
            [np.full(50, 0.1, dtype=np.float32), np.full(50, 0.1, dtype=np.float32)], axis=1
        )
        paths = [tmp_path / "mic1.wav", tmp_path / "mic2.wav"]
        writer = rm.StreamingWriter(
            output_paths=paths, mics=[1, 2], gains=[1.0, 2.0], sample_rate=44100
        )
        writer.start()
        writer.put(block)
        writer.stop_and_join()

        audio1, _ = sf.read(str(paths[0]), dtype="float32", always_2d=False)
        audio2, _ = sf.read(str(paths[1]), dtype="float32", always_2d=False)
        assert np.allclose(audio1, 0.1)
        assert np.allclose(audio2, 0.2)


# ---------------------------------------------------------------------------
# Post-recording verification (#49 Part 2)
# ---------------------------------------------------------------------------


class TestVerifyMicLevels:
    """Tests for rm.verify_mic_levels."""

    def test_normal_levels_pass(self) -> None:
        result = rm.verify_mic_levels("mic1", peak_dbfs=-15.0, rms_dbfs=-30.0)
        assert result.passed is True
        assert "OK" in result.message

    def test_clipping_fails(self) -> None:
        result = rm.verify_mic_levels("mic1", peak_dbfs=-0.1, rms_dbfs=-3.0)
        assert result.passed is False
        assert "CLIPPING" in result.message

    def test_clipping_boundary_is_failure(self) -> None:
        """Peak exactly at the -0.5 dBFS threshold is a failure (>=)."""
        result = rm.verify_mic_levels("mic1", peak_dbfs=-0.5, rms_dbfs=-3.0)
        assert result.passed is False

    def test_too_quiet_fails(self) -> None:
        result = rm.verify_mic_levels("mic1", peak_dbfs=-45.0, rms_dbfs=-60.0)
        assert result.passed is False
        assert "QUIET" in result.message

    def test_too_quiet_boundary_passes(self) -> None:
        """Peak exactly at -30 dBFS is NOT too quiet (condition is strict <)."""
        result = rm.verify_mic_levels("mic1", peak_dbfs=-30.0, rms_dbfs=-40.0)
        assert result.passed is True


class TestVerifyBalance:
    """Tests for rm.verify_balance."""

    def test_balanced_passes(self) -> None:
        result = rm.verify_balance([-30.0, -32.0])
        assert result.passed is True

    def test_unbalanced_fails(self) -> None:
        result = rm.verify_balance([-20.0, -40.0])
        assert result.passed is False
        assert "UNBALANCED" in result.message

    def test_single_mic_passes_trivially(self) -> None:
        result = rm.verify_balance([-30.0])
        assert result.passed is True
        assert "single mic" in result.message


class TestVerifySampleLock:
    """Tests for rm.verify_sample_lock."""

    def test_equal_counts_pass(self) -> None:
        result = rm.verify_sample_lock([1, 2], [44100, 44100])
        assert result.passed is True
        assert "sample-locked" in result.message

    def test_unequal_counts_fail(self) -> None:
        result = rm.verify_sample_lock([1, 2], [44100, 44050])
        assert result.passed is False
        assert "NOT sample-locked" in result.message


class TestVerifyDuration:
    """Tests for rm.verify_duration."""

    def test_matching_duration_passes(self) -> None:
        result = rm.verify_duration(session_duration_sec=908.0, actual_duration_sec=907.8)
        assert result.passed is True

    def test_mismatched_duration_fails(self) -> None:
        result = rm.verify_duration(session_duration_sec=908.0, actual_duration_sec=900.0)
        assert result.passed is False
        assert "mismatch" in result.message

    def test_boundary_exactly_at_tolerance_passes(self) -> None:
        """A 1.0s delta is exactly at the tolerance and should still pass (not >)."""
        result = rm.verify_duration(session_duration_sec=100.0, actual_duration_sec=99.0)
        assert result.passed is True


class TestVerifySessionFilesPresent:
    """Tests for rm.verify_session_files_present."""

    def test_all_files_present_passes(self, tmp_path: Path) -> None:
        (tmp_path / "mic1-x.wav").write_bytes(b"RIFF")
        session_data = {"mics": [{"file": "mic1-x.wav"}]}
        result = rm.verify_session_files_present(session_data, tmp_path)
        assert result.passed is True

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        session_data = {"mics": [{"file": "mic1-missing.wav"}]}
        result = rm.verify_session_files_present(session_data, tmp_path)
        assert result.passed is False
        assert "mic1-missing.wav" in result.message


class TestVerifyRecordingSession:
    """Tests for rm.verify_recording_session (end-to-end, real WAV files)."""

    def _write_session(
        self,
        tmp_path: Path,
        mics: list[int],
        amplitude: float,
        sample_rate: int = 44100,
        n_frames: int = 44100,
        duration_override: float | None = None,
    ) -> Path:
        """Write a fake recording session (WAVs + session JSON) and return the JSON path."""
        mic_meta = []
        for m in mics:
            filename = f"mic{m}-brazil-roast1.wav"
            audio = np.full(n_frames, amplitude, dtype=np.float32)
            sf.write(str(tmp_path / filename), audio, sample_rate, subtype="FLOAT")
            mic_meta.append({"mic_num": m, "label": f"mic{m}", "gain": 1.0, "file": filename})

        duration = duration_override if duration_override is not None else n_frames / sample_rate
        session_data = {
            "origin": "brazil",
            "roast_num": 1,
            "sample_rate": sample_rate,
            "duration_sec": round(duration, 2),
            "recorded_at": "2026-01-01T00:00:00Z",
            "mics": mic_meta,
        }
        session_path = tmp_path / "brazil-roast1-session.json"
        session_path.write_text(json.dumps(session_data))
        return session_path

    def test_valid_session_all_checks_pass(self, tmp_path: Path) -> None:
        session_path = self._write_session(tmp_path, mics=[1, 2], amplitude=0.1)
        results = rm.verify_recording_session(session_path, tmp_path)
        assert results, "expected at least one check result"
        assert all(r.passed for r in results), [r.message for r in results if not r.passed]

    def test_clipping_mic_fails(self, tmp_path: Path) -> None:
        session_path = self._write_session(tmp_path, mics=[1], amplitude=1.0)
        results = rm.verify_recording_session(session_path, tmp_path)
        levels = [r for r in results if "levels" in r.name][0]
        assert levels.passed is False

    def test_corrupt_wav_fails_with_clear_reason(self, tmp_path: Path) -> None:
        """A truncated/corrupt WAV file fails the level check with a clear reason."""
        session_path = self._write_session(tmp_path, mics=[1], amplitude=0.1)
        session_data = json.loads(session_path.read_text())
        wav_path = tmp_path / session_data["mics"][0]["file"]
        # Truncate to a handful of bytes — not a valid WAV/RIFF file anymore.
        wav_path.write_bytes(b"not a wav")

        results = rm.verify_recording_session(session_path, tmp_path)
        levels = [r for r in results if "levels" in r.name][0]
        assert levels.passed is False
        assert "unreadable" in levels.message.lower() or "corrupt" in levels.message.lower()

    def test_missing_wav_file_fails(self, tmp_path: Path) -> None:
        session_path = self._write_session(tmp_path, mics=[1], amplitude=0.1)
        session_data = json.loads(session_path.read_text())
        wav_path = tmp_path / session_data["mics"][0]["file"]
        wav_path.unlink()

        results = rm.verify_recording_session(session_path, tmp_path)
        levels = [r for r in results if "levels" in r.name][0]
        assert levels.passed is False
        assert "not found" in levels.message.lower()
        files_present = [r for r in results if r.name == "Session JSON"][0]
        assert files_present.passed is False

    def test_duration_mismatch_fails(self, tmp_path: Path) -> None:
        session_path = self._write_session(
            tmp_path, mics=[1], amplitude=0.1, duration_override=500.0
        )
        results = rm.verify_recording_session(session_path, tmp_path)
        duration_result = [r for r in results if r.name == "Duration"][0]
        assert duration_result.passed is False

    def test_sample_lock_mismatch_fails(self, tmp_path: Path) -> None:
        # Write mic1 and mic2 with different frame counts directly (bypassing
        # the equal-length helper) to simulate a dropped block for one mic.
        sf.write(
            str(tmp_path / "mic1-brazil-roast1.wav"),
            np.zeros(44100, dtype=np.float32),
            44100,
            subtype="FLOAT",
        )
        sf.write(
            str(tmp_path / "mic2-brazil-roast1.wav"),
            np.zeros(44000, dtype=np.float32),
            44100,
            subtype="FLOAT",
        )
        session_data = {
            "origin": "brazil",
            "roast_num": 1,
            "sample_rate": 44100,
            "duration_sec": 1.0,
            "recorded_at": "2026-01-01T00:00:00Z",
            "mics": [
                {"mic_num": 1, "label": "mic1", "gain": 1.0, "file": "mic1-brazil-roast1.wav"},
                {"mic_num": 2, "label": "mic2", "gain": 1.0, "file": "mic2-brazil-roast1.wav"},
            ],
        }
        session_path = tmp_path / "brazil-roast1-session.json"
        session_path.write_text(json.dumps(session_data))

        results = rm.verify_recording_session(session_path, tmp_path)
        sample_result = [r for r in results if r.name == "Samples"][0]
        assert sample_result.passed is False


class TestPrintVerificationReport:
    """Tests for rm.print_verification_report."""

    def test_all_passed_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [rm.VerificationResult("check1", True, "ok")]
        assert rm.print_verification_report(results) is True
        out = capsys.readouterr().out
        assert "All checks passed" in out

    def test_any_failure_reports_overall_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            rm.VerificationResult("check1", True, "ok"),
            rm.VerificationResult("check2", False, "bad"),
        ]
        assert rm.print_verification_report(results) is False
        out = capsys.readouterr().out
        assert "FAILED" in out


# ---------------------------------------------------------------------------
# SIGTERM handling (#49 acceptance criterion)
# ---------------------------------------------------------------------------


class TestSigtermShutdown:
    """Tests that SIGTERM triggers the same clean shutdown as Ctrl-C.

    These exercise the underlying mechanism (a threading.Event set from a
    signal handler, observed by the recording loop) without spinning up a
    real PortAudio input stream, keeping the test hardware-free.
    """

    def test_sigterm_sets_stop_event_via_registered_handler(self) -> None:
        """A signal.signal-registered handler sets the stop event, mirroring
        the handler cmd_record installs for SIGTERM."""
        stop_requested = threading.Event()

        def _handle_sigterm(_signum: int, _frame: object) -> None:
            stop_requested.set()

        previous = signal.signal(signal.SIGTERM, _handle_sigterm)
        try:
            assert not stop_requested.is_set()
            import os

            os.kill(os.getpid(), signal.SIGTERM)
            # Give the signal a moment to be delivered and handled.
            for _ in range(50):
                if stop_requested.is_set():
                    break
                time.sleep(0.01)
            assert stop_requested.is_set()
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_writer_flushes_and_closes_on_stop_and_join(self, tmp_path: Path) -> None:
        """stop_and_join (what a SIGTERM-triggered shutdown calls) leaves a
        valid, fully closed WAV file — the crash/kill-safety property."""
        path = tmp_path / "mic1.wav"
        writer = rm.StreamingWriter(output_paths=[path], mics=[1], gains=[1.0], sample_rate=44100)
        writer.start()
        writer.put(np.zeros((512, 1), dtype=np.float32))
        writer.stop_and_join()

        # File is fully closed and readable — proves the header was finalized.
        info = sf.info(str(path))
        assert info.frames == 512
