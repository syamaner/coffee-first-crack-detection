"""Tests for scripts/record_mics.py audio-statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

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
