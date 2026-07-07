"""Multi-mic synchronized recording tool for coffee roasting sessions.

Captures 1-N microphones simultaneously through a macOS CoreAudio Aggregate
Device and writes one mono WAV file per mic plus a session metadata JSON.
Recording runs indefinitely until the user presses Ctrl-C.

macOS only — requires a CoreAudio Aggregate Device configured in Audio MIDI
Setup.  See ``docs/multi_mic_setup.md`` for step-by-step instructions.

Usage::

    # List available input devices
    python scripts/record_mics.py list-devices

    # Record with defaults (mic1 + mic2, device name from configs/default.yaml)
    python scripts/record_mics.py record --origin brazil --roast-num 7

    # Single-mic session (setup test)
    python scripts/record_mics.py record --origin brazil --roast-num 7 --mics 1

    # Three mics with custom gains and quiet mode
    python scripts/record_mics.py record --origin brazil --roast-num 7 \\
        --mics 1 2 3 --labels fifine audiotechnica lavalier \\
        --gains 1.0 1.2 0.9 --quiet
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml

_DEFAULT_DEVICE = "RoastMics"
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_MICS = [1, 2]
_DEFAULT_MIN_DURATION_SEC = 60
_CONFIG_PATH = Path("configs/default.yaml")

# Level-monitoring thresholds
_SILENCE_THRESHOLD_DBFS: float = -60.0  # below this → mic is considered silent
_BALANCE_WARN_DB: float = 6.0  # RMS difference above this → balance warning
_SILENCE_CHECK_AFTER_SEC: float = 5.0  # seconds before first silence check fires

# Verification thresholds (Part 2 of #49)
_CLIP_WARN_DBFS: float = -0.5  # peak >= this → clipping warning
_QUIET_WARN_DBFS: float = -30.0  # peak < this → too-quiet warning
_DURATION_TOLERANCE_SEC: float = 1.0  # session JSON vs actual audio duration slack

# Producer-consumer queue sentinel: signals the writer thread to flush + close.
_QUEUE_STOP = None


# ---------------------------------------------------------------------------
# Audio statistics helpers
# ---------------------------------------------------------------------------


def _dbfs(arr: np.ndarray) -> tuple[float, float]:
    """Return (peak_dBFS, rms_dBFS) for a 1-D float32 audio array.

    Args:
        arr: 1-D float32 array with values in ``[-1.0, 1.0]``.

    Returns:
        Tuple of ``(peak_dBFS, rms_dBFS)``.  Both values are ``-120.0`` when
        *arr* is empty (avoids ``log(0)``).
    """
    if arr.size == 0:
        return -120.0, -120.0
    peak = 20.0 * np.log10(float(np.max(np.abs(arr))) + 1e-12)
    rms = 20.0 * np.log10(float(np.sqrt(np.mean(arr**2))) + 1e-12)
    return peak, rms


def _mic_stats_from_chunks(
    chunks: list[np.ndarray],
    mics: list[int],
    gains: list[float],
    start_chunk: int = 0,
) -> list[dict[str, float]]:
    """Compute peak and RMS dBFS per mic from a slice of audio chunks.

    Gain is applied and signal clipped to ``[-1, 1]`` before computing stats,
    matching the final WAV write logic so heartbeat values match the recording.

    Args:
        chunks: Full list of multi-channel audio chunks, each with shape
            ``(frames, n_channels)``.
        mics: Ordered mic numbers (1-indexed; mic N uses channel index N-1).
        gains: Per-mic digital gain multipliers, same length as *mics*.
        start_chunk: Index into *chunks* to start from.  Pass the chunk count
            at the previous heartbeat to obtain window-only stats.

    Returns:
        List of dicts with keys ``"peak"`` and ``"rms"`` (dBFS) per mic, in
        the same order as *mics*.  Returns ``-120.0`` for both fields when the
        requested window is empty.
    """
    window = chunks[start_chunk:]
    if not window:
        return [{"peak": -120.0, "rms": -120.0} for _ in mics]
    audio = np.concatenate(window, axis=0)  # (frames, n_channels)
    result: list[dict[str, float]] = []
    for m, gain in zip(mics, gains, strict=True):
        ch: np.ndarray = np.clip(audio[:, m - 1] * np.float32(gain), -1.0, 1.0).astype(
            np.float32, copy=False
        )
        peak, rms = _dbfs(ch)
        result.append({"peak": peak, "rms": rms})
    return result


def _format_heartbeat(
    elapsed_sec: float,
    stats: list[dict[str, float]],
    mics: list[int],
    labels: list[str],
) -> str:
    """Format a heartbeat line with per-mic level stats and a balance indicator.

    Stats should cover the most recent 30-second window (since the last
    heartbeat) so the line reflects current signal conditions.

    Args:
        elapsed_sec: Total elapsed recording time in seconds.
        stats: Per-mic stats dicts from :func:`_mic_stats_from_chunks`.
        mics: Ordered mic numbers.
        labels: Per-mic hardware labels, same length as *mics*.

    Returns:
        Formatted single-line string, ready to ``print()``.
    """
    mins, secs = divmod(int(elapsed_sec), 60)
    parts: list[str] = []
    for mic_stats, m, lbl in zip(stats, mics, labels, strict=True):
        parts.append(f"mic{m}({lbl}): peak={mic_stats['peak']:.1f} rms={mic_stats['rms']:.1f} dBFS")
    if len(stats) >= 2:
        rms_vals = [s["rms"] for s in stats]
        balance = max(rms_vals) - min(rms_vals)
        bal_sym = "\u26a0\ufe0f" if balance > _BALANCE_WARN_DB else "\u2705"
        parts.append(f"balance={balance:.1f}dB {bal_sym}")
    return f"[{mins:02d}:{secs:02d}] " + " | ".join(parts)


def _check_silent_mics(
    stats: list[dict[str, float]],
    mics: list[int],
    labels: list[str],
    warned: set[int],
) -> set[int]:
    """Warn on stderr for any mic whose RMS is below the silence threshold.

    Each mic is warned at most once per call cycle.  Pass the returned set
    back on the next call to suppress duplicate warnings.

    Args:
        stats: Per-mic stats dicts from :func:`_mic_stats_from_chunks`.
        mics: Ordered mic numbers.
        labels: Per-mic hardware labels, same length as *mics*.
        warned: Set of mic numbers already warned in this session.

    Returns:
        Updated set that includes any newly warned mic numbers.
    """
    new_warned = set(warned)
    for mic_stats, m, lbl in zip(stats, mics, labels, strict=True):
        if mic_stats["rms"] < _SILENCE_THRESHOLD_DBFS and m not in warned:
            print(
                f"\u26a0\ufe0f  mic{m} ({lbl}): no signal detected "
                f"(rms={mic_stats['rms']:.1f} dBFS) \u2014 is it turned on?",
                file=sys.stderr,
            )
            new_warned.add(m)
    return new_warned


def _run_initial_silence_check(
    chunks: list[np.ndarray],
    mics: list[int],
    labels: list[str],
    gains: list[float],
    warned: set[int],
) -> tuple[set[int], bool]:
    """Run the initial silence check once audio has arrived.

    The recorder starts the check timer when the stream opens, but CoreAudio can
    take a moment to deliver the first callback. When no audio has been captured
    yet, keep retrying on the next loop iteration instead of permanently marking
    the check complete.

    Args:
        chunks: Captured multi-channel audio chunks so far.
        mics: Ordered mic numbers.
        labels: Per-mic hardware labels, same length as *mics*.
        gains: Per-mic digital gain multipliers, same length as *mics*.
        warned: Set of mic numbers already warned in this session.

    Returns:
        Tuple of ``(updated_warned, completed)``. ``completed`` is ``True`` only
        after at least one chunk was available and silence stats were computed.
    """
    if not chunks:
        return set(warned), False
    init_stats = _mic_stats_from_chunks(chunks, mics, gains)
    return _check_silent_mics(init_stats, mics, labels, warned), True


def _print_session_summary(
    stats: list[dict[str, float]],
    mics: list[int],
    labels: list[str],
) -> None:
    """Print a full-session per-mic level summary and balance check.

    Called after all WAV files have been written, using stats computed from
    the complete session audio.

    Args:
        stats: Per-mic stats dicts from :func:`_mic_stats_from_chunks`.
        mics: Ordered mic numbers.
        labels: Per-mic hardware labels, same length as *mics*.
    """
    print("\n--- Recording summary ---")
    for mic_stats, m, lbl in zip(stats, mics, labels, strict=True):
        flags = ""
        if mic_stats["peak"] >= -0.5:
            flags = "  \u26a0\ufe0f  CLIPPING"
        elif mic_stats["peak"] < -30.0:
            flags = "  \u26a0\ufe0f  TOO QUIET"
        print(
            f"  mic{m} ({lbl:<14s}): "
            f"peak={mic_stats['peak']:+6.1f} dBFS  "
            f"rms={mic_stats['rms']:+6.1f} dBFS{flags}"
        )
    if len(stats) >= 2:
        rms_vals = [s["rms"] for s in stats]
        balance = max(rms_vals) - min(rms_vals)
        unbalanced = "\u26a0\ufe0f  UNBALANCED (>6dB)"
        bal_sym = unbalanced if balance > _BALANCE_WARN_DB else "\u2705 balanced"
        print(f"  Balance: {balance:.1f} dB  {bal_sym}")


# ---------------------------------------------------------------------------
# Streaming writer (Part 1 of #49)
# ---------------------------------------------------------------------------


@dataclass
class StreamingWriter:
    """Producer-consumer writer that streams multi-mic audio to disk.

    A PortAudio callback (the producer) pushes raw multi-channel blocks onto
    a thread-safe :class:`queue.Queue` with a non-blocking ``put`` \u2014 no I/O,
    no numpy work, no locking happens on the audio thread.  A dedicated writer
    thread (the consumer) owns every per-mic :class:`soundfile.SoundFile`
    handle, applies gain + clipping, and writes incrementally so at most one
    queued block is ever unflushed to disk.  This bounds peak memory to the
    queue depth (a handful of blocks) instead of the full recording, and a
    ``SIGTERM``/crash after the writer thread starts still leaves a valid,
    playable partial WAV file up to the last flushed block because
    ``soundfile`` patches the RIFF header on every close of the stream.

    Args:
        output_paths: Per-mic temporary ``_recording`` WAV paths, in the same
            order as *mics*.
        mics: Ordered mic numbers (1-indexed; mic N reads channel N-1).
        gains: Per-mic digital gain multipliers, same length as *mics*.
        sample_rate: Capture sample rate in Hz.

    Attributes:
        frames_written: Total multi-channel frames written so far (updated
            only by the writer thread; read from the main thread only after
            :meth:`join` has returned, per the class docstring's ownership
            rule).
    """

    output_paths: list[Path]
    mics: list[int]
    gains: list[float]
    sample_rate: int
    frames_written: int = field(default=0, init=False)
    _queue: queue.Queue[np.ndarray | None] = field(default_factory=queue.Queue, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stats_window: list[np.ndarray] = field(default_factory=list, init=False)
    _stats_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        """Start the writer thread. Call once before feeding the queue."""
        self._thread = threading.Thread(target=self._run, name="streaming-writer", daemon=False)
        self._thread.start()

    def put(self, block: np.ndarray) -> None:
        """Enqueue a raw multi-channel block captured by the audio callback.

        Safe to call from the PortAudio callback thread: this only performs a
        non-blocking queue push, never I/O.

        Args:
            block: Multi-channel float32 array, shape ``(frames, n_channels)``.
        """
        self._queue.put(block)

    def stop_and_join(self) -> None:
        """Queue the stop sentinel and block until the writer thread exits.

        Safe to call from the main thread or a signal handler.  Flushes and
        closes every ``SoundFile`` handle before returning.
        """
        self._queue.put(_QUEUE_STOP)
        if self._thread is not None:
            self._thread.join()

    def stats_window(self) -> list[np.ndarray]:
        """Return the raw multi-channel blocks written since the last call.

        Used to compute heartbeat / silence-check stats without retaining the
        full recording in memory.  Draining resets the window.

        Returns:
            List of multi-channel blocks written since the previous call to
            this method (or since :meth:`start`, on the first call).
        """
        with self._stats_lock:
            window, self._stats_window = self._stats_window, []
        return window

    def _run(self) -> None:
        """Writer-thread body: consume the queue, write, close on sentinel."""
        files = [
            sf.SoundFile(str(p), mode="w", samplerate=self.sample_rate, channels=1, subtype="FLOAT")
            for p in self.output_paths
        ]
        try:
            while True:
                block = self._queue.get()
                if block is None:  # sentinel
                    break
                with self._stats_lock:
                    self._stats_window.append(block)
                self.frames_written += len(block)
                for handle, m, gain in zip(files, self.mics, self.gains, strict=True):
                    ch_idx = m - 1
                    audio: np.ndarray = np.clip(
                        block[:, ch_idx] * np.float32(gain), -1.0, 1.0
                    ).astype(np.float32, copy=False)
                    handle.write(audio)
        finally:
            for handle in files:
                handle.close()


# ---------------------------------------------------------------------------
# Post-recording verification (Part 2 of #49)
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of a single post-recording verification check.

    Attributes:
        name: Short machine-stable check name, e.g. ``"mic1 levels"``.
        passed: ``True`` if the check passed (including warn-only checks that
            did not trigger), ``False`` on failure.
        message: Human-readable detail, ready to print.
    """

    name: str
    passed: bool
    message: str


def verify_mic_levels(label: str, peak_dbfs: float, rms_dbfs: float) -> VerificationResult:
    """Flag clipping or too-quiet levels for one mic's full-session audio.

    Args:
        label: Hardware label for this mic, used in the message.
        peak_dbfs: Full-session peak level in dBFS.
        rms_dbfs: Full-session RMS level in dBFS.

    Returns:
        A :class:`VerificationResult`; ``passed`` is ``False`` when the peak
        indicates clipping (``>= -0.5 dBFS``) or is too quiet (``< -30 dBFS``).
    """
    if peak_dbfs >= _CLIP_WARN_DBFS:
        return VerificationResult(
            f"{label} levels",
            False,
            f"peak={peak_dbfs:.1f} dBFS  RMS={rms_dbfs:.1f} dBFS  \u274c CLIPPING",
        )
    if peak_dbfs < _QUIET_WARN_DBFS:
        return VerificationResult(
            f"{label} levels",
            False,
            f"peak={peak_dbfs:.1f} dBFS  RMS={rms_dbfs:.1f} dBFS  \u274c TOO QUIET",
        )
    return VerificationResult(
        f"{label} levels",
        True,
        f"peak={peak_dbfs:.1f} dBFS  RMS={rms_dbfs:.1f} dBFS  \u2705 levels OK",
    )


def verify_balance(rms_values: list[float]) -> VerificationResult:
    """Check that per-mic RMS levels agree within the balance threshold.

    Args:
        rms_values: Full-session RMS dBFS for each recorded mic, in mic order.

    Returns:
        A :class:`VerificationResult`; ``passed`` is ``False`` when the max
        minus min RMS exceeds :data:`_BALANCE_WARN_DB` (6 dB). Always passes
        (with an explanatory message) when fewer than two mics are present.
    """
    if len(rms_values) < 2:
        return VerificationResult("Balance", True, "n/a (single mic)")
    balance = max(rms_values) - min(rms_values)
    if balance > _BALANCE_WARN_DB:
        return VerificationResult(
            "Balance", False, f"{balance:.1f} dB  \u274c UNBALANCED (>{_BALANCE_WARN_DB:.0f}dB)"
        )
    return VerificationResult("Balance", True, f"{balance:.1f} dB  \u2705 balanced")


def verify_sample_lock(mics: list[int], frame_counts: list[int]) -> VerificationResult:
    """Check that every mic's WAV file has an identical sample (frame) count.

    A CoreAudio Aggregate Device with Drift Correction should keep every
    channel sample-locked for the full session; a mismatch means the writer
    dropped or duplicated a block for one mic.

    Args:
        mics: Ordered mic numbers.
        frame_counts: Frame count read back from each mic's WAV file, same
            order as *mics*.

    Returns:
        A :class:`VerificationResult`; ``passed`` is ``False`` when frame
        counts differ across mics.
    """
    detail = "  ".join(f"mic{m}={n:,}" for m, n in zip(mics, frame_counts, strict=True))
    if len(set(frame_counts)) > 1:
        return VerificationResult("Samples", False, f"{detail}  \u274c NOT sample-locked")
    return VerificationResult("Samples", True, f"{detail}  \u2705 sample-locked")


def verify_duration(session_duration_sec: float, actual_duration_sec: float) -> VerificationResult:
    """Check that the session JSON duration matches the actual audio duration.

    Args:
        session_duration_sec: ``duration_sec`` recorded in the session JSON.
        actual_duration_sec: Duration computed from the WAV frame count and
            sample rate.

    Returns:
        A :class:`VerificationResult`; ``passed`` is ``False`` when the two
        values differ by more than :data:`_DURATION_TOLERANCE_SEC` (1s).
    """
    delta = abs(session_duration_sec - actual_duration_sec)
    if delta > _DURATION_TOLERANCE_SEC:
        return VerificationResult(
            "Duration",
            False,
            f"session.json={session_duration_sec:.1f}s  audio={actual_duration_sec:.1f}s  "
            f"\u274c mismatch (\u0394{delta:.1f}s)",
        )
    return VerificationResult(
        "Duration",
        True,
        f"session.json={session_duration_sec:.1f}s  audio={actual_duration_sec:.1f}s  \u2705 match",
    )


def verify_session_files_present(
    session_data: dict[str, Any], output_dir: Path
) -> VerificationResult:
    """Check that every ``file`` entry in the session JSON exists on disk.

    Args:
        session_data: Parsed session JSON dict (must contain a ``"mics"``
            list of dicts each with a ``"file"`` key).
        output_dir: Directory the session's WAV files were written into.

    Returns:
        A :class:`VerificationResult`; ``passed`` is ``False`` when any
        listed file is missing, naming the first missing file.
    """
    missing = [
        mic["file"]
        for mic in session_data.get("mics", [])
        if not (output_dir / mic["file"]).exists()
    ]
    if missing:
        return VerificationResult(
            "Session JSON", False, f"\u274c missing file(s): {', '.join(missing)}"
        )
    return VerificationResult("Session JSON", True, "all files present  \u2705")


def verify_recording_session(session_path: Path, output_dir: Path) -> list[VerificationResult]:
    """Run every post-recording verification check for one recorded session.

    Reads the session JSON and each mic's WAV file directly from disk \u2014 this
    is a fully independent read-back, so it also catches a WAV header left
    inconsistent by a crashed or killed writer.

    Args:
        session_path: Path to the ``*-session.json`` file written by
            :func:`cmd_record`.
        output_dir: Directory containing the session's WAV files (normally
            *session_path*'s parent).

    Returns:
        Ordered list of :class:`VerificationResult` \u2014 per-mic level checks,
        then balance, sample-lock, duration, and session-JSON presence.
    """
    session_data: dict[str, Any] = json.loads(session_path.read_text())
    mics_meta: list[dict[str, Any]] = session_data.get("mics", [])
    sample_rate: int = session_data["sample_rate"]

    results: list[VerificationResult] = []
    rms_values: list[float] = []
    frame_counts: list[int] = []
    mics_ok: list[int] = []
    max_frames = 0

    for mic in mics_meta:
        wav_path = output_dir / mic["file"]
        if not wav_path.exists():
            results.append(
                VerificationResult(
                    f"mic{mic['mic_num']} levels",
                    False,
                    f"\u274c file not found: {wav_path.name}",
                )
            )
            continue
        try:
            with sf.SoundFile(str(wav_path)) as handle:
                frames = len(handle)
                audio = handle.read(dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001 \u2014 surface any libsndfile failure as a check failure
            results.append(
                VerificationResult(
                    f"mic{mic['mic_num']} levels", False, f"\u274c unreadable/corrupt WAV: {exc}"
                )
            )
            continue
        peak, rms = _dbfs(np.asarray(audio, dtype=np.float32))
        mic_label = f"mic{mic['mic_num']} ({mic.get('label', '')})"
        results.append(verify_mic_levels(mic_label, peak, rms))
        rms_values.append(rms)
        frame_counts.append(frames)
        mics_ok.append(mic["mic_num"])
        max_frames = max(max_frames, frames)

    results.append(verify_balance(rms_values))
    if frame_counts:
        results.append(verify_sample_lock(mics_ok, frame_counts))
        actual_duration = max_frames / sample_rate if sample_rate else 0.0
        results.append(verify_duration(session_data.get("duration_sec", 0.0), actual_duration))
    results.append(verify_session_files_present(session_data, output_dir))
    return results


def print_verification_report(results: list[VerificationResult]) -> bool:
    """Print the post-recording verification report and return overall pass/fail.

    Args:
        results: Ordered check results from :func:`verify_recording_session`.

    Returns:
        ``True`` if every check passed, ``False`` otherwise.
    """
    print("\n--- Post-recording verification ---")
    for result in results:
        print(f"{result.name}: {result.message}")
    all_passed = all(r.passed for r in results)
    print("\u2705 All checks passed" if all_passed else "\u274c Some checks FAILED")
    return all_passed


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_recording_config() -> dict[str, Any]:
    """Load the ``recording`` section from ``configs/default.yaml``.

    Returns:
        The ``recording`` sub-dict, or an empty dict when the config file
        is absent or contains no ``recording`` key.
    """
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    return cfg.get("recording", {})


def resolve_labels(
    mics: list[int],
    cli_labels: list[str] | None,
    recording_cfg: dict[str, Any],
) -> list[str]:
    """Resolve per-mic labels from CLI → config → fallback.

    Args:
        mics: Ordered list of mic numbers to record.
        cli_labels: Labels provided via ``--labels``, or ``None``.
        recording_cfg: The ``recording`` section of ``configs/default.yaml``.

    Returns:
        Label string for each entry in *mics*.
    """
    if cli_labels is not None:
        return cli_labels
    config_labels: dict[Any, str] = recording_cfg.get("mic_labels", {})
    return [str(config_labels.get(m, f"mic{m}")) for m in mics]


# ---------------------------------------------------------------------------
# list-devices subcommand
# ---------------------------------------------------------------------------


def cmd_list_devices() -> None:
    """Print all available input audio devices with channel counts."""
    recording_cfg = load_recording_config()
    default_device: str = recording_cfg.get("device", _DEFAULT_DEVICE)

    devices = sd.query_devices()
    print("Available input devices:\n")
    found_any = False
    for i, dev in enumerate(devices):
        n_in: int = dev["max_input_channels"]  # type: ignore[index]
        if n_in == 0:
            continue
        found_any = True
        name: str = dev["name"]  # type: ignore[index]
        marker = " ← configured default" if name == default_device else ""
        print(f"  [{i:2d}] {name:<42s} ({n_in} ch in){marker}")

    if not found_any:
        print("  No input devices found.")


# ---------------------------------------------------------------------------
# record subcommand
# ---------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> None:
    """Run a multi-mic recording session until Ctrl-C, then write output files.

    Args:
        args: Parsed CLI arguments from the ``record`` sub-parser.
    """
    recording_cfg = load_recording_config()

    # Resolve device and sample-rate (CLI overrides config, config overrides defaults)
    # Use `is not None` so that index 0 (a valid device) is not treated as falsy.
    device: str | int = (
        args.device if args.device is not None else recording_cfg.get("device", _DEFAULT_DEVICE)
    )
    # Use `is not None` so an explicit --sample-rate 0 is not silently discarded
    # (mirrors the --device fix).
    sample_rate: int = (
        args.sample_rate
        if args.sample_rate is not None
        else int(recording_cfg.get("sample_rate", _DEFAULT_SAMPLE_RATE))
    )
    mics: list[int] = args.mics

    # Validate origin slug matches _NEW_PATTERN in dataset.py: [a-z0-9-]+
    if not re.fullmatch(r"[a-z0-9-]+", args.origin):
        print(
            f"Error: --origin '{args.origin}' must match [a-z0-9-]+ "
            "(lowercase letters, digits, hyphens only)."
        )
        sys.exit(1)
    if args.roast_num < 1:
        print(f"Error: --roast-num must be >= 1, got {args.roast_num}")
        sys.exit(1)
    if args.min_duration < 1:
        print(f"Error: --min-duration must be >= 1, got {args.min_duration}")
        sys.exit(1)
    if sample_rate < 1:
        source = "--sample-rate" if args.sample_rate is not None else "config/default"
        print(f"Error: sample_rate must be >= 1, got {sample_rate} (from {source})")
        sys.exit(1)

    # Validate mic numbers: must be >= 1 and unique
    invalid = [m for m in mics if m < 1]
    if invalid:
        print(f"Error: mic numbers must be >= 1, got: {invalid}")
        sys.exit(1)
    if len(mics) != len(set(mics)):
        dupes = sorted({m for m in mics if mics.count(m) > 1})
        print(f"Error: duplicate mic numbers not allowed: {dupes}")
        sys.exit(1)

    # Labels: CLI → config → "mic{n}"
    if args.labels and len(args.labels) != len(mics):
        print(f"Error: --labels has {len(args.labels)} value(s) but --mics has {len(mics)} mic(s).")
        sys.exit(1)
    labels = resolve_labels(mics, args.labels or None, recording_cfg)

    # Gains
    gains: list[float]
    if args.gains:
        if len(args.gains) != len(mics):
            print(
                f"Error: --gains has {len(args.gains)} value(s) but --mics has {len(mics)} mic(s)."
            )
            sys.exit(1)
        gains = args.gains
    else:
        gains = [1.0] * len(mics)

    # Output paths
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check normal, _partial, and _recording (in-progress temp) candidates — we
    # don't know the session duration yet, so a re-run could silently overwrite
    # a prior partial session, and a leftover _recording file means either a
    # session is already running under this name or a previous one crashed
    # mid-write without being renamed.
    base = f"{args.origin}-roast{args.roast_num}"
    for sfx in ("", "_partial", "_recording"):
        for m in mics:
            p = output_dir / f"mic{m}-{base}{sfx}.wav"
            if p.exists():
                print(f"Error: {p.name} already exists. Remove it or use a different --roast-num.")
                sys.exit(1)
        p = output_dir / f"{base}-session{sfx}.json"
        if p.exists():
            print(f"Error: {p.name} already exists. Remove it or use a different --roast-num.")
            sys.exit(1)

    # Validate device — mic N uses Aggregate Device channel N-1, so open max(mics) channels
    n_channels = max(mics)
    try:
        dev_info = sd.query_devices(device, "input")
        max_in: int = dev_info["max_input_channels"]  # type: ignore[index]
        if max_in < n_channels:
            print(
                f"Error: device '{device}' has {max_in} input channel(s) but "
                f"--mics {mics} requires at least {n_channels}.\n"
                "Run list-devices to check channel counts."
            )
            sys.exit(1)
    except Exception as exc:
        print(
            f"Error: cannot open device '{device}': {exc}\n"
            "Run list-devices to see available devices.  "
            f"Is the Aggregate Device named '{device}' in Audio MIDI Setup?"
        )
        sys.exit(1)

    # Startup banner
    banner = ", ".join(f"{lbl} (mic{m})" for m, lbl in zip(mics, labels, strict=True))
    print(f"Recording : {banner}")
    print(f"Device    : {device} | {sample_rate} Hz | {n_channels} ch open")
    print("Ctrl-C to stop.\n")

    recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = time.monotonic()

    # Temp filenames during recording; renamed to final/_partial names on stop
    # (see the module docstring's "_partial suffix handling" design).
    temp_paths = [output_dir / f"mic{m}-{base}_recording.wav" for m in mics]
    writer = StreamingWriter(
        output_paths=temp_paths, mics=mics, gains=gains, sample_rate=sample_rate
    )
    writer.start()

    # Live monitoring state
    silence_warned: set[int] = set()  # mic numbers already warned about silence
    silence_checked = False  # True once the initial 5-second silence check has run

    # Rate-limit status warnings to avoid flooding stderr on transient glitches.
    _status_warn_interval = 5.0
    _last_status_warn: float = 0.0

    def _callback(
        indata: np.ndarray,
        _frames: int,
        _cb_time: object,
        status: object,
    ) -> None:
        nonlocal _last_status_warn
        if status:
            now = time.monotonic()
            if now - _last_status_warn >= _status_warn_interval:
                print(f"Warning: audio input status: {status}", file=sys.stderr)
                _last_status_warn = now
        # No I/O, no numpy ops here — just a non-blocking handoff to the
        # writer thread's queue, per the #49 design constraint.
        writer.put(indata.copy())

    # SIGTERM must trigger the same clean shutdown as Ctrl-C: stop the input
    # stream and let the writer thread flush + close every SoundFile handle
    # before the process exits, instead of losing whatever is still queued.
    stop_requested = threading.Event()

    def _handle_sigterm(_signum: int, _frame: object) -> None:
        stop_requested.set()

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=n_channels,
            device=device,
            callback=_callback,
        ):
            next_heartbeat = start + 30.0
            while not stop_requested.is_set():
                time.sleep(0.25)
                now = time.monotonic()
                elapsed = now - start

                # Initial silence check after _SILENCE_CHECK_AFTER_SEC seconds.
                # Fires once regardless of --quiet so the warning always reaches stderr.
                if not silence_checked and elapsed >= _SILENCE_CHECK_AFTER_SEC:
                    silence_warned, silence_checked = _run_initial_silence_check(
                        writer.stats_window(), mics, labels, gains, silence_warned
                    )

                # 30-second heartbeat: live stats + silence re-check.
                if now >= next_heartbeat:
                    hb_stats = _mic_stats_from_chunks(writer.stats_window(), mics, gains)
                    # Re-check for silent mics on each heartbeat (always, even --quiet)
                    silence_warned = _check_silent_mics(hb_stats, mics, labels, silence_warned)
                    if not args.quiet:
                        print(_format_heartbeat(elapsed, hb_stats, mics, labels))
                    next_heartbeat = now + 30.0
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    duration = time.monotonic() - start
    print(f"\nStopped after {duration:.1f}s.")

    # Flush and close every SoundFile handle. Safe to call unconditionally —
    # even a zero-frame session gets valid (empty) WAV headers closed cleanly.
    writer.stop_and_join()

    if writer.frames_written == 0:
        print("No audio captured.")
        for p in temp_paths:
            p.unlink(missing_ok=True)
        return

    # Use actual sample count for duration — wall-clock time includes PortAudio
    # initialisation latency (~1-2s) before the first callback fires.
    audio_duration = writer.frames_written / sample_rate

    # Short-session guard
    is_partial = audio_duration < args.min_duration
    suffix = "_partial" if is_partial else ""
    if is_partial:
        print(
            f"⚠️  Duration ({audio_duration:.1f}s) is shorter than "
            f"--min-duration ({args.min_duration}s) — saving with _partial suffix"
        )

    print()

    # Rename each mic's temp _recording file to its final name.
    mic_meta: list[dict[str, Any]] = []
    final_paths: list[Path] = []
    for temp_path, m, label, gain in zip(temp_paths, mics, labels, gains, strict=True):
        filename = f"mic{m}-{base}{suffix}.wav"
        final_path = output_dir / filename
        temp_path.rename(final_path)
        final_paths.append(final_path)
        print(f"  Wrote {filename}")
        mic_meta.append({"mic_num": m, "label": label, "gain": gain, "file": filename})

    # Write session JSON only after every rename has completed.
    session_filename = f"{base}-session{suffix}.json"
    session_path = output_dir / session_filename
    session_data: dict[str, Any] = {
        "origin": args.origin,
        "roast_num": args.roast_num,
        "sample_rate": sample_rate,
        "duration_sec": round(audio_duration, 2),
        "recorded_at": recorded_at,
        "mics": mic_meta,
    }
    with session_path.open("w") as f:
        json.dump(session_data, f, indent=2)
        f.write("\n")
    print(f"  Wrote {session_filename}")

    # Full-session level summary, read back from the finalized WAV files so
    # the summary reflects exactly what was written to disk.
    summary_audio = [sf.read(str(p), dtype="float32", always_2d=False)[0] for p in final_paths]
    summary_stats = [
        {k: v for k, v in zip(("peak", "rms"), _dbfs(np.asarray(a, dtype=np.float32)), strict=True)}
        for a in summary_audio
    ]
    _print_session_summary(summary_stats, mics, labels)

    n = len(mics)
    print(f"\nDone. {audio_duration:.1f}s audio -> {n} WAV(s) + session JSON in {output_dir}")

    if args.verify:
        results = verify_recording_session(session_path, output_dir)
        all_passed = print_verification_report(results)
        if not all_passed:
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the multi-mic recording tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-mic synchronized recording for coffee roasting. "
            "macOS only — requires a CoreAudio Aggregate Device."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list-devices
    sub.add_parser("list-devices", help="Print available input audio devices")

    # record
    rec = sub.add_parser("record", help="Record a multi-mic roasting session")
    rec.add_argument(
        "--origin",
        required=True,
        help="Coffee bean origin slug, e.g. 'brazil'",
    )
    rec.add_argument(
        "--roast-num",
        type=int,
        required=True,
        help="Roast number, e.g. 7",
    )
    rec.add_argument(
        "--device",
        type=lambda v: int(v) if v.isdigit() else v,
        default=None,
        help=f"Aggregate device name or index (default: config → '{_DEFAULT_DEVICE}')",
    )
    rec.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    rec.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=f"Capture sample rate in Hz (default: config → {_DEFAULT_SAMPLE_RATE})",
    )
    rec.add_argument(
        "--mics",
        type=int,
        nargs="+",
        default=list(_DEFAULT_MICS),
        metavar="N",
        help="Mic numbers to record; mic N = Aggregate Device channel N-1 (default: 1 2)",
    )
    rec.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=None,
        metavar="G",
        help="Per-mic digital gain multipliers, must match --mics length (default: 1.0 each)",
    )
    rec.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        metavar="L",
        help="Per-mic hardware labels, must match --mics length (default: from config)",
    )
    rec.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the 30-second progress heartbeat",
    )
    rec.add_argument(
        "--min-duration",
        type=int,
        default=_DEFAULT_MIN_DURATION_SEC,
        metavar="SEC",
        help=(
            f"Sessions shorter than this (seconds) are saved with a _partial suffix "
            f"(default: {_DEFAULT_MIN_DURATION_SEC})"
        ),
    )
    rec.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run post-recording verification (peak/RMS/dBFS, sample-lock, "
            "balance, duration, session JSON) after writing and print a "
            "report. Off by default to keep the tool fast; exits non-zero "
            "if any check fails."
        ),
    )

    parsed = parser.parse_args()

    if parsed.command == "list-devices":
        cmd_list_devices()
    elif parsed.command == "record":
        cmd_record(parsed)


if __name__ == "__main__":
    main()
