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
import re
import sys
import threading
import time
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
    device: str | int = args.device or recording_cfg.get("device", _DEFAULT_DEVICE)
    sample_rate: int = args.sample_rate or int(
        recording_cfg.get("sample_rate", _DEFAULT_SAMPLE_RATE)
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

    # Check both normal and _partial candidates — we don't know the session
    # duration yet, so a re-run could silently overwrite a prior partial session.
    base = f"{args.origin}-roast{args.roast_num}"
    for sfx in ("", "_partial"):
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

    # Accumulate audio chunks via callback
    chunks: list[np.ndarray] = []
    lock = threading.Lock()
    recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = time.monotonic()

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
        with lock:
            chunks.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=n_channels,
            device=device,
            callback=_callback,
        ):
            next_heartbeat = start + 30.0
            while True:
                time.sleep(0.25)
                if not args.quiet:
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        elapsed = now - start
                        mins, secs = divmod(int(elapsed), 60)
                        print(f"[{mins:02d}:{secs:02d}] Recording...")
                        next_heartbeat = now + 30.0
    except KeyboardInterrupt:
        pass

    duration = time.monotonic() - start
    print(f"\nStopped after {duration:.1f}s.")

    # Copy chunk list under lock to avoid a race with any in-flight callback
    # during stream shutdown (stream.stop() is ordered, but no memory barrier
    # without the lock).
    with lock:
        recorded_chunks = list(chunks)

    if not recorded_chunks:
        print("No audio captured.")
        return

    recording = np.concatenate(recorded_chunks, axis=0)
    # Use actual sample count for duration — wall-clock time includes PortAudio
    # initialisation latency (~1-2s) before the first callback fires.
    audio_duration = len(recording) / sample_rate

    # Short-session guard
    is_partial = audio_duration < args.min_duration
    suffix = "_partial" if is_partial else ""
    if is_partial:
        print(
            f"⚠️  Duration ({audio_duration:.1f}s) is shorter than "
            f"--min-duration ({args.min_duration}s) — saving with _partial suffix"
        )

    print()

    # Write per-mic WAV files
    mic_meta: list[dict[str, Any]] = []
    for m, label, gain in zip(mics, labels, gains, strict=True):
        ch_idx = m - 1
        audio: np.ndarray = np.clip(recording[:, ch_idx] * np.float32(gain), -1.0, 1.0).astype(
            np.float32, copy=False
        )
        filename = f"mic{m}-{args.origin}-roast{args.roast_num}{suffix}.wav"
        sf.write(str(output_dir / filename), audio, sample_rate, subtype="FLOAT")
        print(f"  Wrote {filename}")
        mic_meta.append({"mic_num": m, "label": label, "gain": gain, "file": filename})

    # Write session JSON
    session_filename = f"{args.origin}-roast{args.roast_num}-session{suffix}.json"
    session_data: dict[str, Any] = {
        "origin": args.origin,
        "roast_num": args.roast_num,
        "sample_rate": sample_rate,
        "duration_sec": round(audio_duration, 2),
        "recorded_at": recorded_at,
        "mics": mic_meta,
    }
    with (output_dir / session_filename).open("w") as f:
        json.dump(session_data, f, indent=2)
        f.write("\n")
    print(f"  Wrote {session_filename}")
    n = len(mics)
    print(f"\nDone. {audio_duration:.1f}s audio -> {n} WAV(s) + session JSON in {output_dir}")


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

    parsed = parser.parse_args()

    if parsed.command == "list-devices":
        cmd_list_devices()
    elif parsed.command == "record":
        cmd_record(parsed)


if __name__ == "__main__":
    main()
