"""Propagate primary-mic annotations to all paired mics in a recording session.

Reads session JSON files produced by ``record_mics.py`` and copies the primary
microphone's per-file annotation JSON (output of ``convert_labelstudio_export``)
to every other microphone in the session.  Because recordings are captured through
a CoreAudio Aggregate Device with Drift Correction, event timestamps are
sample-locked across all channels and require no time adjustment.

The script slots between ``convert_labelstudio_export.py`` and ``chunk_audio.py``
in the data preparation pipeline.  Existing recordings that have no session JSON
are untouched.

Usage::

    # Propagate with defaults (session-dir=data/raw, labels-dir=data/labels)
    python scripts/propagate_annotations.py

    # Preview without writing
    python scripts/propagate_annotations.py --dry-run

    # Overwrite existing paired annotation JSONs
    python scripts/propagate_annotations.py --overwrite

    # Custom paths or primary mic
    python scripts/propagate_annotations.py \\
        --session-dir data/raw \\
        --labels-dir data/labels \\
        --audio-dir data/raw \\
        --primary-mic 1
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import librosa

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with path.open("r") as f:
        return json.load(f)  # type: ignore[no-any-return]


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Serialise *data* to a JSON file with 2-space indentation.

    Args:
        path: Destination file path.  Parent directories are created if absent.
        data: Data to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def find_session_files(session_dir: Path) -> list[Path]:
    """Return sorted paths of all session JSON files in *session_dir*.

    Discovers both complete sessions (``*-session.json``) and partial
    sessions (``*-session_partial.json``) so that explicitly-annotated
    short sessions can also be propagated.

    Args:
        session_dir: Directory to search.

    Returns:
        Sorted list of matching paths.
    """
    return sorted(
        list(session_dir.glob("*-session.json")) + list(session_dir.glob("*-session_partial.json"))
    )


def get_audio_duration(audio_path: Path) -> float:
    """Return the duration of *audio_path* in seconds via librosa.

    Args:
        audio_path: Path to the WAV file.

    Returns:
        Duration in seconds.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError: If librosa cannot read the file.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"WAV file not found: {audio_path}")
    try:
        return librosa.get_duration(path=str(audio_path))  # type: ignore[no-any-return]
    except Exception as exc:
        raise RuntimeError(f"Failed to read duration for {audio_path}") from exc


# ---------------------------------------------------------------------------
# Core propagation logic
# ---------------------------------------------------------------------------


def propagate_session(
    session_path: Path,
    labels_dir: Path,
    audio_dir: Path,
    primary_mic: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Propagate annotations for a single recording session.

    Reads *session_path*, locates the primary mic's annotation JSON, and writes
    an identical annotation JSON for every other mic listed in the session.
    The ``audio_file`` and ``duration`` fields are updated per mic; the
    ``annotations`` list is deep-copied unchanged.

    Args:
        session_path: Path to a ``*-session.json`` file from ``record_mics.py``.
        labels_dir: Directory containing per-file annotation JSONs and where
            propagated JSONs are written.
        audio_dir: Directory containing the recorded WAV files.  Used only to
            read the duration of each paired mic's WAV via librosa.
        primary_mic: Mic number whose annotation JSON is the annotation source.
        overwrite: When ``False``, skip paired mics whose annotation JSON already
            exists; when ``True``, overwrite them.
        dry_run: When ``True``, print intended writes without touching disk.

    Returns:
        ``(written, skipped)`` counts for this session.
    """
    session: dict[str, Any] = load_json(session_path)
    origin: str = session["origin"]
    roast_num: int = int(session["roast_num"])
    mics: list[dict[str, Any]] = session["mics"]

    paired = [m for m in mics if int(m["mic_num"]) != primary_mic]

    paired_str = ", ".join(f"mic{m['mic_num']}" for m in paired)
    suffix = f" → {paired_str}" if paired else ""
    print(f"  {session_path.name}: mic{primary_mic} (primary){suffix}")

    if not paired:
        print("    ℹ️  No paired mics in this session — skipping")
        return 0, 0

    # Resolve primary annotation path from the session's mics list so that any
    # filename suffix (e.g. _partial) added by record_mics.py is respected.
    primary_entry = next((m for m in mics if int(m["mic_num"]) == primary_mic), None)
    if primary_entry:
        primary_stem = Path(primary_entry["file"]).stem
    else:
        # primary mic was not part of this session (recorded separately)
        primary_stem = f"mic{primary_mic}-{origin}-roast{roast_num}"
    primary_label_path = labels_dir / f"{primary_stem}.json"
    if not primary_label_path.exists():
        print(f"    ⚠️  Primary annotation not found: {primary_label_path} — skipping")
        return 0, len(paired)

    primary_annotation: dict[str, Any] = load_json(primary_label_path)
    annotations: list[dict[str, Any]] = primary_annotation["annotations"]
    sample_rate: int = int(primary_annotation.get("sample_rate", 44100))

    written = 0
    skipped = 0

    for mic in paired:
        mic_num: int = int(mic["mic_num"])
        # Use the filename recorded in the session JSON — preserves any suffix
        # (e.g. _partial) and stays consistent with convert_labelstudio_export.py.
        wav_filename: str = mic["file"]
        wav_stem = Path(wav_filename).stem
        target_path = labels_dir / f"{wav_stem}.json"

        if target_path.exists() and not overwrite:
            print(
                f"    ⏭️  {target_path.name} already exists — skipping (use --overwrite to replace)"
            )
            skipped += 1
            continue

        wav_path = audio_dir / wav_filename
        try:
            duration = get_audio_duration(wav_path)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"    ⚠️  {exc} — skipping mic{mic_num}")
            skipped += 1
            continue

        paired_annotation: dict[str, Any] = {
            "audio_file": wav_filename,
            "duration": duration,
            "sample_rate": sample_rate,
            "annotations": copy.deepcopy(annotations),
        }

        if dry_run:
            print(f"    [dry-run] Would write {target_path.name} (annotations: {len(annotations)})")
        else:
            write_json(target_path, paired_annotation)
            print(f"    ✅ Wrote {target_path.name} (annotations: {len(annotations)})")
        written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for annotation propagation."""
    parser = argparse.ArgumentParser(
        description=(
            "Propagate primary-mic annotation JSONs to all paired mics in a session. "
            "Session JSONs are produced by record_mics.py."
        )
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing *-session.json files (default: data/raw)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/labels"),
        help="Directory to read/write per-file annotation JSONs (default: data/labels)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing WAV files for duration lookup (default: data/raw)",
    )
    parser.add_argument(
        "--primary-mic",
        type=int,
        default=1,
        help="Mic number whose annotation JSON is the source of truth (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing paired annotation JSONs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended writes without writing anything",
    )
    args = parser.parse_args()

    if not args.session_dir.exists():
        print(f"❌ Session directory not found: {args.session_dir}")
        return

    session_files = find_session_files(args.session_dir)
    if not session_files:
        print(
            f"No session files (*-session.json or *-session_partial.json) found"
            f" in {args.session_dir}"
        )
        return

    print(f"Found {len(session_files)} session(s) in {args.session_dir}")
    if args.dry_run:
        print("[dry-run mode — nothing will be written]")

    total_written = 0
    total_skipped = 0

    for session_path in session_files:
        written, skipped = propagate_session(
            session_path=session_path,
            labels_dir=args.labels_dir,
            audio_dir=args.audio_dir,
            primary_mic=args.primary_mic,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_written += written
        total_skipped += skipped

    verb = "Would propagate" if args.dry_run else "Propagated"
    print(f"\n{verb} {total_written} annotation file(s). {total_skipped} skipped.")


if __name__ == "__main__":
    main()
