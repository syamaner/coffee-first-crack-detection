"""Propagate primary-mic annotations to all paired mics in a recording session.

For the bench workflow, reads session JSON files produced by ``record_mics.py``
and copies a primary annotation across channels captured through one CoreAudio
Aggregate Device. For staged coffee-roaster-mcp captures, reads the capture
manifest and creates deterministic mic2 annotations with explicit independent-
clock uncertainty provenance. MCP streams are not sample-locked.

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
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import librosa

from coffee_first_crack.data_prep.corpus_manifest import (
    index_manifest_streams,
    load_capture_manifest,
    resolve_within,
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a staged recording."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    # Per-mic WAV/annotation files start with mic{digits}- (e.g. mic1-brazil-roast5.wav).
    # Filter them out so we only return session files produced by record_mics.py.
    # Use mic\d+- (digits + hyphen) to avoid excluding a session whose origin
    # slug starts with e.g. 'mic' but is not followed by a digit-hyphen prefix.
    _mic_prefix = re.compile(r"^mic\d+-")
    return sorted(
        p
        for p in list(session_dir.glob("*-session.json"))
        + list(session_dir.glob("*-session_partial.json"))
        if not _mic_prefix.match(p.name)
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


def propagate_manifest(
    manifest_path: Path,
    labels_dir: Path,
    staging_root: Path,
    *,
    audio_root: Path | None = None,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Derive mic2 annotations for every staged MCP capture pair.

    Boundary timestamps remain on the mic1 time axis for auditability, but
    historical MCP captures have no measured cross-stream start offset. Final
    WAV duration differences are diagnostic only and cannot bound that offset,
    so every derived mic2 chunk is excluded from training until a verified
    alignment method supplies a defensible finite uncertainty.

    Args:
        manifest_path: Staged ``capture_manifest.json``.
        labels_dir: Directory containing converted mic1 annotations and where
            mic2 annotations are written.
        staging_root: Root containing the staged ``mic1`` and ``mic2`` WAVs.
        audio_root: Common root used by ``chunk_audio``. Defaults to
            ``staging_root``; the derived ``audio_file`` is relative to it.
        overwrite: Replace an existing derived mic2 annotation when true.
        dry_run: Report intended writes without changing files.

    Returns:
        ``(written, skipped)`` counts.

    Raises:
        FileNotFoundError: If any required mic1 annotation or staged mic2 WAV is
            missing.
        ValueError: If a pair is malformed or ambiguous.
        FileExistsError: If a target exists and overwrite is false.
    """
    manifest = load_capture_manifest(manifest_path)
    manifest_staging_root = Path(manifest["staging_root"]).resolve()
    if staging_root.resolve() != manifest_staging_root:
        raise ValueError(
            f"--staging-root does not match the manifest: "
            f"{staging_root.resolve()} != {manifest_staging_root}"
        )
    _, sessions_by_pair = index_manifest_streams(manifest)
    resolved_audio_root = (audio_root or staging_root).resolve()
    skipped = 0
    plans: list[tuple[Path, Path, dict[str, Any]]] = []

    for pair_id in sorted(sessions_by_pair):
        session = sessions_by_pair[pair_id]
        by_mic = {stream["mic_num"]: stream for stream in session["streams"]}
        if set(by_mic) != {1, 2}:
            raise ValueError(f"MCP pair {pair_id} must contain exactly mic1 and mic2")
        primary = by_mic[1]
        target = by_mic[2]
        primary_path = labels_dir / f"{Path(primary['staged_relative_path']).stem}.json"
        target_path = labels_dir / f"{Path(target['staged_relative_path']).stem}.json"
        if not primary_path.is_file():
            raise FileNotFoundError(
                f"Missing human mic1 annotation for pair {pair_id}: {primary_path}"
            )
        primary_audio = resolve_within(staging_root, primary["staged_relative_path"])
        target_audio = resolve_within(staging_root, target["staged_relative_path"])
        if not primary_audio.is_file():
            raise FileNotFoundError(f"Missing staged mic1 WAV for pair {pair_id}: {primary_audio}")
        if not target_audio.is_file():
            raise FileNotFoundError(f"Missing staged mic2 WAV for pair {pair_id}: {target_audio}")
        for stream, audio_path in ((primary, primary_audio), (target, target_audio)):
            if sha256_file(audio_path) != stream["sha256"]:
                raise ValueError(f"Staged WAV does not match capture manifest: {audio_path}")
        if not target_audio.is_relative_to(resolved_audio_root):
            raise ValueError(
                f"Staged mic2 WAV is outside the configured audio root: {target_audio}"
            )
        target_audio_relative = target_audio.relative_to(resolved_audio_root).as_posix()
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Derived annotation already exists for pair {pair_id}: {target_path}"
            )

        primary_annotation = load_json(primary_path)
        if primary_annotation.get("pair_id") != pair_id or primary_annotation.get("mic_num") != 1:
            raise ValueError(f"Human annotation pair identity mismatch: {primary_path}")
        primary_provenance = primary_annotation.get("provenance")
        if (
            not isinstance(primary_provenance, dict)
            or primary_provenance.get("source_audio_sha256") != primary["sha256"]
        ):
            raise ValueError(f"Human annotation source checksum mismatch: {primary_path}")
        annotations = primary_annotation.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"Human annotation has invalid annotations list: {primary_path}")
        derived_annotations = copy.deepcopy(annotations)
        for annotation in derived_annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"Human annotation region is not an object: {primary_path}")
            if annotation.get("label") != "first_crack":
                raise ValueError(f"Human annotation has an unsupported label: {primary_path}")
            start = annotation.get("start_time")
            end = annotation.get("end_time")
            if (
                not isinstance(start, (int, float))
                or isinstance(start, bool)
                or not isinstance(end, (int, float))
                or isinstance(end, bool)
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or start < 0
                or end <= start
                or end > primary["duration_seconds"]
            ):
                raise ValueError(f"Human annotation has invalid boundaries: {primary_path}")
            annotation["confidence"] = "derived_unaligned_excluded_from_training"
        derived: dict[str, Any] = {
            "audio_file": target_audio_relative,
            "duration": target["duration_seconds"],
            "sample_rate": target["sample_rate"],
            "annotations": derived_annotations,
            "pair_id": pair_id,
            "mic_num": 2,
            "origin": session["origin"],
            "roast_num": session["roast_num"],
            "original_filename": target["original_filename"],
            "provenance": {
                "annotation_source": "derived_from_paired_mic",
                "pair_id": pair_id,
                "derived_from": primary["staged_relative_path"],
                "derivation_method": "copy_timestamps_for_audit_only",
                "boundary_time_axis": "mic1_session_axis",
                "alignment": "independent_clocks_not_sample_locked",
                "observed_pair_duration_delta_seconds": session["observed_duration_delta_seconds"],
                "alignment_uncertainty_seconds": None,
                "alignment_uncertainty_status": (
                    "unbounded_historical_missing_stream_start_offsets"
                ),
                "uncertainty_basis": "duration_delta_is_diagnostic_not_an_alignment_bound",
                "training_policy": "exclude_all_derived_mic2_without_verified_alignment",
                "exact_stream_start_offsets_available": False,
                "source_audio_sha256": target["sha256"],
            },
        }
        plans.append((target_path, primary_path, derived))

    for target_path, primary_path, derived in plans:
        if dry_run:
            print(f"[dry-run] Would derive {target_path.name} from {primary_path.name}")
        else:
            write_json(target_path, derived)
            print(f"Derived {target_path.name} from {primary_path.name} (excluded: unaligned)")

    return len(plans), skipped


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Staged MCP capture_manifest.json; enables uncertainty-aware MCP derivation",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Staged MCP corpus root (defaults to the manifest parent)",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Common chunker audio root; derived audio paths are relative to it",
    )
    args = parser.parse_args()

    if args.manifest is not None:
        staging_root = args.staging_root or args.manifest.parent
        written, skipped = propagate_manifest(
            args.manifest,
            args.labels_dir,
            staging_root,
            audio_root=args.audio_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        verb = "Would derive" if args.dry_run else "Derived"
        print(f"{verb} {written} MCP annotation file(s). {skipped} skipped.")
        return

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
