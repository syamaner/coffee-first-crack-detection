"""Stage coffee-roaster-mcp captures without flattening session identity.

The capture root is treated as immutable. Each capture-directory UUID becomes
the physical-roast ``pair_id`` and is prefixed to staged filenames. Mic1 WAVs
are staged into a dedicated Label Studio import directory; mic2 WAVs are staged
separately as automatic-annotation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from coffee_first_crack.data_prep.corpus_manifest import (
    CaptureManifest,
    SessionRecord,
    StreamRecord,
    safe_relative_path,
)

PAIR_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MIC_FILENAME_RE = re.compile(r"^mic(?P<mic_num>[1-9]\d*)-(?P<slug>.+)\.wav$")


@dataclass(frozen=True)
class SourceFileState:
    """Immutable evidence captured for one source file."""

    path: Path
    size_bytes: int
    modified_ns: int
    sha256: str


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_source(path: Path) -> SourceFileState:
    """Capture byte and metadata evidence for a source file.

    Args:
        path: Source file.

    Returns:
        Source state including content digest.
    """
    stat = path.stat()
    return SourceFileState(path, stat.st_size, stat.st_mtime_ns, sha256_file(path))


def verify_source_unchanged(before: SourceFileState) -> None:
    """Fail if a source file differs from its pre-staging state.

    Args:
        before: Evidence captured before staging.

    Raises:
        RuntimeError: If size, mtime, or bytes changed.
    """
    after = snapshot_source(before.path)
    if after != before:
        raise RuntimeError(f"Source file changed during staging: {before.path}")


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots.

    Args:
        path: JSON file path.

    Returns:
        Parsed object.

    Raises:
        ValueError: If JSON is malformed or not an object.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            value: Any = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to parse JSON sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON sidecar must contain an object: {path}")
    return cast(dict[str, Any], value)


def wav_metadata(path: Path) -> tuple[float, int, int, int, int]:
    """Read PCM WAV metadata without decoding audio.

    Args:
        path: WAV file path.

    Returns:
        ``(duration_seconds, sample_rate, channels, sample_width_bytes,
        frame_count)``.

    Raises:
        ValueError: If the WAV header is invalid.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                raise ValueError(f"Invalid WAV sample rate in {path}")
            frame_count = handle.getnframes()
            return (
                frame_count / rate,
                rate,
                handle.getnchannels(),
                handle.getsampwidth(),
                frame_count,
            )
    except (OSError, wave.Error) as exc:
        raise ValueError(f"Unable to read WAV header {path}: {exc}") from exc


def _require_filename(value: object, *, field: str) -> str:
    """Validate an untrusted sidecar filename.

    Args:
        value: Candidate filename.
        field: Field name for error messages.

    Returns:
        Safe basename.

    Raises:
        ValueError: If the value contains a path or traversal.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a filename string")
    safe = safe_relative_path(value)
    if len(safe.parts) != 1 or safe.name != value:
        raise ValueError(f"{field} must be a basename, got {value!r}")
    return value


def _find_annotation_sidecar(session_dir: Path) -> Path:
    candidates = sorted(session_dir.glob("*-session.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one *-session.json in {session_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _validate_session(
    session_dir: Path,
    staging_root: Path,
) -> tuple[SessionRecord, list[SourceFileState]]:
    pair_id = session_dir.name
    if not PAIR_ID_RE.fullmatch(pair_id):
        raise ValueError(f"Capture directory is not a 32-character lowercase UUID: {session_dir}")
    if session_dir.is_symlink():
        raise ValueError(f"Capture session directories may not be symlinks: {session_dir}")

    annotation_sidecar = _find_annotation_sidecar(session_dir)
    recording_sidecar = session_dir / "roast.recording.json"
    if not recording_sidecar.is_file() or recording_sidecar.is_symlink():
        raise ValueError(f"Missing regular recording sidecar: {recording_sidecar}")
    if annotation_sidecar.is_symlink():
        raise ValueError(f"Session sidecar may not be a symlink: {annotation_sidecar}")

    annotation = load_json_object(annotation_sidecar)
    recording = load_json_object(recording_sidecar)
    origin = annotation.get("origin")
    roast_num = annotation.get("roast_num")
    sample_rate = annotation.get("sample_rate")
    mics = annotation.get("mics")
    if not isinstance(origin, str) or not origin or Path(origin).name != origin:
        raise ValueError(f"Invalid origin in {annotation_sidecar}")
    if not isinstance(roast_num, int) or isinstance(roast_num, bool) or roast_num < 1:
        raise ValueError(f"Invalid roast_num in {annotation_sidecar}")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate in {annotation_sidecar}")
    if not isinstance(mics, list) or len(mics) != 2:
        raise ValueError(f"Expected exactly two mics in {annotation_sidecar}")
    if recording.get("schema_version") != 2 or recording.get("session_id") != pair_id:
        raise ValueError(f"Recording sidecar session identity mismatch: {recording_sidecar}")
    recording_streams = recording.get("streams")
    if not isinstance(recording_streams, list) or len(recording_streams) != 2:
        raise ValueError(f"Expected exactly two recording streams in {recording_sidecar}")

    recording_by_filename: dict[str, dict[str, Any]] = {}
    for item in recording_streams:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid recording stream in {recording_sidecar}")
        filename = _require_filename(item.get("wav_filename"), field="streams[].wav_filename")
        if filename in recording_by_filename:
            raise ValueError(f"Duplicate recording stream filename: {filename}")
        recording_by_filename[filename] = cast(dict[str, Any], item)

    stream_records: list[StreamRecord] = []
    source_states = [snapshot_source(annotation_sidecar), snapshot_source(recording_sidecar)]
    mic_numbers: set[int] = set()
    staged_destinations: set[str] = set()
    for mic in mics:
        if not isinstance(mic, dict):
            raise ValueError(f"Invalid mic entry in {annotation_sidecar}")
        mic_num = mic.get("mic_num")
        label = mic.get("label")
        if not isinstance(mic_num, int) or isinstance(mic_num, bool) or mic_num not in {1, 2}:
            raise ValueError(f"Invalid mic number in {annotation_sidecar}: {mic_num!r}")
        if mic_num in mic_numbers:
            raise ValueError(f"Duplicate mic number in {annotation_sidecar}: {mic_num}")
        mic_numbers.add(mic_num)
        if not isinstance(label, str) or not label:
            raise ValueError(f"Invalid mic label in {annotation_sidecar}")
        filename = _require_filename(mic.get("file"), field="mics[].file")
        match = MIC_FILENAME_RE.fullmatch(filename)
        expected_suffix = f"{origin}-roast{roast_num}.wav"
        if (
            match is None
            or int(match.group("mic_num")) != mic_num
            or not filename.endswith(expected_suffix)
        ):
            raise ValueError(f"Inconsistent mic filename metadata: {filename}")
        source_path = session_dir / filename
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"Missing regular WAV file: {source_path}")
        stream_meta = recording_by_filename.get(filename)
        if stream_meta is None:
            raise ValueError(f"WAV missing from recording sidecar: {filename}")
        if stream_meta.get("sample_rate") != sample_rate:
            raise ValueError(f"Sample-rate mismatch for {source_path}")

        staged_name = f"{pair_id}__{filename}"
        staged_relative = f"mic{mic_num}/{staged_name}"
        if staged_relative in staged_destinations:
            raise ValueError(f"Duplicate staged destination: {staged_relative}")
        staged_destinations.add(staged_relative)
        source_state = snapshot_source(source_path)
        source_states.append(source_state)
        duration, wav_rate, channels, sample_width, frame_count = wav_metadata(source_path)
        recorded_duration = stream_meta.get("duration_seconds")
        if not isinstance(recorded_duration, (int, float)) or isinstance(recorded_duration, bool):
            raise ValueError(f"Invalid duration metadata for {source_path}")
        if abs(duration - float(recorded_duration)) > 0.01:
            raise ValueError(f"WAV duration conflicts with recording sidecar: {source_path}")
        expected_metadata = {
            "sample_rate": wav_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "frame_count": frame_count,
        }
        for field, expected in expected_metadata.items():
            if stream_meta.get(field) != expected:
                raise ValueError(
                    f"WAV {field} conflicts with recording sidecar for {source_path}: "
                    f"header={expected}, sidecar={stream_meta.get(field)!r}"
                )
        stream_records.append(
            StreamRecord(
                mic_num=mic_num,
                label=label,
                original_filename=filename,
                source_path=str(source_path),
                staged_relative_path=staged_relative,
                size_bytes=source_state.size_bytes,
                sha256=source_state.sha256,
                duration_seconds=duration,
                sample_rate=sample_rate,
            )
        )

    if mic_numbers != {1, 2} or set(recording_by_filename) != {
        stream["original_filename"] for stream in stream_records
    }:
        raise ValueError(f"Inconsistent two-mic metadata in {session_dir}")
    stream_records.sort(key=lambda item: item["mic_num"])
    duration_delta = abs(
        stream_records[0]["duration_seconds"] - stream_records[1]["duration_seconds"]
    )
    session_record = SessionRecord(
        pair_id=pair_id,
        origin=origin,
        roast_num=roast_num,
        source_session_dir=str(session_dir),
        session_sidecar_source_path=str(annotation_sidecar),
        session_sidecar_staged_relative_path=f"sessions/{pair_id}__session.json",
        recording_sidecar_source_path=str(recording_sidecar),
        recording_sidecar_staged_relative_path=f"sessions/{pair_id}__roast.recording.json",
        observed_duration_delta_seconds=duration_delta,
        streams=stream_records,
    )
    for relative in (
        session_record["session_sidecar_staged_relative_path"],
        session_record["recording_sidecar_staged_relative_path"],
    ):
        safe_relative_path(relative)
        if (staging_root / relative).exists():
            raise FileExistsError(f"Staged destination already exists: {staging_root / relative}")
    return session_record, source_states


def discover_sessions(capture_root: Path, staging_root: Path) -> list[SessionRecord]:
    """Discover and validate all MCP capture sessions.

    Args:
        capture_root: Immutable capture root.
        staging_root: Destination root, used to detect collisions.

    Returns:
        Sorted validated session records.

    Raises:
        ValueError: If the root or any session is invalid.
        FileExistsError: If a destination already exists.
    """
    if not capture_root.is_dir() or capture_root.is_symlink():
        raise ValueError(f"Capture root must be a regular directory: {capture_root}")
    session_dirs = sorted(path for path in capture_root.iterdir() if path.is_dir())
    if not session_dirs:
        raise ValueError(f"No capture sessions found in {capture_root}")

    sessions: list[SessionRecord] = []
    pair_ids: set[str] = set()
    destinations: set[str] = set()
    for session_dir in session_dirs:
        session, _ = _validate_session(session_dir, staging_root)
        if session["pair_id"] in pair_ids:
            raise ValueError(f"Duplicate pair_id: {session['pair_id']}")
        pair_ids.add(session["pair_id"])
        for stream in session["streams"]:
            relative = stream["staged_relative_path"]
            if relative in destinations or (staging_root / relative).exists():
                raise FileExistsError(f"Duplicate staged destination: {relative}")
            destinations.add(relative)
        sessions.append(session)
    return sessions


def stage_captures(
    capture_root: Path,
    staging_root: Path,
    *,
    dry_run: bool = False,
) -> CaptureManifest:
    """Validate and stage an MCP capture corpus.

    Args:
        capture_root: Immutable MCP capture root.
        staging_root: New destination directory.
        dry_run: Validate and report without writing when true.

    Returns:
        The manifest that was or would be written.

    Raises:
        FileExistsError: If ``staging_root`` already contains data.
        RuntimeError: If copy verification or source-integrity checks fail.
        ValueError: If source validation fails.
    """
    capture_root_resolved = capture_root.resolve()
    staging_root_resolved = staging_root.resolve()
    if staging_root.is_symlink():
        raise ValueError(f"Staging root may not be a symlink: {staging_root}")
    if staging_root_resolved.is_relative_to(capture_root_resolved):
        raise ValueError(f"Staging root must be outside the immutable capture root: {staging_root}")
    if staging_root.exists() and any(staging_root.iterdir()):
        raise FileExistsError(f"Staging root must be absent or empty: {staging_root}")
    sessions = discover_sessions(capture_root, staging_root)
    all_states: list[SourceFileState] = []
    for session_dir in sorted(path for path in capture_root.iterdir() if path.is_dir()):
        _, states = _validate_session(session_dir, staging_root)
        all_states.extend(states)

    max_delta = max(session["observed_duration_delta_seconds"] for session in sessions)
    manifest = CaptureManifest(
        schema_version=1,
        source_root=str(capture_root.resolve()),
        staging_root=str(staging_root.resolve()),
        session_count=len(sessions),
        stream_count=sum(len(session["streams"]) for session in sessions),
        mic1_task_count=sum(
            stream["mic_num"] == 1 for session in sessions for stream in session["streams"]
        ),
        max_observed_duration_delta_seconds=max_delta,
        source_files_verified_unchanged=True,
        sessions=sessions,
    )
    print(
        f"Validated {manifest['session_count']} sessions / {manifest['stream_count']} streams; "
        f"mic1 Label Studio tasks: {manifest['mic1_task_count']}"
    )
    print(f"Maximum observed paired-duration delta: {max_delta:.3f}s")
    if dry_run:
        print(f"[dry-run] Would stage corpus at {staging_root}")
        return manifest

    staging_root.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        sidecar_pairs = (
            (
                Path(session["session_sidecar_source_path"]),
                staging_root / session["session_sidecar_staged_relative_path"],
            ),
            (
                Path(session["recording_sidecar_source_path"]),
                staging_root / session["recording_sidecar_staged_relative_path"],
            ),
        )
        for source, destination in sidecar_pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"Staged sidecar failed checksum verification: {destination}")
        for stream in session["streams"]:
            source = Path(stream["source_path"])
            destination = staging_root / stream["staged_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != stream["sha256"]:
                raise RuntimeError(f"Staged copy failed checksum verification: {destination}")

    for state in all_states:
        verify_source_unchanged(state)

    manifest_path = staging_root / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [
        f"{stream['sha256']}  {stream['staged_relative_path']}"
        for session in sessions
        for stream in session["streams"]
    ]
    (staging_root / "source_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Staged corpus manifest: {manifest_path}")
    print(f"Label Studio mic1 import directory: {staging_root / 'mic1'}")
    return manifest


def main() -> None:
    """Run the MCP capture staging CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stage_captures(args.capture_root, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
