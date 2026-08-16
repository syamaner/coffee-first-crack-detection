"""Manifest helpers for session-aware audio corpus preparation."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePath
from typing import Any, TypedDict, cast

_PAIR_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StreamRecord(TypedDict):
    """One staged microphone stream in a capture manifest."""

    mic_num: int
    label: str
    original_filename: str
    source_path: str
    staged_relative_path: str
    size_bytes: int
    sha256: str
    duration_seconds: float
    sample_rate: int


class SessionRecord(TypedDict):
    """One physical roast session in a capture manifest."""

    pair_id: str
    origin: str
    roast_num: int
    source_session_dir: str
    session_sidecar_source_path: str
    session_sidecar_staged_relative_path: str
    recording_sidecar_source_path: str
    recording_sidecar_staged_relative_path: str
    observed_duration_delta_seconds: float
    streams: list[StreamRecord]


class CaptureManifest(TypedDict):
    """Top-level schema for a staged MCP capture corpus."""

    schema_version: int
    source_root: str
    staging_root: str
    session_count: int
    stream_count: int
    mic1_task_count: int
    max_observed_duration_delta_seconds: float
    source_files_verified_unchanged: bool
    sessions: list[SessionRecord]


def safe_relative_path(value: str) -> Path:
    """Validate and return a portable, non-traversing relative path.

    Args:
        value: Path text from a manifest or annotation.

    Returns:
        A validated relative :class:`~pathlib.Path`.

    Raises:
        ValueError: If the value is empty, absolute, or contains traversal.
    """
    if not value or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    pure = PurePath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*pure.parts)


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve a manifest path while proving it remains under ``root``.

    Args:
        root: Trusted directory root.
        relative: Untrusted relative path text.

    Returns:
        Resolved path under ``root``.

    Raises:
        ValueError: If the path escapes ``root``.
    """
    root_resolved = root.resolve()
    candidate = (root_resolved / safe_relative_path(relative)).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"Path escapes root {root}: {relative!r}")
    return candidate


def load_capture_manifest(path: Path) -> CaptureManifest:
    """Load a validated-enough capture manifest for downstream lookup.

    Structural and source-file validation belongs to the ingester. This loader
    verifies the stable schema fields and uniqueness relied on downstream.

    Args:
        path: Manifest JSON path.

    Returns:
        Parsed capture manifest.

    Raises:
        ValueError: If the schema or identities are invalid.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported capture manifest schema: {path}")
    if not isinstance(raw.get("source_root"), str) or not isinstance(raw.get("staging_root"), str):
        raise ValueError(f"Manifest roots must be strings: {path}")
    if raw.get("source_files_verified_unchanged") is not True:
        raise ValueError(f"Manifest lacks source-integrity proof: {path}")
    max_delta = raw.get("max_observed_duration_delta_seconds")
    if not isinstance(max_delta, (int, float)) or isinstance(max_delta, bool) or max_delta < 0:
        raise ValueError(f"Manifest has invalid alignment uncertainty: {path}")
    sessions = raw.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError(f"Manifest sessions must be a list: {path}")

    pair_ids: set[str] = set()
    staged_paths: set[str] = set()
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError(f"Manifest session must be an object: {path}")
        pair_id = session.get("pair_id")
        if not isinstance(pair_id, str) or _PAIR_ID_RE.fullmatch(pair_id) is None:
            raise ValueError(f"Manifest session has invalid pair_id: {session!r}")
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate pair_id in manifest: {pair_id}")
        pair_ids.add(pair_id)
        streams = session.get("streams")
        origin = session.get("origin")
        roast_num = session.get("roast_num")
        observed_delta = session.get("observed_duration_delta_seconds")
        if not isinstance(origin, str) or not origin:
            raise ValueError(f"Manifest session {pair_id} has invalid origin")
        if not isinstance(roast_num, int) or isinstance(roast_num, bool) or roast_num < 1:
            raise ValueError(f"Manifest session {pair_id} has invalid roast_num")
        if (
            not isinstance(observed_delta, (int, float))
            or isinstance(observed_delta, bool)
            or observed_delta < 0
            or observed_delta > max_delta
        ):
            raise ValueError(f"Manifest session {pair_id} has invalid duration delta")
        for sidecar_field in (
            "session_sidecar_staged_relative_path",
            "recording_sidecar_staged_relative_path",
        ):
            relative_sidecar = session.get(sidecar_field)
            if not isinstance(relative_sidecar, str):
                raise ValueError(f"Manifest session {pair_id} lacks {sidecar_field}")
            safe_relative_path(relative_sidecar)
        if not isinstance(streams, list) or len(streams) != 2:
            raise ValueError(f"Manifest session {pair_id} must have exactly two streams")
        mic_numbers: set[int] = set()
        for stream in streams:
            if not isinstance(stream, dict):
                raise ValueError(f"Manifest stream must be an object: {stream!r}")
            relative = stream.get("staged_relative_path")
            if not isinstance(relative, str):
                raise ValueError(f"Manifest stream has invalid staged path: {stream!r}")
            safe_relative_path(relative)
            if relative in staged_paths:
                raise ValueError(f"Duplicate staged destination in manifest: {relative}")
            staged_paths.add(relative)
            mic_num = stream.get("mic_num")
            duration = stream.get("duration_seconds")
            sample_rate = stream.get("sample_rate")
            original_filename = stream.get("original_filename")
            digest = stream.get("sha256")
            if (
                not isinstance(mic_num, int)
                or isinstance(mic_num, bool)
                or mic_num not in {1, 2}
                or mic_num in mic_numbers
            ):
                raise ValueError(f"Manifest session {pair_id} has invalid mic identity")
            mic_numbers.add(mic_num)
            if not relative.startswith(f"mic{mic_num}/"):
                raise ValueError(f"Manifest stream path disagrees with mic number: {relative}")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or duration <= 0
                or not isinstance(sample_rate, int)
                or isinstance(sample_rate, bool)
                or sample_rate <= 0
            ):
                raise ValueError(f"Manifest stream has invalid audio metadata: {stream!r}")
            if (
                not isinstance(original_filename, str)
                or Path(original_filename).name != original_filename
                or not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None
            ):
                raise ValueError(f"Manifest stream has invalid file provenance: {stream!r}")
        if mic_numbers != {1, 2}:
            raise ValueError(f"Manifest session {pair_id} must contain mic1 and mic2")

    if raw.get("session_count") != len(sessions):
        raise ValueError(f"Manifest session_count does not match sessions: {path}")
    if raw.get("stream_count") != sum(len(session["streams"]) for session in sessions):
        raise ValueError(f"Manifest stream_count does not match sessions: {path}")
    if raw.get("mic1_task_count") != len(sessions):
        raise ValueError(f"Manifest mic1_task_count does not match sessions: {path}")

    return cast(CaptureManifest, raw)


def index_manifest_streams(
    manifest: CaptureManifest,
) -> tuple[dict[str, tuple[SessionRecord, StreamRecord]], dict[str, SessionRecord]]:
    """Index streams by staged basename and sessions by pair identity.

    Args:
        manifest: Loaded capture manifest.

    Returns:
        ``(streams_by_basename, sessions_by_pair_id)``.

    Raises:
        ValueError: If a staged basename is ambiguous.
    """
    streams: dict[str, tuple[SessionRecord, StreamRecord]] = {}
    sessions: dict[str, SessionRecord] = {}
    for session in manifest["sessions"]:
        sessions[session["pair_id"]] = session
        for stream in session["streams"]:
            basename = Path(stream["staged_relative_path"]).name
            if basename in streams:
                raise ValueError(f"Ambiguous staged basename in manifest: {basename}")
            streams[basename] = (session, stream)
    return streams, sessions
