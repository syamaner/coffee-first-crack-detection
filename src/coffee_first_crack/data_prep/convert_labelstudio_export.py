"""Convert Label Studio JSON export to per-file annotation format.

Reads a Label Studio JSON export and produces one annotation JSON per audio
file, suitable for consumption by ``chunk_audio.py``.

Usage::

    python -m coffee_first_crack.data_prep.convert_labelstudio_export \\
        --input data/labels/project-1-at-YYYY-MM-DD.json \\
        --output data/labels \\
        --data-root data/raw
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import librosa

from coffee_first_crack.data_prep.corpus_manifest import (
    CaptureManifest,
    SessionRecord,
    StreamRecord,
    index_manifest_streams,
    resolve_within,
)

SAMPLE_RATE = 44100

# Label Studio prefixes uploads with an 8-char hex hash or a UUID-like string.
_LABEL_STUDIO_HASH_RE = re.compile(
    r"^(?:[0-9a-f]{8}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})-(.+)$",
    re.IGNORECASE,
)
_LABEL_STUDIO_TRUNCATION_RE = re.compile(
    r"^(?P<truncated>.+)_(?P<suffix>[A-Za-z0-9]{7})\.wav$",
    re.IGNORECASE,
)
_STAGED_MIC_RE = re.compile(r"^(?P<pair_id>[0-9a-f]{32})__mic(?P<mic_num>[12])-")


def strip_hash_prefix(filename: str) -> str:
    """Remove the hash prefix Label Studio adds to uploaded filenames.

    Only strips prefixes that match known Label Studio hash formats.
    Filenames that merely contain hyphens are returned unchanged.

    Args:
        filename: Potentially prefixed filename, e.g. ``"0d93a737-roast-1.wav"``.

    Returns:
        Original filename without a recognised Label Studio hash prefix.
    """
    match = _LABEL_STUDIO_HASH_RE.match(filename)
    if match:
        return match.group(1)
    return filename


def resolve_manifest_stream(
    uploaded_name: str,
    manifest: CaptureManifest,
) -> tuple[SessionRecord, StreamRecord]:
    """Resolve an uploaded filename to exactly one manifest stream.

    Label Studio stores uploads in a Django ``FileField``. Long UUID-prefixed
    filenames can be truncated and receive a seven-character collision suffix.
    Exact staged basenames remain the primary lookup. The fallback accepts only
    that specific truncation shape, requires the immutable pair ID and mic
    number to agree, and proves the manifest basename starts with the retained
    prefix.

    Args:
        uploaded_name: Filename after removing Label Studio's upload hash.
        manifest: Staged MCP capture manifest.

    Returns:
        The matching ``(session, stream)`` tuple.

    Raises:
        ValueError: If the upload is unknown, malformed, or inconsistent with
            the manifest.
    """
    streams_by_basename, sessions_by_pair_id = index_manifest_streams(manifest)
    exact = streams_by_basename.get(uploaded_name)
    if exact is not None:
        return exact

    truncation_match = _LABEL_STUDIO_TRUNCATION_RE.fullmatch(uploaded_name)
    if truncation_match is None:
        raise ValueError(
            f"Label Studio task is not present in the capture manifest: {uploaded_name}"
        )
    truncated = truncation_match.group("truncated")
    staged_match = _STAGED_MIC_RE.match(truncated)
    if staged_match is None:
        raise ValueError(
            f"Label Studio task is not present in the capture manifest: {uploaded_name}"
        )

    pair_id = staged_match.group("pair_id")
    mic_num = int(staged_match.group("mic_num"))
    session = sessions_by_pair_id.get(pair_id)
    if session is None:
        raise ValueError(f"Label Studio task has unknown pair_id {pair_id}: {uploaded_name}")
    candidates = [stream for stream in session["streams"] if stream["mic_num"] == mic_num]
    if len(candidates) != 1:
        raise ValueError(f"Label Studio task has ambiguous mic identity: {uploaded_name}")
    stream = candidates[0]
    staged_basename = Path(stream["staged_relative_path"]).name
    if not staged_basename.startswith(truncated):
        raise ValueError(
            f"Label Studio truncated upload does not match the manifest basename: {uploaded_name}"
        )
    return session, stream


def convert_task(
    task: dict[str, Any],
    data_root: Path,
    manifest: CaptureManifest | None = None,
) -> dict[str, Any]:
    """Convert a single Label Studio task to our annotation format.

    Args:
        task: A single task dict from the Label Studio JSON export.
        data_root: Local directory containing the WAV files.
        manifest: Optional staged MCP capture manifest. When supplied, uploaded
            basenames are mapped back to collision-free staged relative paths
            and human-label pair provenance is added.

    Returns:
        Dict with keys ``audio_file``, ``duration``, ``sample_rate``,
        and ``annotations`` (list of region dicts).
    """
    file_upload = task.get("file_upload")
    if file_upload:
        hashed_name = Path(file_upload).name
    else:
        audio_path = task.get("data", {}).get("audio", "")
        hashed_name = Path(audio_path).name

    if not hashed_name:
        raise ValueError(
            "Task is missing both 'file_upload' and 'data.audio', so no audio "
            "filename could be resolved."
        )

    original_name = strip_hash_prefix(hashed_name)
    if not original_name:
        raise ValueError(f"Resolved an empty audio filename from task source {hashed_name!r}.")
    session = None
    stream = None
    audio_file = original_name
    sample_rate = SAMPLE_RATE
    if manifest is not None:
        session, stream = resolve_manifest_stream(original_name, manifest)
        if stream["mic_num"] != 1:
            raise ValueError(f"Only mic1 may be human-labelled for MCP sessions: {original_name}")
        staged_audio_path = resolve_within(
            Path(manifest["staging_root"]), stream["staged_relative_path"]
        )
        data_root_resolved = data_root.resolve()
        if not staged_audio_path.is_relative_to(data_root_resolved):
            raise ValueError(
                f"Manifest staging root is outside --data-root: {manifest['staging_root']}"
            )
        audio_file = staged_audio_path.relative_to(data_root_resolved).as_posix()
        sample_rate = stream["sample_rate"]
        local_audio_path = staged_audio_path
    else:
        local_audio_path = data_root / original_name
    if not local_audio_path.exists() or not local_audio_path.is_file():
        raise FileNotFoundError(
            f"Resolved audio file does not exist or is not a file: {local_audio_path}"
        )

    try:
        duration = librosa.get_duration(path=str(local_audio_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read duration for audio file: {local_audio_path}") from exc

    annotations: list[dict[str, Any]] = []
    raw_annotations = task.get("annotations")
    if manifest is not None:
        if not isinstance(raw_annotations, list):
            raise ValueError(f"MCP mic1 task has no annotation list: {original_name}")
        submitted: list[dict[str, Any]] = []
        for annotation in raw_annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"MCP mic1 task has a malformed annotation: {original_name}")
            was_cancelled = annotation.get("was_cancelled", False)
            if not isinstance(was_cancelled, bool):
                raise ValueError(f"MCP mic1 task has invalid cancellation state: {original_name}")
            if not was_cancelled:
                submitted.append(annotation)
        if len(submitted) != 1 or not isinstance(submitted[0].get("result"), list):
            raise ValueError(
                "MCP mic1 task must contain exactly one submitted, non-cancelled annotation "
                f"object: {original_name}"
            )
        ann_list = submitted
    else:
        ann_list = raw_annotations or []
    for ann in ann_list:
        results = ann.get("result") or []
        for r in results:
            if r.get("type") == "labels" and "value" in r:
                val = r["value"]
                labels = val.get("labels") or []
                if not labels:
                    continue
                annotations.append(
                    {
                        "id": f"chunk_{len(annotations):03d}",
                        "start_time": float(val.get("start", 0.0)),
                        "end_time": float(val.get("end", 0.0)),
                        "label": str(labels[0]),
                        "confidence": "high",
                    }
                )

    if manifest is not None:
        if len(annotations) > 1:
            raise ValueError(
                f"MCP mic1 task must have at most one first_crack region: {original_name}"
            )
        for annotation in annotations:
            start = annotation["start_time"]
            end = annotation["end_time"]
            if annotation["label"] != "first_crack":
                raise ValueError(
                    f"MCP mic1 task has unsupported label {annotation['label']!r}: {original_name}"
                )
            if start < 0 or end <= start or end > duration:
                raise ValueError(
                    f"MCP mic1 task has invalid first_crack boundary {start}-{end}: "
                    f"{original_name} duration={duration}"
                )

    converted: dict[str, Any] = {
        "audio_file": audio_file,
        "duration": duration,
        "sample_rate": sample_rate,
        "annotations": annotations,
    }
    if session is not None and stream is not None:
        converted.update(
            {
                "pair_id": session["pair_id"],
                "mic_num": stream["mic_num"],
                "origin": session["origin"],
                "roast_num": session["roast_num"],
                "original_filename": stream["original_filename"],
                "provenance": {
                    "annotation_source": "human_label_studio",
                    "pair_id": session["pair_id"],
                    "staged_from": stream["source_path"],
                },
            }
        )
    return converted


def main() -> None:
    """CLI entry point for converting Label Studio exports."""
    parser = argparse.ArgumentParser(
        description="Convert Label Studio JSON export to per-file annotation JSONs"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to Label Studio JSON export file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write per-file annotation JSONs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Local directory where WAV files live (default: data/raw)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional staged MCP capture_manifest.json",
    )
    args = parser.parse_args()

    out_dir: Path = args.output

    with args.input.open("r") as f:
        exported = json.load(f)

    manifest = None
    if args.manifest is not None:
        from coffee_first_crack.data_prep.corpus_manifest import load_capture_manifest

        manifest = load_capture_manifest(args.manifest)

    if not isinstance(exported, list):
        raise ValueError("Label Studio export must contain a JSON list of tasks")

    converted_items: list[tuple[Path, dict[str, Any]]] = []
    seen_outputs: set[Path] = set()
    for task in exported:
        if not isinstance(task, dict):
            raise ValueError("Every Label Studio task must be a JSON object")
        converted = convert_task(task, args.data_root, manifest)
        stem = Path(converted["audio_file"]).stem
        out_path = out_dir / f"{stem}.json"
        if out_path in seen_outputs or out_path.exists():
            raise FileExistsError(f"Refusing to overwrite annotation output: {out_path}")
        seen_outputs.add(out_path)
        converted_items.append((out_path, converted))

    if manifest is not None:
        expected_pair_ids = {session["pair_id"] for session in manifest["sessions"]}
        converted_pair_ids = {converted.get("pair_id") for _, converted in converted_items}
        if converted_pair_ids != expected_pair_ids or len(converted_items) != len(
            expected_pair_ids
        ):
            missing = sorted(expected_pair_ids - converted_pair_ids)
            unexpected = sorted(converted_pair_ids - expected_pair_ids, key=str)
            raise ValueError(
                "Label Studio export must contain exactly one mic1 task per pair; "
                f"missing={missing}, unexpected={unexpected}, tasks={len(converted_items)}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    for out_path, converted in converted_items:
        with out_path.open("w") as f:
            json.dump(converted, f, indent=2)
        print(f"Wrote {out_path}")

    print(f"Converted {len(converted_items)} tasks -> {out_dir}")


if __name__ == "__main__":
    main()
