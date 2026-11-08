"""Split chunked audio into train/validation/test sets with stratification.

Splits at the **physical pair/session level** (not chunk or stream level) to
prevent data leakage. All chunks from every microphone in one roast stay in the
same split.

Usage::

    python -m coffee_first_crack.data_prep.dataset_splitter \\
        --input data/processed \\
        --output data/splits \\
        --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import stat
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO, cast

from sklearn.model_selection import train_test_split

from coffee_first_crack.data_prep.output_paths import require_absent_output_dir

logger = logging.getLogger(__name__)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
_MAX_PRIOR_BYTES = 16 * 1024 * 1024
_CHUNK_LABELS = frozenset({"first_crack", "no_first_crack"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def extract_recording_stem(chunk_filename: str) -> str:
    """Extract the source recording stem from a chunk filename.

    Chunk filenames follow the pattern ``{recording_stem}_w{start}.wav``.

    Args:
        chunk_filename: Chunk WAV filename, e.g.
            ``"roast-1-costarica-hermosa-hp-a_w0530.0.wav"``.

    Returns:
        Recording stem, e.g. ``"roast-1-costarica-hermosa-hp-a"``.
    """
    stem = Path(chunk_filename).stem
    # Remove the trailing _wNNNN.N suffix added by chunk_audio.py
    match = re.match(r"^(.+)_w\d+\.\d+$", stem)
    if match:
        return match.group(1)
    return stem


def group_chunks_by_recording(
    input_dir: Path,
) -> dict[str, dict[str, list[Path]]]:
    """Group chunk files by their source recording and label.

    Args:
        input_dir: Root directory containing ``first_crack/`` and
            ``no_first_crack/`` subdirectories.

    Returns:
        Nested dict: ``{recording_stem: {label: [paths]}}``.
    """
    groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue
        label = label_dir.name
        for wav_file in sorted(label_dir.glob("*.wav")):
            rec_stem = extract_recording_stem(wav_file.name)
            groups[rec_stem][label].append(wav_file)

    return dict(groups)


def group_chunks_by_pair(
    input_dir: Path,
    chunk_manifest_path: Path,
    *,
    _bound_input: _BoundChunkInput | None = None,
) -> tuple[dict[str, dict[str, list[Path]]], dict[str, set[str]]]:
    """Group included chunks by physical pair identity.

    Args:
        input_dir: Root containing label directories and chunk WAVs.
        chunk_manifest_path: JSONL manifest emitted by ``chunk_audio``.
        _bound_input: Descriptor-backed IO for preservation; identity rules are shared.

    Returns:
        ``(groups, recordings_by_pair)`` where groups have the same label-file
        shape used by :func:`recording_level_split`, keyed by pair ID.

    Raises:
        FileNotFoundError: If the chunk manifest or a declared chunk is absent.
        ValueError: If identities are missing, duplicated, or disagree with the
            filesystem.
    """
    if _bound_input is None and not chunk_manifest_path.is_file():
        raise FileNotFoundError(
            f"Pair-aware chunk manifest is required for leakage-free splitting: "
            f"{chunk_manifest_path}"
        )
    groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    recordings_by_pair: dict[str, set[str]] = defaultdict(set)
    pair_by_source_hash: dict[str, str] = {}
    declared_filenames: set[str] = set()
    reader = (
        chunk_manifest_path.open("r", encoding="utf-8")
        if _bound_input is None
        else _bound_input._open_manifest()
    )
    with reader as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed chunk manifest JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Chunk manifest line {line_number} is not an object")
            filename = record.get("chunk_filename")
            pair_id = record.get("pair_id")
            recording_id = record.get("recording_id")
            source_audio_sha256 = record.get("source_audio_sha256")
            label = record.get("label")
            included = record.get("included")
            if (
                not isinstance(filename, str)
                or not filename
                or not isinstance(pair_id, str)
                or not pair_id
                or not isinstance(recording_id, str)
                or not recording_id
                or not isinstance(source_audio_sha256, str)
                or _SHA256_RE.fullmatch(source_audio_sha256) is None
                or not isinstance(label, str)
                or not label
                or not isinstance(included, bool)
            ):
                raise ValueError(f"Chunk manifest line {line_number} lacks required identities")
            if label not in _CHUNK_LABELS:
                raise ValueError(f"Unsupported chunk label at line {line_number}: {label!r}")
            previous_pair = pair_by_source_hash.setdefault(source_audio_sha256, pair_id)
            if previous_pair != pair_id:
                raise ValueError(
                    "Source checksum is assigned to multiple pair IDs: "
                    f"{source_audio_sha256} -> {previous_pair!r}, {pair_id!r}"
                )
            recordings_by_pair[pair_id].add(recording_id)
            if not included:
                continue
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe chunk filename at line {line_number}: {filename!r}")
            if filename in declared_filenames:
                raise ValueError(f"Duplicate included chunk in manifest: {filename}")
            declared_filenames.add(filename)
            label_dir = input_dir / label
            chunk_path = label_dir / filename
            if _bound_input is not None:
                _bound_input._validate_chunk(label, filename)
            else:
                if not label_dir.is_dir() or label_dir.is_symlink():
                    raise ValueError(
                        f"Chunk label directory must be a regular directory: {label_dir}"
                    )
                if not chunk_path.is_file() or chunk_path.is_symlink():
                    raise FileNotFoundError(
                        f"Declared chunk must be a regular non-symlink file: {chunk_path}"
                    )
            groups[pair_id][label].append(chunk_path)

    actual_filenames = (
        _bound_input._filenames()
        if _bound_input is not None
        else {
            path.name
            for label_dir in input_dir.iterdir()
            if label_dir.is_dir() and not label_dir.name.startswith(".")
            for path in label_dir.glob("*.wav")
        }
    )
    if declared_filenames != actual_filenames:
        missing = sorted(actual_filenames - declared_filenames)
        extra = sorted(declared_filenames - actual_filenames)
        raise ValueError(
            f"Chunk manifest/filesystem mismatch; undeclared_files={missing}, missing_files={extra}"
        )
    return dict(groups), dict(recordings_by_pair)


def recording_level_split(
    groups: dict[str, dict[str, list[Path]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Assign recordings to train/val/test splits.

    Uses stratified splitting based on whether each recording contains any
    ``first_crack`` chunks.

    Args:
        groups: Output of :func:`group_chunks_by_recording`.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        Tuple of ``(train_recordings, val_recordings, test_recordings)``
        as lists of recording stems.
    """
    recordings = sorted(groups.keys())
    # Stratify by whether the recording has any first_crack chunks
    has_fc = [
        1 if "first_crack" in groups[r] and groups[r]["first_crack"] else 0 for r in recordings
    ]

    def _safe_split(
        data: list[str],
        test_size: float,
        random_state: int,
        stratify_labels: list[int],
    ) -> tuple[list[str], list[str]]:
        """train_test_split with fallback to unstratified if too few samples."""
        try:
            return cast(
                tuple[list[str], list[str]],
                train_test_split(
                    data, test_size=test_size, random_state=random_state, stratify=stratify_labels
                ),
            )
        except ValueError:
            logger.warning("Too few recordings for stratified split — falling back to random.")
            try:
                return cast(
                    tuple[list[str], list[str]],
                    train_test_split(data, test_size=test_size, random_state=random_state),
                )
            except ValueError as exc:
                raise ValueError(
                    "Unable to split recordings with the requested test_size="
                    f"{test_size}. Got {len(data)} recording(s), which is insufficient "
                    "for either stratified or unstratified splitting."
                ) from exc

    # First split: separate test set
    train_val_recs, test_recs = _safe_split(recordings, test_ratio, seed, has_fc)

    # Second split: train vs val
    train_val_fc = [
        1 if "first_crack" in groups[r] and groups[r]["first_crack"] else 0 for r in train_val_recs
    ]
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_recs, val_recs = _safe_split(train_val_recs, val_ratio_adjusted, seed, train_val_fc)

    return train_recs, val_recs, test_recs


def copy_chunks(
    groups: dict[str, dict[str, list[Path]]],
    recording_stems: list[str],
    output_dir: Path,
    split_name: str,
) -> dict[str, int]:
    """Copy all chunks for the given recordings to the split directory.

    Args:
        groups: Output of :func:`group_chunks_by_recording`.
        recording_stems: Recordings assigned to this split.
        output_dir: Root output directory.
        split_name: Split name (``"train"``, ``"val"``, ``"test"``).

    Returns:
        Dict with counts per label.
    """
    counts: dict[str, int] = defaultdict(int)
    split_dir = output_dir / split_name

    # Clean stale data from previous runs
    if split_dir.exists():
        shutil.rmtree(split_dir)

    for rec_stem in recording_stems:
        for label, files in groups[rec_stem].items():
            label_dir = split_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, label_dir / f.name)
                counts[label] += 1

    total = sum(counts.values())
    print(f"  ✅ {split_name}: {total} chunks from {len(recording_stems)} recordings")
    for label in sorted(counts):
        print(f"      - {label}: {counts[label]}")
    return dict(counts)


def generate_split_report(
    output_dir: Path,
    groups: dict[str, dict[str, list[Path]]],
    train_recs: list[str],
    val_recs: list[str],
    test_recs: list[str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    test_counts: dict[str, int],
    recordings_by_pair: dict[str, set[str]] | None = None,
    *,
    exclusive: bool = False,
    _write_report: Callable[[str, str], None] | None = None,
) -> None:
    """Generate a markdown split report.

    Args:
        output_dir: Root output directory.
        groups: Recording groups.
        train_recs: Training recording stems.
        val_recs: Validation recording stems.
        test_recs: Test recording stems.
        train_counts: Training label counts.
        val_counts: Validation label counts.
        test_counts: Test label counts.
        recordings_by_pair: Physical pair-to-stream identities.
        exclusive: Refuse to overwrite reports in preservation mode.
        _write_report: Bound output writer used only by preservation orchestration.
    """
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())
    total_all = total_train + total_val + total_test

    recordings_by_pair = recordings_by_pair or {pair: {pair} for pair in groups}
    total_recordings = sum(len(recordings) for recordings in recordings_by_pair.values())
    lines = [
        "# Dataset Split Report",
        "",
        "## Split Configuration",
        "",
        "- **Splitting strategy**: physical pair/session level",
        f"- **Total physical sessions / pair IDs**: {len(groups)}",
        f"- **Total streams / recordings**: {total_recordings}",
        f"- **Total chunks**: {total_all}",
        "",
        "## Recording Assignments",
        "",
    ]

    for split_name, recs in [("Train", train_recs), ("Validation", val_recs), ("Test", test_recs)]:
        lines.append(f"### {split_name}")
        lines.append("")
        for r in sorted(recs):
            fc = len(groups[r].get("first_crack", []))
            nfc = len(groups[r].get("no_first_crack", []))
            streams = ", ".join(sorted(recordings_by_pair[r]))
            lines.append(
                f"- {r}: {len(recordings_by_pair[r])} stream(s) [{streams}]; "
                f"{fc} first_crack, {nfc} no_first_crack"
            )
        lines.append("")

    lines.extend(
        [
            "## Chunk Distribution",
            "",
            "| Split | first_crack | no_first_crack | Total | % of dataset |",
            "|-------|-------------|----------------|-------|-------------|",
        ]
    )
    for name, counts, total in [
        ("Train", train_counts, total_train),
        ("Val", val_counts, total_val),
        ("Test", test_counts, total_test),
    ]:
        fc = counts.get("first_crack", 0)
        nfc = counts.get("no_first_crack", 0)
        pct = total / total_all * 100 if total_all > 0 else 0
        lines.append(f"| {name} | {fc} | {nfc} | {total} | {pct:.1f}% |")

    report_path = output_dir / "split_report.md"
    if _write_report is None:
        with report_path.open("x" if exclusive else "w") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        _write_report(report_path.name, "\n".join(lines) + "\n")
    print(f"\n📊 Split report saved to: {report_path}")

    train_pairs = set(train_recs)
    val_pairs = set(val_recs)
    test_pairs = set(test_recs)
    overlaps = {
        "train_validation": sorted(train_pairs & val_pairs),
        "train_test": sorted(train_pairs & test_pairs),
        "validation_test": sorted(val_pairs & test_pairs),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Pair leakage detected after splitting: {overlaps}")
    integrity = {
        "schema_version": 1,
        "strategy": "pair_id",
        "physical_session_count": len(groups),
        "stream_recording_count": total_recordings,
        "splits": {
            "train": {
                "pair_ids": sorted(train_pairs),
                "physical_session_count": len(train_pairs),
                "stream_recording_count": sum(
                    len(recordings_by_pair[pair]) for pair in train_pairs
                ),
            },
            "validation": {
                "pair_ids": sorted(val_pairs),
                "physical_session_count": len(val_pairs),
                "stream_recording_count": sum(len(recordings_by_pair[pair]) for pair in val_pairs),
            },
            "test": {
                "pair_ids": sorted(test_pairs),
                "physical_session_count": len(test_pairs),
                "stream_recording_count": sum(len(recordings_by_pair[pair]) for pair in test_pairs),
            },
        },
        "pair_id_intersections": overlaps,
        "integrity_passed": True,
    }
    integrity_path = output_dir / "split_integrity.json"
    if _write_report is None:
        with integrity_path.open("x" if exclusive else "w") as handle:
            handle.write(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    else:
        _write_report(integrity_path.name, json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    print(f"🔒 Pair integrity report saved to: {integrity_path}")


def validate_prior_assignments(document: object) -> dict[str, list[str]]:
    """Validate decoded prior pair membership without trusting historical counts.

    Args:
        document: Decoded prior integrity JSON.

    Returns:
        Disjoint pair IDs indexed by canonical split name.

    Raises:
        ValueError: If schema or membership is ambiguous or invalid.
    """
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("strategy") != "pair_id"
        or document.get("integrity_passed") is not True
    ):
        raise ValueError("Prior integrity must be schema_version 1, pair_id, integrity_passed true")
    splits = document.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Prior splits must contain exactly train/validation/test")
    assignments: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name in ("train", "validation", "test"):
        split = splits[name]
        pairs = split.get("pair_ids") if isinstance(split, dict) else None
        if not isinstance(pairs, list) or any(
            not isinstance(pair, str) or not pair for pair in pairs
        ):
            raise ValueError("Prior pair_ids must be lists of nonempty strings")
        if len(set(pairs)) != len(pairs) or seen.intersection(pairs):
            raise ValueError("Prior pair_ids must be unique and disjoint")
        seen.update(pairs)
        assignments[name] = sorted(pairs)
    return assignments


def preserve_pair_assignments(
    prior: dict[str, list[str]], current_pair_ids: set[str]
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Project validated membership onto current usable pair identities.

    Args:
        prior: Validated canonical split memberships.
        current_pair_ids: Identities with usable chunks.

    Returns:
        Current assignments, unassigned IDs, and missing prior IDs, all sorted.
    """
    known = {pair for pairs in prior.values() for pair in pairs}
    return (
        {name: sorted(set(pairs) & current_pair_ids) for name, pairs in prior.items()},
        sorted(current_pair_ids - known),
        sorted(known - current_pair_ids),
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate prior JSON object key: {key!r}")
        result[key] = value
    return result


def _load_prior(path: Path) -> tuple[dict[str, list[str]], str]:
    # O_NONBLOCK prevents a raced FIFO from hanging before fstat can reject it.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Prior integrity must be a regular non-symlink file")
        if metadata.st_size > _MAX_PRIOR_BYTES:
            raise ValueError("Prior integrity exceeds 16 MiB maximum")
        payload = handle.read(_MAX_PRIOR_BYTES + 1)
        if len(payload) > _MAX_PRIOR_BYTES:
            raise ValueError("Prior integrity exceeds 16 MiB maximum")
    return validate_prior_assignments(
        json.loads(payload, object_pairs_hook=_unique_json_object)
    ), hashlib.sha256(payload).hexdigest()


def _open_source_file(root: int, parts: tuple[str, ...], handles: ExitStack) -> int:
    if not parts:
        raise ValueError("Preserved input must be a regular non-symlink file")
    if any(component in {"", ".", ".."} for component in parts):
        raise ValueError("Preserved source path components must not be empty, dot or dotdot")
    directory = root
    for component in parts[:-1]:
        directory = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory)
        handles.callback(os.close, directory)
    descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    handles.callback(os.close, descriptor)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise ValueError("Preserved input must be a regular non-symlink file")
    return descriptor


class _BoundChunkInput:
    """Filesystem adapter sharing one held root with grouping and copying."""

    def __init__(self, root: int, manifest: int) -> None:
        self._root = root
        self._manifest = manifest
        self._chunk_identities: dict[tuple[str, str], tuple[int, int]] = {}

    def _open_manifest(self) -> TextIO:
        return os.fdopen(os.dup(self._manifest), "r", encoding="utf-8")

    def _validate_chunk(self, label: str, filename: str) -> None:
        with ExitStack() as handles:
            descriptor = _open_source_file(self._root, (label, filename), handles)
            self._chunk_identities[label, filename] = _identity(descriptor)

    def _expected_identity(self, label: str, filename: str) -> tuple[int, int]:
        return self._chunk_identities[label, filename]

    def _filenames(self) -> set[str]:
        filenames: set[str] = set()
        with os.scandir(self._root) as labels:
            for label in labels:
                if label.name.startswith("."):
                    continue
                if label.is_symlink():
                    raise ValueError("Input label entries must not be symlinks")
                if not label.is_dir(follow_symlinks=False):
                    continue
                with ExitStack() as handles:
                    directory = os.open(label.name, _DIRECTORY_FLAGS, dir_fd=self._root)
                    handles.callback(os.close, directory)
                    filenames.update(
                        name for name in os.listdir(directory) if name.endswith(".wav")
                    )
        return filenames


def _identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    return metadata.st_dev, metadata.st_ino


def _require_outside_input(directory: int, input_root: int) -> None:
    # Walk actual directory identities, including case/firmlink aliases.
    current = os.dup(directory)
    try:
        while True:
            if _identity(current) == _identity(input_root):
                raise ValueError("Output must not be inside the input chunk tree")
            ancestor = os.open("..", _DIRECTORY_FLAGS, dir_fd=current)
            if _identity(ancestor) == _identity(current):
                os.close(ancestor)
                return
            os.close(current)
            current = ancestor
    finally:
        os.close(current)


def _exclusive_output_file(directory: int, filename: str, handles: ExitStack) -> int:
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o666,
        dir_fd=directory,
    )
    handles.callback(os.close, descriptor)
    return descriptor


def _copy_preserved_chunk(
    root_descriptor: int,
    label: str,
    filename: str,
    destination_descriptor: int,
    expected_identity: tuple[int, int],
) -> None:
    with ExitStack() as handles:
        source = _open_source_file(root_descriptor, (label, filename), handles)
        if _identity(source) != expected_identity:
            raise ValueError("Validated chunk filesystem identity changed")
        destination = _exclusive_output_file(destination_descriptor, filename, handles)
        reader = handles.enter_context(os.fdopen(source, "rb", closefd=False))
        writer = handles.enter_context(os.fdopen(destination, "wb", closefd=False))
        shutil.copyfileobj(reader, writer)


def _preserve_to_output(
    input_dir: Path, output_dir: Path, manifest: Path, prior_path: Path
) -> None:
    require_absent_output_dir(output_dir)
    prior, prior_hash = _load_prior(prior_path)
    # Resolve only manifest placement before binding the input. No later path
    # grouping may observe a different root than the descriptor used to copy.
    try:
        relative_manifest = manifest.absolute().relative_to(input_dir.absolute()).parts
    except ValueError:
        relative_manifest = (manifest.name,) if manifest.parent.samefile(input_dir) else None
    with ExitStack() as handles:
        root = os.open(input_dir, _DIRECTORY_FLAGS)
        handles.callback(os.close, root)
        parent = os.open(output_dir.parent, _DIRECTORY_FLAGS)
        handles.callback(os.close, parent)
        _require_outside_input(parent, root)
        if relative_manifest is None:
            manifest_descriptor = os.open(manifest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            handles.callback(os.close, manifest_descriptor)
            if not stat.S_ISREG(os.fstat(manifest_descriptor).st_mode):
                raise ValueError("Chunk manifest must be a regular file")
        else:
            manifest_descriptor = _open_source_file(root, relative_manifest, handles)
        bound_input = _BoundChunkInput(root, manifest_descriptor)
        groups, streams = group_chunks_by_pair(input_dir, manifest, _bound_input=bound_input)
        assignments, unassigned, missing = preserve_pair_assignments(prior, set(groups))
        assigned = {pair for pairs in assignments.values() for pair in pairs}
        assigned_groups = {pair: groups[pair] for pair in sorted(assigned)}
        assigned_streams = {pair: streams[pair] for pair in sorted(assigned)}
        bindings: list[tuple[int, str, int]] = []

        def verify_output() -> None:
            metadata = output_dir.parent.stat(follow_symlinks=False)
            if (metadata.st_dev, metadata.st_ino) != _identity(parent):
                raise ValueError("Output parent identity changed")
            _require_outside_input(parent, root)
            for container, name, descriptor in bindings:
                entry = os.stat(name, dir_fd=container, follow_symlinks=False)
                if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != _identity(
                    descriptor
                ):
                    raise ValueError("Output directory identity changed")
                _require_outside_input(descriptor, root)

        def create_directory(container: int, name: str) -> int:
            verify_output()
            os.mkdir(name, dir_fd=container)
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=container)
            handles.callback(os.close, descriptor)
            bindings.append((container, name, descriptor))
            verify_output()
            return descriptor

        output = create_directory(parent, output_dir.name)
        counts: dict[str, dict[str, int]] = {}
        for name, pairs in assignments.items():
            split = create_directory(output, "val" if name == "validation" else name)
            counts[name] = {}
            for label in sorted(_CHUNK_LABELS):
                destination = create_directory(split, label)
                count = 0
                for pair in pairs:
                    for source in sorted(groups[pair].get(label, [])):
                        verify_output()
                        _copy_preserved_chunk(
                            root,
                            label,
                            source.name,
                            destination,
                            bound_input._expected_identity(label, source.name),
                        )
                        count += 1
                counts[name][label] = count

        def write_report(filename: str, content: str) -> None:
            verify_output()
            with ExitStack() as files:
                descriptor = _exclusive_output_file(output, filename, files)
                verify_output()
                writer = files.enter_context(
                    os.fdopen(descriptor, "w", encoding="utf-8", closefd=False)
                )
                writer.write(content)
            verify_output()

        report = {
            "schema_version": 1,
            "assignment_mode": "preserved",
            "prior_split_integrity_sha256": prior_hash,
            "seed_and_ratios_used": False,
            "fresh_holdout": False,
            "unassigned_pair_ids": unassigned,
            "unassigned_source_stream_ids": {pair: sorted(streams[pair]) for pair in unassigned},
            "missing_prior_pair_ids": missing,
            "splits": {
                name: {
                    "pair_ids": pairs,
                    "physical_session_count": len(pairs),
                    "stream_recording_count": sum(len(streams[pair]) for pair in pairs),
                    "chunk_count": sum(counts[name].values()),
                    "label_counts": counts[name],
                }
                for name, pairs in assignments.items()
            },
        }
        # Integrity is the final marker: sidecar/report failures cannot leave it true.
        write_report(
            "split_assignment_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        generate_split_report(
            output_dir,
            assigned_groups,
            assignments["train"],
            assignments["validation"],
            assignments["test"],
            counts["train"],
            counts["validation"],
            counts["test"],
            assigned_streams,
            exclusive=True,
            _write_report=write_report,
        )


def main() -> None:
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Split chunked dataset into train/val/test (physical pair level)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed"),
        help="Input directory with labelled audio chunks (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for splits (default: data/splits)",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default: 0.7)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=None,
        help="chunk_manifest.jsonl (defaults to <input>/chunk_manifest.jsonl)",
    )
    parser.add_argument(
        "--preserve-assignments",
        type=Path,
        default=None,
        help="Prior split_integrity.json; preserve membership and exclude new pairs",
    )
    args = parser.parse_args()
    if args.preserve_assignments is not None:
        try:
            _preserve_to_output(
                args.input,
                args.output,
                args.chunk_manifest or args.input / "chunk_manifest.jsonl",
                args.preserve_assignments,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Preservation failed: {exc}\n")
        print("Preserved assignments; seed and ratios were not used; fresh_holdout=false")
        return

    total_ratio = args.train + args.val + args.test
    if not (0.99 <= total_ratio <= 1.01):
        print(f"❌ Split ratios must sum to 1.0 (got {total_ratio})")
        return

    print("📊 Dataset Splitter (physical pair/session level)")
    print("=" * 50)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Ratios:  train={args.train}, val={args.val}, test={args.test}")
    print(f"Seed:    {args.seed}")

    chunk_manifest = args.chunk_manifest or args.input / "chunk_manifest.jsonl"
    groups, recordings_by_pair = group_chunks_by_pair(args.input, chunk_manifest)
    if not groups:
        print("❌ No chunk files found in input directory")
        return

    total_chunks = sum(len(files) for rec in groups.values() for files in rec.values())
    print(
        f"\nFound {len(groups)} physical sessions / "
        f"{sum(len(value) for value in recordings_by_pair.values())} streams, "
        f"{total_chunks} total chunks"
    )
    for pair_id in sorted(groups):
        fc = len(groups[pair_id].get("first_crack", []))
        nfc = len(groups[pair_id].get("no_first_crack", []))
        print(f"  {pair_id}: {len(recordings_by_pair[pair_id])} streams, {fc} FC, {nfc} NFC")

    print("\n🔀 Performing physical-pair-level stratified split...")
    train_recs, val_recs, test_recs = recording_level_split(
        groups, args.train, args.val, args.test, args.seed
    )

    print("\n📁 Copying chunks to split directories...")
    train_counts = copy_chunks(groups, train_recs, args.output, "train")
    val_counts = copy_chunks(groups, val_recs, args.output, "val")
    test_counts = copy_chunks(groups, test_recs, args.output, "test")

    generate_split_report(
        args.output,
        groups,
        train_recs,
        val_recs,
        test_recs,
        train_counts,
        val_counts,
        test_counts,
        recordings_by_pair,
    )

    print("\n✅ Dataset split complete!")
    print(f"   Train: {args.output}/train/")
    print(f"   Val:   {args.output}/val/")
    print(f"   Test:  {args.output}/test/")


if __name__ == "__main__":
    main()
