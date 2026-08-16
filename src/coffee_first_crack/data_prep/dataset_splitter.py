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
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
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
) -> tuple[dict[str, dict[str, list[Path]]], dict[str, set[str]]]:
    """Group included chunks by physical pair identity.

    Args:
        input_dir: Root containing label directories and chunk WAVs.
        chunk_manifest_path: JSONL manifest emitted by ``chunk_audio``.

    Returns:
        ``(groups, recordings_by_pair)`` where groups have the same label-file
        shape used by :func:`recording_level_split`, keyed by pair ID.

    Raises:
        FileNotFoundError: If the chunk manifest or a declared chunk is absent.
        ValueError: If identities are missing, duplicated, or disagree with the
            filesystem.
    """
    if not chunk_manifest_path.is_file():
        raise FileNotFoundError(
            f"Pair-aware chunk manifest is required for leakage-free splitting: "
            f"{chunk_manifest_path}"
        )
    groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    recordings_by_pair: dict[str, set[str]] = defaultdict(set)
    declared_filenames: set[str] = set()
    with chunk_manifest_path.open("r", encoding="utf-8") as handle:
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
            recordings_by_pair[pair_id].add(recording_id)
            if not included:
                continue
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe chunk filename at line {line_number}: {filename!r}")
            if filename in declared_filenames:
                raise ValueError(f"Duplicate included chunk in manifest: {filename}")
            declared_filenames.add(filename)
            label_dir = input_dir / label
            if not label_dir.is_dir() or label_dir.is_symlink():
                raise ValueError(f"Chunk label directory must be a regular directory: {label_dir}")
            chunk_path = label_dir / filename
            if not chunk_path.is_file() or chunk_path.is_symlink():
                raise FileNotFoundError(
                    f"Declared chunk must be a regular non-symlink file: {chunk_path}"
                )
            groups[pair_id][label].append(chunk_path)

    actual_filenames = {
        path.name
        for label_dir in input_dir.iterdir()
        if label_dir.is_dir() and not label_dir.name.startswith(".")
        for path in label_dir.glob("*.wav")
    }
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
    report_path.write_text("\n".join(lines) + "\n")
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
    integrity_path.write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    print(f"🔒 Pair integrity report saved to: {integrity_path}")


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
    args = parser.parse_args()

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
