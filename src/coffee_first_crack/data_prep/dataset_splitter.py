"""Split chunked audio into train/validation/test sets with stratification.

Splits at the **recording level** (not chunk level) to prevent data leakage —
all chunks from the same source recording go to the same split.

Usage::

    python -m coffee_first_crack.data_prep.dataset_splitter \\
        --input data/processed \\
        --output data/splits \\
        --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import cast

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


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
    """
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())
    total_all = total_train + total_val + total_test

    lines = [
        "# Dataset Split Report",
        "",
        "## Split Configuration",
        "",
        "- **Splitting strategy**: recording-level (prevents data leakage)",
        f"- **Total recordings**: {len(groups)}",
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
            lines.append(f"- {r}: {fc} first_crack, {nfc} no_first_crack")
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


def main() -> None:
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Split chunked dataset into train/val/test (recording-level)"
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
    args = parser.parse_args()

    total_ratio = args.train + args.val + args.test
    if not (0.99 <= total_ratio <= 1.01):
        print(f"❌ Split ratios must sum to 1.0 (got {total_ratio})")
        return

    print("📊 Dataset Splitter (recording-level)")
    print("=" * 50)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Ratios:  train={args.train}, val={args.val}, test={args.test}")
    print(f"Seed:    {args.seed}")

    groups = group_chunks_by_recording(args.input)
    if not groups:
        print("❌ No chunk files found in input directory")
        return

    total_chunks = sum(len(files) for rec in groups.values() for files in rec.values())
    print(f"\nFound {len(groups)} recordings, {total_chunks} total chunks")
    for rec_stem in sorted(groups):
        fc = len(groups[rec_stem].get("first_crack", []))
        nfc = len(groups[rec_stem].get("no_first_crack", []))
        print(f"  {rec_stem}: {fc} FC, {nfc} NFC")

    print("\n🔀 Performing recording-level stratified split...")
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
    )

    print("\n✅ Dataset split complete!")
    print(f"   Train: {args.output}/train/")
    print(f"   Val:   {args.output}/val/")
    print(f"   Test:  {args.output}/test/")


if __name__ == "__main__":
    main()
