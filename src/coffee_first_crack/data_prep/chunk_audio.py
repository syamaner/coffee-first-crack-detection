"""Chunk full-length recordings into fixed-size labelled training windows.

Slides a fixed-size window (default 10 s) across each recording and labels
each window based on its overlap with annotated ``first_crack`` regions.

Replaces the prototype's ``audio_processor.py`` which extracted variable-length
chunks matching each annotated region.

Usage::

    python -m coffee_first_crack.data_prep.chunk_audio \\
        --labels-dir data/labels \\
        --audio-dir data/raw \\
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def compute_overlap(
    window_start: float,
    window_end: float,
    regions: list[dict[str, Any]],
    label: str = "first_crack",
) -> float:
    """Compute total overlap between a window and all regions of a given label.

    Args:
        window_start: Start time of the window in seconds.
        window_end: End time of the window in seconds.
        regions: List of annotation dicts with ``start_time``, ``end_time``,
            and ``label`` keys.
        label: The label to match (default: ``"first_crack"``).

    Returns:
        Union overlap in seconds (cannot exceed window duration).
    """
    # Collect intervals clipped to the window, then merge to compute union.
    intervals: list[tuple[float, float]] = []
    for r in regions:
        if r["label"] != label:
            continue
        seg_start = max(window_start, r["start_time"])
        seg_end = min(window_end, r["end_time"])
        if seg_end > seg_start:
            intervals.append((seg_start, seg_end))
    if not intervals:
        return 0.0
    # Merge overlapping intervals
    intervals.sort()
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def label_window(
    window_start: float,
    window_end: float,
    regions: list[dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> str:
    """Determine the label for a single window.

    Args:
        window_start: Start time of the window in seconds.
        window_end: End time of the window in seconds.
        regions: List of annotation dicts.
        overlap_threshold: Fraction of the window that must overlap with
            ``first_crack`` regions to be labelled as such (default: 0.5).

    Returns:
        ``"first_crack"`` or ``"no_first_crack"``.
    """
    window_duration = window_end - window_start
    overlap = compute_overlap(window_start, window_end, regions)
    if overlap >= overlap_threshold * window_duration:
        return "first_crack"
    return "no_first_crack"


def chunk_recording(
    audio: np.ndarray,
    sr: int,
    regions: list[dict[str, Any]],
    window_size: float = 10.0,
    hop_size: float | None = None,
    overlap_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Chunk a single recording into fixed-size labelled windows.

    Args:
        audio: Audio waveform array (mono).
        sr: Sample rate.
        regions: Annotation regions for this recording.
        window_size: Window length in seconds (default: 10).
        hop_size: Hop between windows in seconds.  Defaults to ``window_size``
            (no overlap).
        overlap_threshold: Fraction of overlap required with ``first_crack``
            regions to label a window as ``first_crack``.

    Returns:
        List of dicts, each with ``start_sec``, ``end_sec``, ``label``,
        ``overlap_sec``, and ``samples`` (numpy array).
    """
    hop_size = window_size if hop_size is None else hop_size

    if window_size <= 0:
        raise ValueError(f"window_size must be > 0 seconds, got {window_size!r}")
    if hop_size <= 0:
        raise ValueError(f"hop_size must be > 0 seconds, got {hop_size!r}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr!r}")

    window_samples = round(window_size * sr)
    hop_samples = round(hop_size * sr)

    if window_samples < 1:
        raise ValueError(
            "window_size is too small for the sample rate: "
            f"round(window_size * sr) must be >= 1, got {window_samples} "
            f"for window_size={window_size!r}, sr={sr!r}"
        )
    if hop_samples < 1:
        raise ValueError(
            "hop_size is too small for the sample rate: "
            f"round(hop_size * sr) must be >= 1, got {hop_samples} "
            f"for hop_size={hop_size!r}, sr={sr!r}"
        )
    chunks: list[dict[str, Any]] = []
    sample_pos = 0

    while sample_pos + window_samples <= len(audio):
        pos_sec = sample_pos / sr
        end_sec = (sample_pos + window_samples) / sr
        lbl = label_window(pos_sec, end_sec, regions, overlap_threshold)
        overlap = compute_overlap(pos_sec, end_sec, regions)
        samples = audio[sample_pos : sample_pos + window_samples]

        chunks.append(
            {
                "start_sec": pos_sec,
                "end_sec": end_sec,
                "label": lbl,
                "overlap_sec": overlap,
                "samples": samples,
            }
        )
        sample_pos += hop_samples

    return chunks


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_annotation(path: Path) -> dict[str, Any]:
    """Load a per-file annotation JSON.

    Args:
        path: Path to the annotation JSON.

    Returns:
        Parsed annotation dict.
    """
    with path.open("r") as f:
        return json.load(f)


def save_chunk(samples: np.ndarray, path: Path, sr: int) -> None:
    """Write a WAV chunk to disk.

    Args:
        samples: Audio samples to write.
        path: Output file path.
        sr: Sample rate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)


def process_recording(
    annotation_path: Path,
    audio_dir: Path,
    output_dir: Path,
    window_size: float = 10.0,
    hop_size: float | None = None,
    overlap_threshold: float = 0.5,
    sample_rate: int = 44100,
) -> dict[str, int]:
    """Process a single recording into chunked WAV files.

    Args:
        annotation_path: Path to the per-file annotation JSON.
        audio_dir: Directory containing the source WAV files.
        output_dir: Root output directory (will contain ``first_crack/``
            and ``no_first_crack/`` subdirectories).
        window_size: Window length in seconds.
        hop_size: Hop between windows in seconds.
        overlap_threshold: Fraction threshold for ``first_crack`` labelling.
        sample_rate: Sample rate to load audio at.

    Returns:
        Dict with counts ``{"first_crack": N, "no_first_crack": M}``.
    """
    ann = load_annotation(annotation_path)
    audio_file = ann["audio_file"]
    audio_path = audio_dir / audio_file

    if not audio_path.exists():
        print(f"⚠️  Audio file not found: {audio_path}")
        return {"first_crack": 0, "no_first_crack": 0}

    print(f"\n📁 Processing: {audio_file}")
    audio, loaded_sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    sr = int(loaded_sr)
    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f}s, Sample rate: {sr}Hz")

    chunks = chunk_recording(
        audio, sr, ann["annotations"], window_size, hop_size, overlap_threshold
    )

    counts: dict[str, int] = {"first_crack": 0, "no_first_crack": 0}
    stem = Path(audio_file).stem

    for chunk in chunks:
        lbl: str = chunk["label"]
        start: float = chunk["start_sec"]
        filename = f"{stem}_w{start:06.1f}.wav"
        out_path = output_dir / lbl / filename
        save_chunk(chunk["samples"], out_path, sr)
        counts[lbl] = counts.get(lbl, 0) + 1

    print(f"   ✅ Created {len(chunks)} chunks")
    print(f"      - first_crack: {counts['first_crack']}")
    print(f"      - no_first_crack: {counts['no_first_crack']}")
    return counts


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def generate_summary(
    output_dir: Path,
    all_counts: list[dict[str, int]],
    annotation_files: list[Path],
    window_size: float,
    hop_size: float,
    overlap_threshold: float,
) -> None:
    """Write a processing summary markdown file.

    Args:
        output_dir: Root output directory.
        all_counts: Per-recording count dicts.
        annotation_files: Paths to annotation JSONs that were processed.
        window_size: Window size used.
        hop_size: Hop size used.
        overlap_threshold: Overlap threshold used.
    """
    total_fc = sum(c.get("first_crack", 0) for c in all_counts)
    total_nfc = sum(c.get("no_first_crack", 0) for c in all_counts)
    total = total_fc + total_nfc

    lines = [
        "# Audio Processing Summary",
        "",
        "**Generated by**: chunk_audio.py",
        "",
        "## Parameters",
        "",
        f"- Window size: {window_size}s",
        f"- Hop size: {hop_size}s",
        f"- Overlap threshold: {overlap_threshold:.0%}",
        "",
        "## Overview",
        "",
        f"- **Total files processed**: {len(annotation_files)}",
        f"- **Total chunks created**: {total}",
        f"  - first_crack: {total_fc}",
        f"  - no_first_crack: {total_nfc}",
        "",
        "## Class Balance",
        "",
    ]

    if total > 0:
        lines.append(f"- first_crack: {total_fc / total * 100:.1f}%")
        lines.append(f"- no_first_crack: {total_nfc / total * 100:.1f}%")
    lines.append("")

    lines.extend(
        [
            "## Per-File Breakdown",
            "",
            "| File | First Crack | No First Crack | Total |",
            "|------|-------------|----------------|-------|",
        ]
    )
    for ann_file, counts in zip(annotation_files, all_counts, strict=True):
        fc = counts.get("first_crack", 0)
        nfc = counts.get("no_first_crack", 0)
        lines.append(f"| {ann_file.stem} | {fc} | {nfc} | {fc + nfc} |")

    report_path = output_dir / "processing_summary.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n📊 Summary report saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for chunking audio into fixed-size training windows."""
    parser = argparse.ArgumentParser(
        description="Chunk recordings into fixed-size labelled training windows"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/labels"),
        help="Directory containing per-file annotation JSONs (default: data/labels)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing source WAV files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for chunked WAVs (default: data/processed)",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=10.0,
        help="Window length in seconds (default: 10)",
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        default=None,
        help="Hop between windows in seconds (default: same as window-size)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Fraction of window overlapping first_crack to label as such (default: 0.5)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for loading audio (default: 44100)",
    )
    args = parser.parse_args()

    hop = args.hop_size if args.hop_size is not None else args.window_size

    # Find per-file annotation JSONs (exclude Label Studio exports / backups)
    candidates = sorted(args.labels_dir.glob("*.json"))
    annotation_files = [
        p
        for p in candidates
        if not (p.name.startswith("project-") or p.name.startswith("labelstudio-export-"))
    ]

    if not annotation_files:
        print(f"❌ No annotation files found in {args.labels_dir}")
        print("   Hint: run convert_labelstudio_export first")
        return

    print("🎵 Audio Chunk Processor (sliding window)")
    print("=" * 50)
    print(f"Annotation files: {len(annotation_files)}")
    print(f"Audio directory:  {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Window size:      {args.window_size}s")
    print(f"Hop size:         {hop}s")
    print(f"Overlap threshold:{args.overlap_threshold:.0%}")
    print(f"Sample rate:      {args.sample_rate}Hz")

    all_counts: list[dict[str, int]] = []
    for ann_file in annotation_files:
        counts = process_recording(
            ann_file,
            args.audio_dir,
            args.output_dir,
            window_size=args.window_size,
            hop_size=hop,
            overlap_threshold=args.overlap_threshold,
            sample_rate=args.sample_rate,
        )
        all_counts.append(counts)

    generate_summary(
        args.output_dir,
        all_counts,
        annotation_files,
        args.window_size,
        hop,
        args.overlap_threshold,
    )

    total = sum(sum(c.values()) for c in all_counts)
    print(f"\n✅ Processing complete! {total} chunks in {args.output_dir}")


if __name__ == "__main__":
    main()
