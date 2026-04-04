#!/usr/bin/env python3
"""Simulate sliding-window first-crack detection across parameter combinations.

Uses per-sample probabilities from a threshold sweep (produced by
``evaluate_onnx.py --threshold-sweep``) to explore the detection parameter
space *without* running inference again.  This makes it cheap and fast to
evaluate different combinations of overlap, threshold, min_pops, and
confirmation_window on any machine.

Usage::

    python scripts/simulate_detection.py \
        --sweep-results results/threshold_sweep.json \
        --output results/simulation.json

    # Custom parameter grid
    python scripts/simulate_detection.py \
        --sweep-results results/threshold_sweep.json \
        --thresholds 0.6 0.7 0.75 0.8 \
        --overlaps 0.5 0.6 0.7 \
        --min-pops 3 4 5 \
        --confirmation-windows 20 25 30 \
        --output results/simulation.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

WINDOW_SIZE_SEC = 10.0


@dataclass
class SimulationResult:
    """Result of one parameter combination applied to a sequence of samples."""

    threshold: float
    overlap: float
    min_pops: int
    confirmation_window: float
    hop_sec: float
    windows_per_confirmation: int
    detection_triggered: bool
    detection_delay_sec: float | None
    """Seconds from the first true-positive sample to confirmed detection."""
    true_positives_detected: int
    false_positives_in_window: int
    missed_detections: int
    first_crack_samples: int
    no_first_crack_samples: int


def _simulate_sequence(
    samples: list[dict],
    threshold: float,
    overlap: float,
    min_pops: int,
    confirmation_window: float,
) -> SimulationResult:
    """Simulate sliding-window detection on a sequence of samples.

    Samples are treated as if they arrive in order, spaced by ``hop_sec``.
    The confirmation logic mirrors ``inference.py``: count the number of
    positive windows within a trailing ``confirmation_window`` and trigger
    when ``min_pops`` is reached.

    Args:
        samples: List of dicts with ``label_id`` (0/1) and ``prob`` (float).
        threshold: Classification threshold.
        overlap: Window overlap fraction (determines hop spacing).
        min_pops: Positive windows required for confirmation.
        confirmation_window: Time span (seconds) over which pops are counted.

    Returns:
        A :class:`SimulationResult` with detection metrics.
    """
    hop_sec = WINDOW_SIZE_SEC * (1 - overlap)
    windows_per_conf = int(confirmation_window / hop_sec)

    first_crack_samples = sum(1 for s in samples if s["label_id"] == 1)
    no_first_crack_samples = sum(1 for s in samples if s["label_id"] == 0)

    # Walk through samples as a time series
    history: list[tuple[float, bool, int]] = []  # (time, is_positive, label_id)
    detection_triggered = False
    detection_delay_sec: float | None = None
    first_true_positive_time: float | None = None

    for i, sample in enumerate(samples):
        current_time = i * hop_sec
        is_positive = sample["prob"] >= threshold

        if sample["label_id"] == 1 and is_positive and first_true_positive_time is None:
            first_true_positive_time = current_time

        history.append((current_time, is_positive, sample["label_id"]))

        # Count pops within confirmation window
        cutoff = current_time - confirmation_window
        recent_positives = sum(1 for t, pos, _ in history if t >= cutoff and pos)

        if not detection_triggered and recent_positives >= min_pops:
            detection_triggered = True
            if first_true_positive_time is not None:
                detection_delay_sec = current_time - first_true_positive_time

    # Count FPs in the whole sequence: positive predictions on no_first_crack samples
    false_positives = sum(1 for s in samples if s["label_id"] == 0 and s["prob"] >= threshold)
    # Missed detections: first_crack samples predicted negative
    missed = sum(1 for s in samples if s["label_id"] == 1 and s["prob"] < threshold)

    return SimulationResult(
        threshold=threshold,
        overlap=overlap,
        min_pops=min_pops,
        confirmation_window=confirmation_window,
        hop_sec=round(hop_sec, 2),
        windows_per_confirmation=windows_per_conf,
        detection_triggered=detection_triggered,
        detection_delay_sec=(
            round(detection_delay_sec, 2) if detection_delay_sec is not None else None
        ),
        true_positives_detected=first_crack_samples - missed,
        false_positives_in_window=false_positives,
        missed_detections=missed,
        first_crack_samples=first_crack_samples,
        no_first_crack_samples=no_first_crack_samples,
    )


def simulate(
    sweep_path: Path,
    thresholds: list[float] | None = None,
    overlaps: list[float] | None = None,
    min_pops_list: list[int] | None = None,
    confirmation_windows: list[float] | None = None,
    output_path: Path | None = None,
) -> list[SimulationResult]:
    """Run detection simulation across a parameter grid.

    Args:
        sweep_path: Path to threshold sweep JSON (from ``evaluate_onnx.py --threshold-sweep``).
        thresholds: Thresholds to test (default: 0.6, 0.7, 0.75, 0.8, 0.85).
        overlaps: Overlap fractions to test (default: 0.5, 0.6, 0.7).
        min_pops_list: min_pops values to test (default: 3, 4, 5).
        confirmation_windows: Confirmation windows in seconds (default: 20, 25, 30).
        output_path: Optional path to write JSON results.

    Returns:
        List of :class:`SimulationResult` for each parameter combination.
    """
    with sweep_path.open() as f:
        sweep_data = json.load(f)

    samples = sweep_data["per_sample_probabilities"]
    if not samples:
        raise ValueError(f"No per-sample probabilities found in {sweep_path}")

    # Default parameter grids
    if thresholds is None:
        thresholds = [0.6, 0.7, 0.75, 0.8, 0.85]
    if overlaps is None:
        overlaps = [0.5, 0.6, 0.7]
    if min_pops_list is None:
        min_pops_list = [3, 4, 5]
    if confirmation_windows is None:
        confirmation_windows = [20.0, 25.0, 30.0]

    results: list[SimulationResult] = []
    total = len(thresholds) * len(overlaps) * len(min_pops_list) * len(confirmation_windows)
    print(f"Running {total} parameter combinations on {len(samples)} samples...\n")

    for thresh in thresholds:
        for overlap in overlaps:
            for min_pops in min_pops_list:
                for conf_window in confirmation_windows:
                    result = _simulate_sequence(samples, thresh, overlap, min_pops, conf_window)
                    results.append(result)

    # Print summary table
    print("=" * 100)
    print("SIMULATION RESULTS")
    print("=" * 100)
    print(
        f"{'Thresh':>7s}  {'Overlap':>7s}  {'Pops':>4s}  {'ConfWin':>7s}  "
        f"{'Hop(s)':>6s}  {'Win/CW':>6s}  {'Detect':>6s}  "
        f"{'Delay(s)':>8s}  {'FP':>4s}  {'Missed':>6s}"
    )
    print("-" * 100)

    for r in results:
        delay_str = f"{r.detection_delay_sec:.1f}" if r.detection_delay_sec is not None else "N/A"
        detect_str = "YES" if r.detection_triggered else "NO"
        line1 = f"  {r.threshold:5.2f}  {r.overlap:7.2f}  {r.min_pops:4d}"
        line2 = f"  {r.confirmation_window:7.1f}  {r.hop_sec:6.1f}"
        line3 = f"  {r.windows_per_confirmation:6d}  {detect_str:>6s}"
        line4 = f"  {delay_str:>8s}  {r.false_positives_in_window:4d}"
        line5 = f"  {r.missed_detections:6d}"
        print(f"{line1}{line2}{line3}{line4}{line5}")

    # Highlight best candidates: detection triggered + fewest FPs + shortest delay
    triggered = [r for r in results if r.detection_triggered]
    if triggered:
        best = min(
            triggered,
            key=lambda r: (r.false_positives_in_window, r.detection_delay_sec or float("inf")),
        )
        print(
            f"\nBest candidate: threshold={best.threshold}, overlap={best.overlap}, "
            f"min_pops={best.min_pops}, confirmation_window={best.confirmation_window}s"
        )
        print(
            f"  Detection delay: {best.detection_delay_sec}s, "
            f"FPs: {best.false_positives_in_window}, "
            f"Missed: {best.missed_detections}"
        )
    else:
        print("\nNo parameter combination triggered detection.")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialised = [asdict(r) for r in results]
        with output_path.open("w") as f:
            json.dump(serialised, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate detection parameter combinations using threshold sweep probabilities"
    )
    parser.add_argument(
        "--sweep-results",
        type=Path,
        required=True,
        help="Path to threshold_sweep.json from evaluate_onnx.py --threshold-sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON simulation results",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Threshold values to simulate (default: 0.6 0.7 0.75 0.8 0.85)",
    )
    parser.add_argument(
        "--overlaps",
        type=float,
        nargs="+",
        default=None,
        help="Overlap fractions to simulate (default: 0.5 0.6 0.7)",
    )
    parser.add_argument(
        "--min-pops",
        type=int,
        nargs="+",
        default=None,
        help="min_pops values to simulate (default: 3 4 5)",
    )
    parser.add_argument(
        "--confirmation-windows",
        type=float,
        nargs="+",
        default=None,
        help="Confirmation window sizes in seconds (default: 20 25 30)",
    )
    args = parser.parse_args()

    if not args.sweep_results.exists():
        print(f"Error: sweep results not found: {args.sweep_results}")
        raise SystemExit(1)

    simulate(
        sweep_path=args.sweep_results,
        thresholds=args.thresholds,
        overlaps=args.overlaps,
        min_pops_list=args.min_pops,
        confirmation_windows=args.confirmation_windows,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
