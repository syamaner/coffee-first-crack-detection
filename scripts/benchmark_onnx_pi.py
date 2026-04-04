#!/usr/bin/env python3
"""Benchmark ONNX inference latency — no PyTorch dependency.

Designed to run on Raspberry Pi 5 with only ``requirements-pi.txt`` installed.
Uses dummy audio to measure pure inference throughput without I/O variance.

Usage::

    python scripts/benchmark_onnx_pi.py \
        --onnx-dir exports/onnx \
        --n-runs 30 \
        --output results/pi5_latency.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from transformers import ASTFeatureExtractor

# ASTFeatureExtractor parameters (same as model.py)
FEATURE_EXTRACTOR_KWARGS: dict[str, object] = {
    "max_length": 1024,
    "num_mel_bins": 128,
    "sampling_rate": 16000,
    "do_normalize": True,
    "mean": -4.2677393,
    "std": 4.5689974,
}

SAMPLE_RATE = 16000
WINDOW_SEC = 10.0


def benchmark_model(
    onnx_path: Path,
    extractor: ASTFeatureExtractor,
    n_warmup: int = 5,
    n_runs: int = 30,
) -> dict[str, object]:
    """Benchmark a single ONNX model.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        extractor: Pre-built ``ASTFeatureExtractor``.
        n_warmup: Number of warmup runs (not counted).
        n_runs: Number of timed runs.

    Returns:
        Dict with backend name, latency stats, and model size.
    """
    import onnxruntime as rt

    print(f"\n[{onnx_path.parent.name}/{onnx_path.name}]")
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Dummy 10s audio window
    dummy_audio = np.random.randn(int(SAMPLE_RATE * WINDOW_SEC)).astype(np.float32)

    # Warmup (includes feature extraction)
    for _ in range(n_warmup):
        inputs = extractor([dummy_audio.tolist()], sampling_rate=SAMPLE_RATE, return_tensors="np")
        sess.run(None, {input_name: inputs["input_values"]})

    # Timed runs — end-to-end (feature extraction + ONNX inference)
    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inputs = extractor([dummy_audio.tolist()], sampling_rate=SAMPLE_RATE, return_tensors="np")
        sess.run(None, {input_name: inputs["input_values"]})
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    stats = {
        "backend": f"onnxruntime/cpu/{onnx_path.parent.name}",
        "model_file": onnx_path.name,
        "model_size_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "mean_ms": round(float(np.mean(lat)), 1),
        "min_ms": round(float(np.min(lat)), 1),
        "max_ms": round(float(np.max(lat)), 1),
        "std_ms": round(float(np.std(lat)), 1),
        "n_runs": n_runs,
    }
    print(
        f"  p50={stats['p50_ms']}ms  p95={stats['p95_ms']}ms  "
        f"mean={stats['mean_ms']}ms  size={stats['model_size_mb']}MB"
    )
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX inference latency (no PyTorch required)"
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        required=True,
        help="Root ONNX directory containing fp32/ and int8/ subdirs",
    )
    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup runs (default: 5)")
    parser.add_argument("--n-runs", type=int, default=30, help="Timed runs (default: 30)")
    parser.add_argument("--output", type=Path, default=None, help="Path for JSON results")
    args = parser.parse_args()

    import onnxruntime as rt

    # System info
    print("=" * 60)
    print("ONNX LATENCY BENCHMARK")
    print("=" * 60)
    print(f"  Platform:      {platform.machine()} / {platform.system()}")
    print(f"  Python:        {platform.python_version()}")
    print(f"  ONNX Runtime:  {rt.__version__}")
    print(f"  Window:        {WINDOW_SEC}s @ {SAMPLE_RATE}Hz")
    print(f"  Warmup:        {args.n_warmup} runs")
    print(f"  Timed:         {args.n_runs} runs")

    extractor = ASTFeatureExtractor(**FEATURE_EXTRACTOR_KWARGS)

    # Find all ONNX models
    onnx_files = sorted(args.onnx_dir.rglob("*.onnx"))
    if not onnx_files:
        print(f"Error: no .onnx files found under {args.onnx_dir}")
        raise SystemExit(1)

    results: list[dict[str, object]] = []
    for onnx_path in onnx_files:
        stats = benchmark_model(onnx_path, extractor, args.n_warmup, args.n_runs)
        results.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (per 10s audio window)")
    print("=" * 60)
    target_ms = 500
    for r in results:
        p50 = r["p50_ms"]
        status = "✅" if p50 < target_ms else "⚠️ "
        print(f"  {status} {r['backend']:40s} p50={p50:>6}ms  (target <{target_ms}ms)")

    # Save results
    output = {
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "onnxruntime": rt.__version__,
        },
        "config": {
            "window_sec": WINDOW_SEC,
            "sample_rate": SAMPLE_RATE,
            "n_warmup": args.n_warmup,
            "n_runs": args.n_runs,
        },
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
