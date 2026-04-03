#!/usr/bin/env python3
"""Benchmark inference latency across MPS/CUDA/CPU and ONNX backends.

Usage::

    python scripts/benchmark_platforms.py \\
        --model-dir experiments/baseline_v1/checkpoint-best \\
        --onnx-dir exports/onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from coffee_first_crack.model import FirstCrackClassifier, build_feature_extractor
from coffee_first_crack.utils.device import get_device


def _make_dummy_input(sample_rate: int = 16000, window_sec: float = 10.0) -> torch.Tensor:
    return torch.randn(1, int(sample_rate * window_sec))


def benchmark_pytorch(
    model_dir: str,
    device: str,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict[str, float]:
    """Benchmark PyTorch inference on the given device."""
    print(f"\n[PyTorch/{device.upper()}] Loading model...")
    classifier = FirstCrackClassifier(model_name_or_path=model_dir, device=device)
    classifier.model.eval()
    dummy = _make_dummy_input()

    for _ in range(n_warmup):
        _ = classifier.predict_proba(dummy)

    def _sync() -> None:
        """Synchronise async GPU/MPS ops so timing is accurate."""
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(n_runs):
            _sync()
            t0 = time.perf_counter()
            classifier.predict_proba(dummy)
            _sync()
            latencies.append((time.perf_counter() - t0) * 1000)

    stats = {
        "backend": f"pytorch/{device}",
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
        "n_runs": n_runs,
    }
    print(
        f"  p50={stats['p50_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
        f"mean={stats['mean_ms']:.1f}ms"
    )
    return stats


def benchmark_onnx_runtime(
    onnx_path: Path,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict[str, float]:
    """Benchmark ONNX Runtime inference on CPU."""
    try:
        import onnxruntime as rt
    except ImportError:
        print("onnxruntime not installed — skipping ONNX benchmark")
        return {}

    print(f"\n[ONNX Runtime/CPU] Loading {onnx_path.name}...")
    extractor = build_feature_extractor()
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    dummy_audio = np.random.randn(160000).astype(np.float32)

    # Warmup — include feature extraction to match PyTorch benchmark
    for _ in range(n_warmup):
        inputs = extractor([dummy_audio.tolist()], sampling_rate=16000, return_tensors="np")
        sess.run(None, {input_name: inputs["input_features"]})

    # Timed runs — end-to-end (feature extraction + model inference)
    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inputs = extractor([dummy_audio.tolist()], sampling_rate=16000, return_tensors="np")
        sess.run(None, {input_name: inputs["input_features"]})
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = {
        "backend": f"onnxruntime/cpu/{onnx_path.stem}",
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
        "n_runs": n_runs,
    }
    print(
        f"  p50={stats['p50_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
        f"mean={stats['mean_ms']:.1f}ms"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--onnx-dir", type=Path, default=None)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results: list[dict] = []

    # PyTorch on auto-detected device
    device = get_device()
    results.append(benchmark_pytorch(args.model_dir, device, n_runs=args.n_runs))

    # ONNX Runtime
    if args.onnx_dir and args.onnx_dir.exists():
        for onnx_file in sorted(args.onnx_dir.rglob("*.onnx")):
            r = benchmark_onnx_runtime(onnx_file, n_runs=args.n_runs)
            if r:
                results.append(r)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (per 10s audio window)")
    print("=" * 60)
    targets = {"mps": 100, "cuda": 30, "cpu": 500}
    for r in results:
        backend = r.get("backend", "")
        p50 = r.get("p50_ms", 0)
        target = next((v for k, v in targets.items() if k in backend), 500)
        status = "✅" if p50 < target else "⚠️ "
        print(f"  {status} {backend:40s} p50={p50:6.1f}ms  (target <{target}ms)")

    # Save JSON
    output = args.output or Path("experiments/benchmarks") / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
