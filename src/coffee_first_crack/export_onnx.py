"""Export a trained model to ONNX format for edge deployment (Raspberry Pi 5).

Produces:
- ``model.onnx`` — FP32 full-precision ONNX model
- ``model_quantized.onnx`` — INT8 dynamically quantized (smaller, faster on CPU)

Usage::

    python -m coffee_first_crack.export_onnx \\
        --model-dir experiments/baseline_v1/checkpoint-best \\
        --output-dir exports/onnx \\
        --quantize
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from coffee_first_crack.model import build_feature_extractor


def export_onnx(
    model_dir: str | Path,
    output_dir: str | Path,
    quantize: bool = True,
    opset_version: int = 14,
) -> dict[str, Path]:
    """Export a fine-tuned AST model to ONNX.

    Args:
        model_dir: Path to ``save_pretrained`` checkpoint or HuggingFace model ID.
        output_dir: Directory to write ONNX files.
        quantize: If ``True``, also produce an INT8 quantized variant.
        opset_version: ONNX opset version (default: 14).

    Returns:
        Dict with keys ``"fp32"`` (and optionally ``"int8"``) mapping to output paths.
    """
    try:
        from optimum.onnxruntime import ORTModelForAudioClassification
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from optimum.onnxruntime import ORTQuantizer
    except ImportError as exc:
        raise ImportError(
            "Install optimum: pip install 'optimum[onnxruntime]'"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = str(model_dir)

    print(f"Exporting model from: {model_dir}")
    print(f"Output directory:     {output_dir}")

    # Export FP32 ONNX
    print("\nExporting FP32 ONNX...")
    ort_model = ORTModelForAudioClassification.from_pretrained(
        model_dir,
        export=True,
        opset=opset_version,
    )
    fp32_dir = output_dir / "fp32"
    fp32_dir.mkdir(exist_ok=True)
    ort_model.save_pretrained(str(fp32_dir))
    fp32_path = fp32_dir / "model.onnx"
    print(f"FP32 ONNX saved to: {fp32_path}")

    results: dict[str, Path] = {"fp32": fp32_path}

    # INT8 dynamic quantization
    if quantize:
        print("\nApplying INT8 dynamic quantization...")
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
        )
        quantizer = ORTQuantizer.from_pretrained(str(fp32_dir))
        int8_dir = output_dir / "int8"
        int8_dir.mkdir(exist_ok=True)
        quantizer.quantize(
            save_dir=str(int8_dir),
            quantization_config=quantization_config,
        )
        int8_path = int8_dir / "model_quantized.onnx"
        if not int8_path.exists():
            # fallback: optimum may use a different name
            candidates = list(int8_dir.glob("*.onnx"))
            if candidates:
                int8_path = candidates[0]
        print(f"INT8 ONNX saved to: {int8_path}")
        results["int8"] = int8_path

    # Copy feature extractor config alongside ONNX for deployment convenience
    feature_extractor = build_feature_extractor()
    feature_extractor.save_pretrained(str(output_dir))
    print(f"\nFeature extractor config saved to: {output_dir}")

    _print_size_summary(results)
    return results


def _print_size_summary(results: dict[str, Path]) -> None:
    """Print model file sizes."""
    print("\n--- Model sizes ---")
    for variant, path in results.items():
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  {variant}: {path.name} ({size_mb:.1f} MB)")


def benchmark_onnx(
    onnx_path: Path,
    n_warmup: int = 5,
    n_runs: int = 20,
    sample_rate: int = 16000,
    window_sec: float = 10.0,
) -> dict[str, float]:
    """Benchmark ONNX inference latency on the current device.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        n_warmup: Number of warmup runs (not counted).
        n_runs: Number of timed runs.
        sample_rate: Audio sample rate.
        window_sec: Audio window length in seconds.

    Returns:
        Dict with ``p50_ms``, ``p95_ms``, ``mean_ms`` latency values.
    """
    try:
        import onnxruntime as rt
    except ImportError as exc:
        raise ImportError("Install onnxruntime: pip install onnxruntime") from exc

    feature_extractor = build_feature_extractor()
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Generate a dummy audio window
    dummy_audio = np.random.randn(int(sample_rate * window_sec)).astype(np.float32)
    inputs = feature_extractor(
        [dummy_audio.tolist()],
        sampling_rate=sample_rate,
        return_tensors="np",
    )
    input_features = inputs["input_features"]

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, {input_name: input_features})

    # Timed runs
    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: input_features})
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
    }


def main() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(description="Export first crack model to ONNX")
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="save_pretrained checkpoint dir or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("exports/onnx"),
        help="Output directory (default: exports/onnx)",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Produce INT8 quantized variant (default: True, use --no-quantize to skip)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark after export",
    )
    parser.add_argument(
        "--opset", type=int, default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    results = export_onnx(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        quantize=args.quantize,
        opset_version=args.opset,
    )

    if args.benchmark:
        print("\n--- Latency benchmark ---")
        for variant, path in results.items():
            if path.exists():
                stats = benchmark_onnx(path)
                print(
                    f"  {variant}: p50={stats['p50_ms']:.1f}ms  "
                    f"p95={stats['p95_ms']:.1f}ms  "
                    f"mean={stats['mean_ms']:.1f}ms"
                )


if __name__ == "__main__":
    main()
