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
) -> dict[str, Path]:
    """Export a fine-tuned AST model to ONNX.

    Args:
        model_dir: Path to ``save_pretrained`` checkpoint or HuggingFace model ID.
        output_dir: Directory to write ONNX files.
        quantize: If ``True``, also produce an INT8 quantized variant.

    Returns:
        Dict with keys ``"fp32"`` (and optionally ``"int8"``) mapping to output paths.
    """
    try:
        from optimum.onnxruntime import ORTModelForAudioClassification
    except ImportError as exc:
        raise ImportError("Install optimum: pip install 'optimum[onnxruntime]'") from exc

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
    )
    fp32_dir = output_dir / "fp32"
    fp32_dir.mkdir(exist_ok=True)
    ort_model.save_pretrained(str(fp32_dir))
    fp32_path = fp32_dir / "model.onnx"
    print(f"FP32 ONNX saved to: {fp32_path}")

    results: dict[str, Path] = {"fp32": fp32_path}

    # INT8 dynamic quantization — uses onnxruntime.quantization.quantize_dynamic
    # which is portable across architectures (x86, ARM64/RPi5, etc.)
    if quantize:
        print("\nApplying INT8 dynamic quantization (portable, ARM64-compatible)...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise ImportError("Install onnxruntime: pip install onnxruntime") from exc

        int8_dir = output_dir / "int8"
        int8_dir.mkdir(exist_ok=True)
        int8_path = int8_dir / "model_quantized.onnx"

        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        if not int8_path.exists():
            raise FileNotFoundError(
                f"Quantized ONNX model was not found after quantization in: {int8_dir}"
            )
        print(f"INT8 ONNX saved to: {int8_path}")
        results["int8"] = int8_path

    # Save feature extractor config into every variant directory so each can be
    # deployed or uploaded independently (e.g., just the int8/ folder to RPi5)
    feature_extractor = build_feature_extractor()
    config_dirs = {output_dir} | {path.parent for path in results.values()}
    for config_dir in sorted(config_dirs, key=str):
        feature_extractor.save_pretrained(str(config_dir))
    print(
        "\nFeature extractor config saved to: "
        + ", ".join(str(d) for d in sorted(config_dirs, key=str))
    )

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
    input_values = inputs["input_values"]

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, {input_name: input_values})

    # Timed runs — include feature extraction for end-to-end latency
    # (matches PyTorch benchmark which times predict_proba including feature extraction)
    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_inputs = feature_extractor(
            [dummy_audio.tolist()],
            sampling_rate=sample_rate,
            return_tensors="np",
        )
        sess.run(None, {input_name: run_inputs["input_values"]})
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
        "--model-dir",
        type=str,
        required=True,
        help="save_pretrained checkpoint dir or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/onnx"),
        help="Output directory (default: exports/onnx)",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Produce INT8 quantized variant (default: True, use --no-quantize to skip)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark after export",
    )
    args = parser.parse_args()

    results = export_onnx(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        quantize=args.quantize,
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
