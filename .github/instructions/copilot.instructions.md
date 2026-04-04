# Copilot Code Review Instructions

## Project Overview

Coffee first crack audio detection using an Audio Spectrogram Transformer (AST).
Full project rules are in `AGENTS.md` at the repo root.

## Key Conventions

### PyTorch + ONNX Runtime
- ONNX scripts (`evaluate_onnx.py`, `benchmark_onnx_pi.py`, `inference_onnx.py`) use **ONNX Runtime for model inference** but still require **PyTorch CPU** for `ASTFeatureExtractor` filterbank computation. This is intentional and documented — do not flag it as inconsistent.
- The `ASTFeatureExtractor` is always called with batch format: `extractor([audio.tolist()], ...)`, not `extractor(audio.tolist(), ...)`.

### HuggingFace Hub
- Models are loaded from HF Hub using `hf_hub_download` + `ASTFeatureExtractor.from_pretrained(repo_id, subfolder=...)`.
- Exception handling in HF download loops: only `EntryNotFoundError` should be swallowed when trying alternative filenames. `RepositoryNotFoundError` and other exceptions must propagate.

### Raspberry Pi 5 Deployment
- Production config: INT8 ONNX, 2 threads, threshold=0.90, overlap=0.3.
- The Pi also runs an MCP server and agent UI — 2 ONNX threads leaves 2 cores free for those services.
- Threshold 0.90 is an **intentional production tradeoff** over the sweep script's automatic 0.95 recommendation, to preserve higher recall (0.909 vs 0.773) with only 1 FP.

### Code Style
- Python 3.11+ with full type hints on all public functions.
- Google-style docstrings.
- `ruff check` and `ruff format` must pass.
- Parameter defaults use `if param is None` pattern, not `param or default` (to support falsy values like 0.0).

### Result Files
- `results/*.json` files are committed evaluation artifacts. They may contain model filenames (not full paths) and use the current `evaluate_onnx.py` output schema.
- The `simulation.json` uses the `SimulationResult` dataclass schema from `scripts/simulate_detection.py`.

### Testing
- Tests in `tests/` — run with `pytest tests/ -v`.
- Do not create test fakes/mocks in production code.
