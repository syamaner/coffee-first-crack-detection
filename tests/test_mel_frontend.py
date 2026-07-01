"""Tests for coffee_first_crack.mel_frontend.

The keystone test is ``test_numeric_mel_diff_vs_ast``: it feeds the same WAV
window through the reference ``ASTFeatureExtractor`` (numpy/torchaudio path) and
through :class:`MelFrontend` and asserts the absolute difference is below
a tight tolerance.  This is the numeric equivalence guarantee that validates the
torch-free swap.

Other tests cover the factory seam, normalisation formula, and the call interface
so the module behaves as a drop-in replacement for ``ASTFeatureExtractor``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coffee_first_crack.mel_frontend import (
    MelFrontend,
    _build_kaldi_mel_filters,
    _hann_window_symmetric,
    extract_mel,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "exports" / "onnx" / "int8"
_TEST_SPLIT = _REPO_ROOT / "data" / "splits" / "test"

_CONFIG_AVAILABLE = _CONFIG_DIR.is_dir() and (_CONFIG_DIR / "preprocessor_config.json").exists()
_TEST_DATA_AVAILABLE = _TEST_SPLIT.is_dir()


def _first_wav(subdir: str) -> Path | None:
    """Return the first WAV file in a test subdirectory, or None if absent."""
    d = _TEST_SPLIT / subdir
    if not d.is_dir():
        return None
    wavs = sorted(d.glob("*.wav"))
    return wavs[0] if wavs else None


# ── Kaldi mel filter bank ─────────────────────────────────────────────────────


def test_kaldi_mel_filters_shape() -> None:
    """Filter bank must have shape (num_freq_bins, num_mel_filters)."""
    filters = _build_kaldi_mel_filters(
        num_frequency_bins=257,
        num_mel_filters=128,
        min_frequency=20.0,
        max_frequency=8000.0,
        sampling_rate=16000,
    )
    assert filters.shape == (257, 128)
    assert filters.dtype == np.float32


def test_kaldi_mel_filters_non_negative() -> None:
    """All filter weights must be non-negative (triangular filters)."""
    filters = _build_kaldi_mel_filters(
        num_frequency_bins=257,
        num_mel_filters=128,
        min_frequency=20.0,
        max_frequency=8000.0,
        sampling_rate=16000,
    )
    assert (filters >= 0).all()


@pytest.mark.skipif(
    not _CONFIG_AVAILABLE, reason="exports/onnx/int8/preprocessor_config.json not present"
)
def test_kaldi_mel_filters_match_transformers() -> None:
    """Our Kaldi filter bank must match transformers.audio_utils.mel_filter_bank to < 1e-6."""
    pytest.importorskip("transformers")
    from transformers.audio_utils import mel_filter_bank  # type: ignore[import-untyped]

    our_filters = _build_kaldi_mel_filters(
        num_frequency_bins=257,
        num_mel_filters=128,
        min_frequency=20.0,
        max_frequency=8000.0,
        sampling_rate=16000,
    )
    ref_filters = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=128,
        min_frequency=20,
        max_frequency=8000,
        sampling_rate=16000,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True,
    )
    diff = np.abs(our_filters - ref_filters.astype(np.float32))
    assert diff.max() < 1e-6, f"Max filter diff {diff.max():.2e} exceeds 1e-6"


# ── Hann window ───────────────────────────────────────────────────────────────


def test_hann_window_shape_and_dtype() -> None:
    """Hann window must be float32 with the requested length."""
    w = _hann_window_symmetric(400)
    assert w.shape == (400,)
    assert w.dtype == np.float32


def test_hann_window_symmetric() -> None:
    """Symmetric Hann window must be equal at both ends."""
    w = _hann_window_symmetric(400)
    assert w[0] == pytest.approx(0.0, abs=1e-7)
    assert w[-1] == pytest.approx(0.0, abs=1e-7)
    # Symmetric: w[i] == w[N-1-i]
    assert np.allclose(w, w[::-1], atol=1e-7)


# ── extract_mel ───────────────────────────────────────────────────────────────


def test_extract_mel_output_shape() -> None:
    """extract_mel must return (max_length, num_mel_bins) float32."""
    rng = np.random.default_rng(42)
    waveform = rng.standard_normal(160000).astype(np.float32)
    mel_filters = _build_kaldi_mel_filters(257, 128, 20.0, 8000.0, 16000)
    window = _hann_window_symmetric(400)
    result = extract_mel(waveform, mel_filters, window, max_length=1024)
    assert result.shape == (1024, 128)
    assert result.dtype == np.float32


def test_extract_mel_silent_window_is_finite() -> None:
    """extract_mel must handle a silent window without NaN/Inf."""
    waveform = np.zeros(160000, dtype=np.float32)
    mel_filters = _build_kaldi_mel_filters(257, 128, 20.0, 8000.0, 16000)
    window = _hann_window_symmetric(400)
    result = extract_mel(waveform, mel_filters, window)
    assert np.isfinite(result).all()


def test_extract_mel_short_audio_pads_to_max_length() -> None:
    """Short audio (< 10 s) must be zero-padded to max_length frames."""
    rng = np.random.default_rng(7)
    waveform = rng.standard_normal(8000).astype(np.float32)  # 0.5 s only
    mel_filters = _build_kaldi_mel_filters(257, 128, 20.0, 8000.0, 16000)
    window = _hann_window_symmetric(400)
    result = extract_mel(waveform, mel_filters, window, max_length=1024)
    assert result.shape == (1024, 128)


# ── MelFrontend ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _CONFIG_AVAILABLE, reason="exports/onnx/int8 not present")
def test_from_config_reads_mean_std() -> None:
    """from_config must correctly read mean and std from preprocessor_config.json."""
    fe = MelFrontend.from_config(_CONFIG_DIR)
    assert fe.mean == pytest.approx(-4.2677393, rel=1e-6)
    assert fe.std == pytest.approx(4.5689974, rel=1e-6)


def test_from_config_missing_raises() -> None:
    """from_config must raise FileNotFoundError for a missing config."""
    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        MelFrontend.from_config("/tmp/nonexistent_dir_xyz")


def test_extract_output_shape() -> None:
    """extract must return (max_length, num_mel_bins) float32."""
    fe = MelFrontend(mean=-4.27, std=4.57)
    rng = np.random.default_rng(0)
    window = rng.standard_normal(160000).astype(np.float32)
    result = fe.extract(window)
    assert result.shape == (1024, 128)
    assert result.dtype == np.float32


def test_call_interface_returns_input_values() -> None:
    """__call__ must return dict with 'input_values' key, batch dim first."""
    fe = MelFrontend(mean=-4.27, std=4.57)
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(160000).astype(np.float32)
    out = fe([audio.tolist()], sampling_rate=16000, return_tensors="np")
    assert "input_values" in out
    assert out["input_values"].shape == (1, 1024, 128)


def test_call_interface_wrong_sample_rate_raises() -> None:
    """__call__ must raise ValueError when sampling_rate mismatches."""
    fe = MelFrontend(mean=-4.27, std=4.57, sampling_rate=16000)
    rng = np.random.default_rng(2)
    audio = rng.standard_normal(160000).astype(np.float32)
    with pytest.raises(ValueError, match="sampling_rate"):
        fe([audio.tolist()], sampling_rate=22050)


def test_normalisation_formula() -> None:
    """Normalisation must apply (x - mean) / (std * 2), matching ASTFeatureExtractor."""
    mean = -4.2677393
    std = 4.5689974
    fe = MelFrontend(mean=mean, std=std)
    rng = np.random.default_rng(3)
    waveform = rng.standard_normal(160000).astype(np.float32)

    # extract_mel returns un-normalised log-mel; extract() normalises
    mel_filters = _build_kaldi_mel_filters(257, 128, 20.0, 8000.0, 16000)
    window = _hann_window_symmetric(400)
    raw = extract_mel(waveform, mel_filters, window)
    expected = (raw - mean) / (std * 2)

    result = fe.extract(waveform)
    assert np.allclose(result, expected, atol=1e-6)


# ── Numeric mel-diff vs ASTFeatureExtractor (KEYSTONE) ───────────────────────

# Tolerance: max absolute difference between MelFrontend and the
# ASTFeatureExtractor numpy path on a 10 s audio window.  Empirically measured
# at < 1.3e-05; gate is set to 1e-04 with 7× headroom.
_MEL_DIFF_ATOL: float = 1e-04


@pytest.mark.skipif(
    not (_CONFIG_AVAILABLE and _TEST_DATA_AVAILABLE),
    reason="exports/onnx/int8 or data/splits/test not present",
)
def test_numeric_mel_diff_vs_ast() -> None:
    """MelFrontend input_values must match ASTFeatureExtractor to < 1e-04.

    This is the keystone numeric equivalence test for D27 Phase 1.  It confirms
    the numpy/scipy Kaldi-compatible mel front-end produces the same features the
    ONNX model was trained on (via the ASTFeatureExtractor numpy path).
    """
    pytest.importorskip("transformers")
    import librosa  # type: ignore[import-untyped]
    from transformers import ASTFeatureExtractor  # type: ignore[import-untyped]

    ast_extractor = ASTFeatureExtractor.from_pretrained(str(_CONFIG_DIR))
    our_frontend = MelFrontend.from_config(_CONFIG_DIR)

    test_wavs: list[Path] = []
    for subdir in ("first_crack", "no_first_crack"):
        wav = _first_wav(subdir)
        if wav is not None:
            test_wavs.append(wav)

    assert test_wavs, "No test WAVs found — check data/splits/test/"

    for wav_path in test_wavs:
        audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        window = audio[:160000]

        ast_out = ast_extractor([window.tolist()], sampling_rate=16000, return_tensors="np")
        iv_ast = ast_out["input_values"][0]

        our_out = our_frontend([window.tolist()], sampling_rate=16000, return_tensors="np")
        iv_our = our_out["input_values"][0]

        diff = np.abs(iv_ast - iv_our)
        assert diff.max() < _MEL_DIFF_ATOL, (
            f"{wav_path.name}: max abs diff {diff.max():.2e} exceeds {_MEL_DIFF_ATOL:.0e}"
        )
