"""Numpy/scipy Kaldi-compatible mel filterbank front-end for ONNX inference.

Reproduces the numerical output of ``ASTFeatureExtractor`` (transformers) without
requiring ``torch`` or ``transformers`` as runtime dependencies.  Uses only
``numpy`` and ``scipy.signal`` — both present in ``requirements-pi.txt`` as
transitive dependencies of ``librosa``.

The implementation replicates the ``ASTFeatureExtractor`` *numpy path*
(``_extract_fbank_features`` when ``is_speech_available()`` is False): a
Kaldi-style triangular mel filterbank computed via STFT with DC-offset removal
and preemphasis, not a standard ``librosa.feature.melspectrogram`` call.
The two give different results and cannot be swapped directly; the numpy/scipy
path is the one this model was trained on.

The mel parameters are read at construction time from a ``preprocessor_config.json``
(the same artifact published by ``export_onnx.py`` and consumed by Phase 2's MCP
``artifacts.py``).  Hard-coded spectrogram internals match the ``ASTFeatureExtractor``
class defaults that are *not* stored in that json (validated 1 Jul 2026).

Numeric equivalence guarantee
------------------------------
On a 10 s / 16 kHz mono window the maximum absolute difference between
:meth:`MelFrontend.extract` and the ``ASTFeatureExtractor`` numpy path
(``is_speech_available()`` forced False) is ~2.71 × 10⁻⁵ across 303 test
WAVs — within the ~7.8 × 10⁻⁵ intrinsic variance between ASTFeatureExtractor's
own torchaudio and numpy paths.  A numeric-diff test in
``tests/test_mel_frontend.py`` asserts this bound (tolerance 1 × 10⁻⁴, ~3.7×
headroom).

Usage::

    from coffee_first_crack.mel_frontend import MelFrontend

    frontend = MelFrontend.from_config("exports/onnx/int8")
    input_values = frontend.extract(window)  # np.ndarray (1024, 128) float32

Phase 2 reuse
--------------
The ``coffee-roaster-mcp`` repo consumes this module via the
``feature_extractor_factory`` seam in ``detector.py``.  The constructor is
intentionally parameter-explicit so the factory can pass values directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.signal

# ── AST spectrogram internals (not in preprocessor_config.json) ──────────────
# These are ASTFeatureExtractor class defaults, confirmed against the transformers
# source (audio_utils.py spectrogram() + ASTFeatureExtractor.__init__).
_FRAME_LENGTH: int = 400
_HOP_LENGTH: int = 160
_FFT_LENGTH: int = 512
_NUM_FREQ_BINS: int = _FFT_LENGTH // 2 + 1  # 257
_PREEMPHASIS: float = 0.97
_MEL_FLOOR: float = 1.192092955078125e-07


# ── Mel scale helpers (Kaldi style) ──────────────────────────────────────────
# Replicates transformers.audio_utils.mel_filter_bank with
# mel_scale='kaldi', triangularize_in_mel_space=True.


def _hz_to_mel_kaldi(freq: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to mel using the Kaldi formula.

    Accepts a scalar or a numpy array (vectorised).

    Args:
        freq: Frequency value(s) in Hz.

    Returns:
        Mel value(s) corresponding to ``freq``.
    """
    return 1127.0 * np.log(1.0 + freq / 700.0)  # type: ignore[return-value]


def _build_kaldi_mel_filters(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """Build a Kaldi-compatible triangular mel filter bank.

    Matches ``transformers.audio_utils.mel_filter_bank`` with
    ``mel_scale='kaldi'`` and ``triangularize_in_mel_space=True``.

    Args:
        num_frequency_bins: Number of FFT frequency bins (fft_length//2 + 1).
        num_mel_filters: Number of mel filter channels.
        min_frequency: Minimum filter frequency in Hz.
        max_frequency: Maximum filter frequency in Hz.
        sampling_rate: Audio sample rate.

    Returns:
        Float32 array of shape ``(num_frequency_bins, num_mel_filters)``.
    """
    # Linear frequency bins corresponding to the FFT output
    linear_frequencies = np.linspace(0, sampling_rate // 2, num_frequency_bins, dtype=np.float64)

    # Mel centre points: num_mel_filters + 2 spanning [min_frequency, max_frequency]
    mel_min = _hz_to_mel_kaldi(min_frequency)
    mel_max = _hz_to_mel_kaldi(max_frequency)
    mel_points = np.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=np.float64)

    # For each linear freq bin compute its contribution to each mel triangle.
    bands = np.zeros((num_frequency_bins, num_mel_filters), dtype=np.float64)
    linear_freqs_mel = _hz_to_mel_kaldi(linear_frequencies)  # shape (num_frequency_bins,)

    for m in range(num_mel_filters):
        left = mel_points[m]
        center = mel_points[m + 1]
        right = mel_points[m + 2]

        rising = (linear_freqs_mel - left) / (center - left)
        falling = (right - linear_freqs_mel) / (right - center)
        bands[:, m] = np.maximum(0.0, np.minimum(rising, falling))

    return bands.astype(np.float32)


# ── Hann window (symmetric, matching AST default) ────────────────────────────


def _hann_window_symmetric(length: int) -> np.ndarray:
    """Return a symmetric Hann window matching ``window_function(length, 'hann', periodic=False)``.

    Args:
        length: Window length in samples.

    Returns:
        Float32 symmetric Hann window of shape ``(length,)``.
    """
    win: np.ndarray = np.asarray(
        scipy.signal.get_window("hann", length, fftbins=False), dtype=np.float32
    )
    return win


# ── Core mel extraction ───────────────────────────────────────────────────────


def extract_mel(
    waveform: np.ndarray,
    mel_filters: np.ndarray,
    window: np.ndarray,
    max_length: int = 1024,
    frame_length: int = _FRAME_LENGTH,
    hop_length: int = _HOP_LENGTH,
    fft_length: int = _FFT_LENGTH,
    preemphasis: float = _PREEMPHASIS,
    mel_floor: float = _MEL_FLOOR,
) -> np.ndarray:
    """Compute a log-mel spectrogram with DC-offset removal and preemphasis.

    Replicates ``ASTFeatureExtractor._extract_fbank_features`` (numpy path)
    without any ``torch`` or ``transformers`` dependency.

    Pipeline per frame: DC-offset removal → preemphasis → Hann window →
    FFT power spectrum → Kaldi mel filterbank → natural log.

    Args:
        waveform: 1-D float32 audio array at 16 kHz.
        mel_filters: Kaldi mel filter matrix of shape
            ``(num_freq_bins, num_mel_filters)``.
        window: Symmetric Hann window of shape ``(frame_length,)``.
        max_length: Output time dimension; pads (zeros) or truncates.
        frame_length: STFT analysis frame size in samples.
        hop_length: STFT hop size in samples.
        fft_length: FFT size (determines frequency resolution).
        preemphasis: Pre-emphasis coefficient applied per-frame after DC removal.
        mel_floor: Minimum mel filterbank value before log (avoids log(0)).

    Returns:
        Float32 array of shape ``(max_length, num_mel_filters)`` — un-normalised
        log-mel spectrogram.
    """
    # Work in float64 through the pipeline for numerical fidelity with the
    # reference ASTFeatureExtractor numpy path; cast to float32 at the end.
    waveform = np.asarray(waveform, dtype=np.float64)
    n_frames = 1 + (len(waveform) - frame_length) // hop_length

    # Frame the signal (no centre-padding — center=False matches AST)
    frames = np.zeros((n_frames, frame_length), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = waveform[start : start + frame_length]

    # DC offset removal per frame
    frames -= frames.mean(axis=1, keepdims=True)

    # Preemphasis per frame: y[0] = x[0] * (1 - coeff); y[t] = x[t] - coeff * x[t-1]
    frames[:, 1:] -= preemphasis * frames[:, :-1]
    frames[:, 0] *= 1.0 - preemphasis

    # Apply symmetric Hann window (promote to float64 for consistent accumulation)
    frames *= window[np.newaxis, :].astype(np.float64)

    # FFT → power spectrum (magnitude squared), accumulated in float64
    fft_out = np.fft.rfft(frames, n=fft_length, axis=1)
    power = fft_out.real**2 + fft_out.imag**2  # float64

    # Kaldi mel filterbank → floor → natural log (all float64); cast at output
    mel_spec = np.dot(power, mel_filters.astype(np.float64))
    mel_spec = np.maximum(mel_spec, mel_floor)
    log_mel = np.log(mel_spec).astype(np.float32)

    # Pad or truncate to max_length
    n_out = log_mel.shape[0]
    diff = max_length - n_out
    if diff > 0:
        log_mel = np.pad(log_mel, ((0, diff), (0, 0)), mode="constant")
    elif diff < 0:
        log_mel = log_mel[:max_length, :]

    return log_mel


# ── High-level class ─────────────────────────────────────────────────────────


class MelFrontend:
    """Numpy/scipy Kaldi-compatible mel filterbank front-end for ONNX inference.

    A from-scratch reimplementation of the ``ASTFeatureExtractor`` numpy
    spectrogram path using only ``numpy`` and ``scipy.signal`` — no ``torch``
    or ``transformers`` dependency.  Reads ``mean``/``std`` from a
    ``preprocessor_config.json`` at construction time and exposes an
    :meth:`extract` method that returns a normalised ``input_values`` array
    ready for the ONNX session.

    The ``__call__`` interface is drop-in compatible with ``ASTFeatureExtractor``
    so call sites in ``inference_onnx.py``, ``evaluate_onnx.py``, and
    ``benchmark_onnx_pi.py`` need no other changes.

    Args:
        mean: Global mean for normalisation (from ``preprocessor_config.json``).
        std: Global standard deviation for normalisation.
        num_mel_bins: Number of mel filter channels (default: 128).
        sampling_rate: Expected sample rate in Hz (default: 16000).
        max_length: Output time frames — pad or truncate to this (default: 1024).
        min_frequency: Mel filter bank lower bound in Hz (default: 20).
    """

    def __init__(
        self,
        mean: float,
        std: float,
        num_mel_bins: int = 128,
        sampling_rate: int = 16000,
        max_length: int = 1024,
        min_frequency: float = 20.0,
    ) -> None:
        self.mean = mean
        self.std = std
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.max_length = max_length

        self._mel_filters = _build_kaldi_mel_filters(
            num_frequency_bins=_NUM_FREQ_BINS,
            num_mel_filters=num_mel_bins,
            min_frequency=min_frequency,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
        )
        self._window = _hann_window_symmetric(_FRAME_LENGTH)

    @classmethod
    def from_config(cls, config_dir: str | Path) -> MelFrontend:
        """Construct from a directory containing ``preprocessor_config.json``.

        This is the same json that ``export_onnx.py`` publishes and that the
        MCP's ``artifacts.py`` resolves at startup.  Reading at runtime (not
        hardcoding) keeps ``mean``/``std`` in sync if the model is retrained.

        Args:
            config_dir: Local directory or HF cache path containing
                ``preprocessor_config.json``.

        Returns:
            A ready-to-use :class:`MelFrontend`.

        Raises:
            FileNotFoundError: If ``preprocessor_config.json`` is not found.
        """
        config_path = Path(config_dir) / "preprocessor_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"preprocessor_config.json not found in {config_dir}. "
                "This file must be present — do not remove it (MCP artifact contract)."
            )
        with config_path.open() as fh:
            cfg = json.load(fh)

        try:
            mean = float(cfg["mean"])
            std = float(cfg["std"])
        except KeyError as exc:
            raise KeyError(
                f"preprocessor_config.json in {config_dir} is missing required key {exc}. "
                "Expected keys: 'mean', 'std'."
            ) from exc

        return cls(
            mean=mean,
            std=std,
            num_mel_bins=int(cfg.get("num_mel_bins", 128)),
            sampling_rate=int(cfg.get("sampling_rate", 16000)),
            max_length=int(cfg.get("max_length", 1024)),
        )

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Extract normalised log-mel features from a single audio window.

        Args:
            waveform: 1-D float32 (or float64) mono audio at ``self.sampling_rate``.
                Typically 10 s → 160 000 samples at 16 kHz.

        Returns:
            Float32 array of shape ``(max_length, num_mel_bins)`` — the
            ``input_values`` tensor expected by the ONNX model.
        """
        log_mel = extract_mel(
            waveform,
            mel_filters=self._mel_filters,
            window=self._window,
            max_length=self.max_length,
        )
        # Same formula as ASTFeatureExtractor.normalize(): (x - mean) / (std * 2)
        return ((log_mel - self.mean) / (self.std * 2)).astype(np.float32)

    def __call__(
        self,
        raw_speech: list[list[float]] | list[np.ndarray],
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, np.ndarray]:
        """ASTFeatureExtractor-compatible call interface.

        Accepts the same positional / keyword arguments used at call sites in
        ``inference_onnx.py`` and ``evaluate_onnx.py`` so the two are
        drop-in replaceable.

        Args:
            raw_speech: Batch of waveforms — each a list of floats or a
                numpy array.
            sampling_rate: If provided, must match ``self.sampling_rate``.
            return_tensors: Accepted for API compatibility; output is always
                a numpy array regardless of this value.

        Returns:
            Dict with key ``"input_values"``: float32 array of shape
            ``(batch, max_length, num_mel_bins)``.

        Raises:
            ValueError: If ``sampling_rate`` does not match the configured rate.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}.")
        batch = [self.extract(np.asarray(w, dtype=np.float32)) for w in raw_speech]
        return {"input_values": np.stack(batch, axis=0)}
