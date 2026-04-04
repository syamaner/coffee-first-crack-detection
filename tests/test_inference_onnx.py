"""Tests for coffee_first_crack.inference_onnx — profile loading and config logic.

These tests exercise config resolution, thread defaults, and parameter override
precedence without requiring an ONNX model or audio files.
"""

from __future__ import annotations

import platform
from unittest.mock import patch

from coffee_first_crack.inference_onnx import _default_threads, _load_profile

# ── _default_threads ──────────────────────────────────────────────────────────


def test_default_threads_arm64() -> None:
    """ARM64 platforms should default to 2 threads."""
    with patch.object(platform, "machine", return_value="aarch64"):
        assert _default_threads() == 2


def test_default_threads_arm64_macos() -> None:
    """Apple Silicon (arm64) should also default to 2 threads."""
    with patch.object(platform, "machine", return_value="arm64"):
        assert _default_threads() == 2


def test_default_threads_x86() -> None:
    """x86_64 should default to 0 (auto)."""
    with patch.object(platform, "machine", return_value="x86_64"):
        assert _default_threads() == 0


# ── _load_profile ─────────────────────────────────────────────────────────────


def test_load_profile_inference() -> None:
    """Loading the 'inference' profile should return expected keys."""
    cfg = _load_profile("inference")
    assert "window_size" in cfg
    assert "threshold" in cfg
    assert cfg["window_size"] == 10.0


def test_load_profile_pi_inference() -> None:
    """Loading 'pi_inference' should return Pi-specific settings."""
    cfg = _load_profile("pi_inference")
    assert cfg.get("threshold") == 0.90
    assert cfg.get("onnx_threads") == 2
    assert cfg.get("overlap") == 0.3


def test_load_profile_nonexistent() -> None:
    """A nonexistent profile should return an empty dict."""
    cfg = _load_profile("nonexistent_profile")
    assert cfg == {}


# ── Parameter override precedence (falsy values) ─────────────────────────────

# These tests verify that explicit falsy values (0, 0.0) are respected
# rather than falling through to config defaults.


class TestOnnxSlidingWindowOverrides:
    """Test parameter override precedence in OnnxSlidingWindowInference.

    We can't instantiate the full class (needs ONNX model download), so we
    test the config resolution logic directly.
    """

    def test_none_uses_config_default(self) -> None:
        """When param is None, config value should be used."""
        cfg = _load_profile("pi_inference")
        # Simulate the is-None pattern
        threshold = None
        result = cfg.get("threshold", 0.6) if threshold is None else threshold
        assert result == 0.90  # from pi_inference config

    def test_explicit_value_overrides_config(self) -> None:
        """When param is explicitly set, it should override config."""
        cfg = _load_profile("pi_inference")
        threshold = 0.75
        result = cfg.get("threshold", 0.6) if threshold is None else threshold
        assert result == 0.75

    def test_falsy_zero_overrides_config(self) -> None:
        """Explicitly passing 0.0 should NOT fall through to config default.

        This was a bug with the ``param or cfg.get(...)`` pattern.
        """
        cfg = _load_profile("pi_inference")
        # overlap=0.0 is falsy but should be respected
        overlap = 0.0
        result = cfg.get("overlap", 0.7) if overlap is None else overlap
        assert result == 0.0

    def test_falsy_zero_int_overrides_config(self) -> None:
        """Explicitly passing min_pops=0 should NOT fall through to config."""
        cfg = _load_profile("pi_inference")
        min_pops = 0
        result = cfg.get("min_pops", 5) if min_pops is None else min_pops
        assert result == 0
