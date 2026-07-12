"""Tests for scripts/rf_mfcc_baseline.py.

Hardware/data-free — uses synthetic audio, never touches ``data/splits/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

import scripts.rf_mfcc_baseline as rf_baseline


def _write_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    """Write a tiny synthetic WAV file at the given path (silence, fast to load)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    sf.write(str(path), samples, sample_rate)


def test_extract_mfcc_features_shape() -> None:
    """The feature vector is 2 * N_MFCC (mean + std per coefficient)."""
    rng = np.random.default_rng(seed=0)
    audio = rng.standard_normal(rf_baseline.SAMPLE_RATE * 10).astype(np.float32)

    features = rf_baseline.extract_mfcc_features(audio)

    assert features.shape == (2 * rf_baseline.N_MFCC,)
    assert np.all(np.isfinite(features))


def test_extract_mfcc_features_deterministic() -> None:
    """The same audio always yields the same feature vector (no hidden RNG)."""
    rng = np.random.default_rng(seed=1)
    audio = rng.standard_normal(rf_baseline.SAMPLE_RATE * 10).astype(np.float32)

    first = rf_baseline.extract_mfcc_features(audio)
    second = rf_baseline.extract_mfcc_features(audio.copy())

    np.testing.assert_array_equal(first, second)


def test_extract_mfcc_features_distinguishes_silence_from_noise() -> None:
    """Silence and noise should not collapse to an identical feature vector."""
    silence = np.zeros(rf_baseline.SAMPLE_RATE * 10, dtype=np.float32)
    rng = np.random.default_rng(seed=2)
    noise = rng.standard_normal(rf_baseline.SAMPLE_RATE * 10).astype(np.float32)

    silence_features = rf_baseline.extract_mfcc_features(silence)
    noise_features = rf_baseline.extract_mfcc_features(noise)

    assert not np.allclose(silence_features, noise_features)


def test_collect_samples_reads_label_directories(tmp_path: Path) -> None:
    """`_collect_samples` walks first_crack/no_first_crack subdirs and labels them."""
    fc_dir = tmp_path / "first_crack"
    nfc_dir = tmp_path / "no_first_crack"
    fc_dir.mkdir()
    nfc_dir.mkdir()
    (fc_dir / "a.wav").touch()
    (nfc_dir / "b.wav").touch()
    (nfc_dir / "c.wav").touch()

    samples = rf_baseline._collect_samples(tmp_path)

    labels = sorted(label for _, label in samples)
    assert labels == [0, 0, 1]
    assert len(samples) == 3


def test_collect_samples_missing_label_dir_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing label subdirectory is skipped with a warning, not a crash."""
    (tmp_path / "first_crack").mkdir()

    samples = rf_baseline._collect_samples(tmp_path)

    assert samples == []
    assert "no_first_crack" in capsys.readouterr().out


def test_run_raises_on_empty_train_dir(tmp_path: Path) -> None:
    """`run()` fails fast if the training split has no samples."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    (train_dir / "first_crack").mkdir(parents=True)
    (train_dir / "no_first_crack").mkdir(parents=True)
    (test_dir / "first_crack").mkdir(parents=True)
    (test_dir / "no_first_crack").mkdir(parents=True)

    with pytest.raises(ValueError, match="No training samples"):
        rf_baseline.run(train_dir=train_dir, test_dir=test_dir)


def test_run_fits_before_reading_test_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run()` must fully fit the classifier on train data before it ever reads the
    test directory — a leakage-regression guard for the "train, then evaluate,
    never the other way round" protocol invariant.

    Wraps ``RandomForestClassifier.fit`` and the module's ``_collect_samples`` to
    record call order into a shared event log, using real (silent) WAV fixtures
    for both splits so the wrapped calls exercise the actual code path, not a
    stub. Asserts ``fit`` is logged before the test directory's ``_collect_samples``
    call — i.e. training is complete before test data is even enumerated, let
    alone loaded or scored.
    """
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    for split_dir, count in ((train_dir, 4), (test_dir, 2)):
        for label in ("first_crack", "no_first_crack"):
            for i in range(count):
                _write_wav(split_dir / label / f"chunk_{i:03d}.wav")

    events: list[str] = []

    real_fit = RandomForestClassifier.fit

    def _tracked_fit(self: RandomForestClassifier, *args: object, **kwargs: object) -> object:
        events.append("fit")
        return real_fit(self, *args, **kwargs)

    real_collect_samples = rf_baseline._collect_samples

    def _tracked_collect_samples(split_dir: Path) -> list[tuple[Path, int]]:
        events.append(f"collect_samples:{split_dir.name}")
        return real_collect_samples(split_dir)

    monkeypatch.setattr(RandomForestClassifier, "fit", _tracked_fit)
    monkeypatch.setattr(rf_baseline, "_collect_samples", _tracked_collect_samples)

    rf_baseline.run(train_dir=train_dir, test_dir=test_dir)

    assert "fit" in events, "fit() was never called"
    fit_index = events.index("fit")
    test_collect_index = events.index(f"collect_samples:{test_dir.name}")
    assert fit_index < test_collect_index, (
        f"classifier must be fit before the test directory is read; got order {events}"
    )
