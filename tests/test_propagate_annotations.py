"""Tests for scripts/propagate_annotations.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.propagate_annotations as pa  # noqa: E402 (scripts/ not a package)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_ANNOTATIONS: list[dict[str, Any]] = [
    {
        "id": "chunk_000",
        "start_time": 452.7,
        "end_time": 512.7,
        "label": "first_crack",
        "confidence": "high",
    }
]

_PRIMARY_LABEL = {
    "audio_file": "mic1-brazil-roast5.wav",
    "duration": 612.4,
    "sample_rate": 44100,
    "annotations": _ANNOTATIONS,
}

_SESSION_2MIC = {
    "origin": "brazil",
    "roast_num": 5,
    "sample_rate": 44100,
    "duration_sec": 612.4,
    "recorded_at": "2026-04-11T10:00:00Z",
    "mics": [
        {"mic_num": 1, "label": "fifine", "gain": 1.0, "file": "mic1-brazil-roast5.wav"},
        {"mic_num": 2, "label": "audiotechnica", "gain": 1.0, "file": "mic2-brazil-roast5.wav"},
    ],
}

_SESSION_3MIC = {
    "origin": "brazil",
    "roast_num": 6,
    "sample_rate": 44100,
    "duration_sec": 600.0,
    "recorded_at": "2026-04-11T11:00:00Z",
    "mics": [
        {"mic_num": 1, "label": "fifine", "gain": 1.0, "file": "mic1-brazil-roast6.wav"},
        {"mic_num": 2, "label": "audiotechnica", "gain": 1.0, "file": "mic2-brazil-roast6.wav"},
        {"mic_num": 3, "label": "lavalier", "gain": 0.9, "file": "mic3-brazil-roast6.wav"},
    ],
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# find_session_files
# ---------------------------------------------------------------------------


class TestFindSessionFiles:
    """Tests for session file discovery."""

    def test_finds_session_json_files(self, tmp_path: Path) -> None:
        """Matches *-session.json but not other JSON files."""
        (tmp_path / "brazil-roast5-session.json").write_text("{}")
        (tmp_path / "brazil-roast5.json").write_text("{}")  # should not match
        (tmp_path / "mic1-brazil-roast5.json").write_text("{}")  # should not match
        result = pa.find_session_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "brazil-roast5-session.json"

    def test_returns_sorted(self, tmp_path: Path) -> None:
        """Results are sorted lexicographically."""
        for name in ["c-session.json", "a-session.json", "b-session.json"]:
            (tmp_path / name).write_text("{}")
        names = [p.name for p in pa.find_session_files(tmp_path)]
        assert names == sorted(names)

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert pa.find_session_files(tmp_path) == []


# ---------------------------------------------------------------------------
# get_audio_duration
# ---------------------------------------------------------------------------


class TestGetAudioDuration:
    """Tests for WAV duration lookup."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="WAV file not found"):
            pa.get_audio_duration(tmp_path / "nonexistent.wav")

    def test_librosa_error_raises_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = tmp_path / "stub.wav"
        wav.write_bytes(b"stub")

        def _raise(**_: object) -> float:
            raise ValueError("bad")

        monkeypatch.setattr(pa.librosa, "get_duration", _raise)
        with pytest.raises(RuntimeError, match="Failed to read duration"):
            pa.get_audio_duration(wav)

    def test_returns_duration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"stub")
        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: 99.5)
        assert pa.get_audio_duration(wav) == pytest.approx(99.5)


# ---------------------------------------------------------------------------
# propagate_session — happy paths
# ---------------------------------------------------------------------------


class TestPropagateSessionHappyPath:
    """Tests for successful annotation propagation."""

    def _setup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        session: dict[str, Any],
        primary_mic: int = 1,
        duration: float = 612.4,
    ) -> tuple[Path, Path, Path]:
        """Scaffold session dir, labels dir, audio dir with session JSON and primary annotation."""
        session_dir = tmp_path / "raw"
        labels_dir = tmp_path / "labels"
        audio_dir = tmp_path / "raw"
        session_dir.mkdir()
        labels_dir.mkdir()

        origin = session["origin"]
        roast_num = session["roast_num"]

        # Session JSON
        _write_json(session_dir / f"{origin}-roast{roast_num}-session.json", session)

        # Primary annotation JSON — derive filename from mic['file'] in the session,
        # matching the logic in propagate_session.
        primary_entry = next((m for m in session["mics"] if int(m["mic_num"]) == primary_mic), None)
        primary_filename = (
            primary_entry["file"]
            if primary_entry
            else f"mic{primary_mic}-{origin}-roast{roast_num}.wav"
        )
        primary_stem = Path(primary_filename).stem
        primary_label = dict(_PRIMARY_LABEL)
        primary_label["audio_file"] = primary_filename
        _write_json(labels_dir / f"{primary_stem}.json", primary_label)

        # Stub WAVs for paired mics — use mic['file'] so names match session JSON.
        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: duration)
        for mic in session["mics"]:
            if int(mic["mic_num"]) != primary_mic:
                (audio_dir / mic["file"]).write_bytes(b"stub")

        return session_dir, labels_dir, audio_dir

    def test_two_mic_propagation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Propagates mic1 annotation to mic2; fields match spec."""
        session_dir, labels_dir, audio_dir = self._setup(tmp_path, monkeypatch, _SESSION_2MIC)
        session_path = session_dir / "brazil-roast5-session.json"

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            audio_dir,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )

        assert written == 1
        assert skipped == 0

        result = _read_json(labels_dir / "mic2-brazil-roast5.json")
        assert result["audio_file"] == "mic2-brazil-roast5.wav"
        assert result["sample_rate"] == 44100
        assert result["annotations"] == _ANNOTATIONS
        assert result["duration"] == pytest.approx(612.4)

    def test_three_mic_propagation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Propagates mic1 annotation to both mic2 and mic3."""
        session = dict(_SESSION_3MIC)
        primary_label = dict(_PRIMARY_LABEL)
        primary_label["audio_file"] = "mic1-brazil-roast6.wav"

        session_dir = tmp_path / "raw"
        labels_dir = tmp_path / "labels"
        session_dir.mkdir()
        labels_dir.mkdir()

        _write_json(session_dir / "brazil-roast6-session.json", session)
        _write_json(labels_dir / "mic1-brazil-roast6.json", primary_label)
        for n in (2, 3):
            (session_dir / f"mic{n}-brazil-roast6.wav").write_bytes(b"stub")

        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: 600.0)

        written, skipped = pa.propagate_session(
            session_dir / "brazil-roast6-session.json",
            labels_dir,
            session_dir,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )

        assert written == 2
        assert skipped == 0
        assert (labels_dir / "mic2-brazil-roast6.json").exists()
        assert (labels_dir / "mic3-brazil-roast6.json").exists()

    def test_annotations_are_deep_copied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagated annotation files are independent: mutating one does not affect another."""
        session_dir, labels_dir, audio_dir = self._setup(tmp_path, monkeypatch, _SESSION_3MIC)
        # _setup already wrote mic2 WAV stub; add mic3 WAV stub
        (audio_dir / "mic3-brazil-roast6.wav").write_bytes(b"stub")
        session_path = session_dir / "brazil-roast6-session.json"

        pa.propagate_session(
            session_path,
            labels_dir,
            audio_dir,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )

        mic2_path = labels_dir / "mic2-brazil-roast6.json"
        mic3_path = labels_dir / "mic3-brazil-roast6.json"
        r2 = _read_json(mic2_path)
        r3 = _read_json(mic3_path)
        assert r2["annotations"] == r3["annotations"]

        # Mutate mic2 on disk and confirm mic3 is unchanged.
        original_mic3_annotations = _read_json(mic3_path)["annotations"]
        r2["annotations"].append(
            {
                "id": "chunk_999",
                "start_time": 599.0,
                "end_time": 600.0,
                "label": "cooldown",
                "confidence": "low",
            }
        )
        _write_json(mic2_path, r2)

        reread_mic3 = _read_json(mic3_path)
        assert reread_mic3["annotations"] == original_mic3_annotations
        assert _read_json(mic2_path)["annotations"] != reread_mic3["annotations"]


# ---------------------------------------------------------------------------
# propagate_session — skip / guard conditions
# ---------------------------------------------------------------------------


class TestPropagateSessionSkips:
    """Tests for warning/skip behaviour."""

    def test_no_paired_mics(self, tmp_path: Path) -> None:
        """Session with only the primary mic returns (0, 0)."""
        session = {
            "origin": "brazil",
            "roast_num": 1,
            "sample_rate": 44100,
            "duration_sec": 600.0,
            "recorded_at": "2026-04-11T10:00:00Z",
            "mics": [
                {"mic_num": 1, "label": "fifine", "gain": 1.0, "file": "mic1-brazil-roast1.wav"}
            ],
        }
        session_path = tmp_path / "brazil-roast1-session.json"
        _write_json(session_path, session)
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )
        assert written == 0
        assert skipped == 0

    def test_missing_primary_annotation(self, tmp_path: Path) -> None:
        """Missing primary annotation returns (0, n_paired) and writes nothing."""
        session_path = tmp_path / "brazil-roast5-session.json"
        _write_json(session_path, _SESSION_2MIC)
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )
        assert written == 0
        assert skipped == 1  # 1 paired mic
        assert not (labels_dir / "mic2-brazil-roast5.json").exists()

    def test_missing_paired_wav_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the paired WAV is missing, that mic is skipped."""
        session_path = tmp_path / "brazil-roast5-session.json"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_json(session_path, _SESSION_2MIC)
        _write_json(labels_dir / "mic1-brazil-roast5.json", _PRIMARY_LABEL)
        # No mic2 WAV written

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )
        assert written == 0
        assert skipped == 1

    def test_existing_target_not_overwritten_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing paired JSON is skipped without --overwrite."""
        session_path = tmp_path / "brazil-roast5-session.json"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_json(session_path, _SESSION_2MIC)
        _write_json(labels_dir / "mic1-brazil-roast5.json", _PRIMARY_LABEL)

        existing_content = {"audio_file": "mic2-brazil-roast5.wav", "sentinel": True}
        _write_json(labels_dir / "mic2-brazil-roast5.json", existing_content)

        (tmp_path / "mic2-brazil-roast5.wav").write_bytes(b"stub")
        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: 612.4)

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=False,
            dry_run=False,
        )
        assert written == 0
        assert skipped == 1
        # Original file unchanged
        assert _read_json(labels_dir / "mic2-brazil-roast5.json").get("sentinel") is True

    def test_existing_target_overwritten_with_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing paired JSON is replaced when --overwrite is set."""
        session_path = tmp_path / "brazil-roast5-session.json"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_json(session_path, _SESSION_2MIC)
        _write_json(labels_dir / "mic1-brazil-roast5.json", _PRIMARY_LABEL)
        _write_json(labels_dir / "mic2-brazil-roast5.json", {"sentinel": True})
        (tmp_path / "mic2-brazil-roast5.wav").write_bytes(b"stub")
        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: 612.4)

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=True,
            dry_run=False,
        )
        assert written == 1
        assert skipped == 0
        result = _read_json(labels_dir / "mic2-brazil-roast5.json")
        assert "sentinel" not in result
        assert result["annotations"] == _ANNOTATIONS


# ---------------------------------------------------------------------------
# propagate_session — dry-run
# ---------------------------------------------------------------------------


class TestPropagateSessionDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_does_not_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry-run reports the intended write but creates no files."""
        session_path = tmp_path / "brazil-roast5-session.json"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_json(session_path, _SESSION_2MIC)
        _write_json(labels_dir / "mic1-brazil-roast5.json", _PRIMARY_LABEL)
        (tmp_path / "mic2-brazil-roast5.wav").write_bytes(b"stub")
        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: 612.4)

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=False,
            dry_run=True,
        )
        assert written == 1
        assert skipped == 0
        assert not (labels_dir / "mic2-brazil-roast5.json").exists()

    def test_dry_run_with_existing_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry-run + overwrite still reports write count without touching disk."""
        session_path = tmp_path / "brazil-roast5-session.json"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_json(session_path, _SESSION_2MIC)
        _write_json(labels_dir / "mic1-brazil-roast5.json", _PRIMARY_LABEL)
        _write_json(labels_dir / "mic2-brazil-roast5.json", {"sentinel": True})
        (tmp_path / "mic2-brazil-roast5.wav").write_bytes(b"stub")
        monkeypatch.setattr(pa.librosa, "get_duration", lambda **_: 612.4)

        written, skipped = pa.propagate_session(
            session_path,
            labels_dir,
            tmp_path,
            primary_mic=1,
            overwrite=True,
            dry_run=True,
        )
        assert written == 1
        # File must be unchanged
        assert _read_json(labels_dir / "mic2-brazil-roast5.json").get("sentinel") is True
