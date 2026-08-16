"""Tests for fail-closed MCP capture staging."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from coffee_first_crack.data_prep.corpus_manifest import load_capture_manifest
from coffee_first_crack.data_prep.ingest_mcp_captures import sha256_file, stage_captures


def _write_wav(path: Path, duration_seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(16_000 * duration_seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * frames)


def _make_session(
    capture_root: Path,
    pair_id: str,
    *,
    origin: str = "colombia-excelso-huila-washed",
    roast_num: int = 1,
) -> Path:
    session_dir = capture_root / pair_id
    session_dir.mkdir(parents=True)
    streams: list[dict[str, Any]] = []
    mics: list[dict[str, Any]] = []
    for mic_num in (1, 2):
        filename = f"mic{mic_num}-{origin}-roast{roast_num}.wav"
        wav_path = session_dir / filename
        _write_wav(wav_path, 1.0 + (0.25 if mic_num == 2 else 0.0))
        duration = 1.0 + (0.25 if mic_num == 2 else 0.0)
        streams.append(
            {
                "device": f"mic{mic_num}",
                "wav_filename": filename,
                "sample_rate": 16_000,
                "channels": 1,
                "sample_width_bytes": 2,
                "frame_count": round(duration * 16_000),
                "duration_seconds": duration,
            }
        )
        mics.append({"mic_num": mic_num, "label": f"device-{mic_num}", "file": filename})
    (session_dir / f"{origin}-roast{roast_num}-session.json").write_text(
        json.dumps(
            {
                "origin": origin,
                "roast_num": roast_num,
                "sample_rate": 16_000,
                "mics": mics,
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "roast.recording.json").write_text(
        json.dumps({"schema_version": 2, "session_id": pair_id, "streams": streams}),
        encoding="utf-8",
    )
    return session_dir


class TestMcpCaptureStaging:
    """Session discovery, collision resistance, and source integrity."""

    def test_duplicate_original_basenames_are_retained_and_sources_unchanged(
        self, tmp_path: Path
    ) -> None:
        capture_root = tmp_path / "captures"
        first = _make_session(capture_root, "a" * 32)
        second = _make_session(capture_root, "b" * 32)
        source_wavs = sorted(capture_root.glob("*/*.wav"))
        before = {path: sha256_file(path) for path in source_wavs}

        output = tmp_path / "staged"
        manifest = stage_captures(capture_root, output)

        assert manifest["session_count"] == 2
        assert manifest["stream_count"] == 4
        assert manifest["mic1_task_count"] == 2
        assert len(list((output / "mic1").glob("*.wav"))) == 2
        assert len(list((output / "mic2").glob("*.wav"))) == 2
        assert {path.name.split("__", 1)[0] for path in (output / "mic1").glob("*.wav")} == {
            "a" * 32,
            "b" * 32,
        }
        assert all(sha256_file(path) == digest for path, digest in before.items())
        assert first.exists() and second.exists()
        loaded = load_capture_manifest(output / "capture_manifest.json")
        assert loaded["source_files_verified_unchanged"] is True

    def test_one_session_yields_one_mic1_task_and_one_mic2_target(self, tmp_path: Path) -> None:
        capture_root = tmp_path / "captures"
        _make_session(capture_root, "c" * 32)

        manifest = stage_captures(capture_root, tmp_path / "staged")

        session = manifest["sessions"][0]
        assert [stream["mic_num"] for stream in session["streams"]] == [1, 2]
        assert manifest["mic1_task_count"] == 1

    def test_dry_run_validates_without_writing(self, tmp_path: Path) -> None:
        capture_root = tmp_path / "captures"
        _make_session(capture_root, "d" * 32)
        output = tmp_path / "staged"

        manifest = stage_captures(capture_root, output, dry_run=True)

        assert manifest["session_count"] == 1
        assert not output.exists()

    def test_staging_inside_capture_root_is_rejected_without_writing(self, tmp_path: Path) -> None:
        capture_root = tmp_path / "captures"
        _make_session(capture_root, "1" * 32)
        output = capture_root / "staged"

        with pytest.raises(ValueError, match="outside the immutable capture root"):
            stage_captures(capture_root, output)

        assert not output.exists()

    @pytest.mark.parametrize("defect", ["missing_wav", "malformed_sidecar", "traversal"])
    def test_invalid_sessions_fail_closed(self, tmp_path: Path, defect: str) -> None:
        capture_root = tmp_path / "captures"
        session_dir = _make_session(capture_root, "e" * 32)
        sidecar = next(session_dir.glob("*-session.json"))
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if defect == "missing_wav":
            (session_dir / data["mics"][1]["file"]).unlink()
        elif defect == "malformed_sidecar":
            sidecar.write_text("{", encoding="utf-8")
        else:
            data["mics"][1]["file"] = "../escape.wav"
            sidecar.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises((ValueError, FileNotFoundError)):
            stage_captures(capture_root, tmp_path / "staged")

    def test_duplicate_pair_id_in_manifest_is_rejected(self, tmp_path: Path) -> None:
        capture_root = tmp_path / "captures"
        _make_session(capture_root, "f" * 32)
        output = tmp_path / "staged"
        stage_captures(capture_root, output)
        raw = json.loads((output / "capture_manifest.json").read_text(encoding="utf-8"))
        raw["sessions"].append(raw["sessions"][0])
        raw["session_count"] = 2
        raw["stream_count"] = 4
        raw["mic1_task_count"] = 2
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate pair_id"):
            load_capture_manifest(path)
