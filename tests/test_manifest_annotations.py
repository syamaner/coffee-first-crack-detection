"""Tests for manifest-aware Label Studio conversion and mic2 derivation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.propagate_annotations as propagation
from coffee_first_crack.data_prep import convert_labelstudio_export as conversion
from coffee_first_crack.data_prep.corpus_manifest import CaptureManifest


def _manifest(staging_root: Path) -> CaptureManifest:
    pair_id = "a" * 32
    streams: list[dict[str, Any]] = []
    for mic_num, duration in ((1, 100.0), (2, 100.25)):
        name = f"{pair_id}__mic{mic_num}-fazenda-inhame-roast1.wav"
        streams.append(
            {
                "mic_num": mic_num,
                "label": f"device-{mic_num}",
                "original_filename": f"mic{mic_num}-fazenda-inhame-roast1.wav",
                "source_path": f"/captures/{pair_id}/mic{mic_num}.wav",
                "staged_relative_path": f"mic{mic_num}/{name}",
                "size_bytes": 4,
                "sha256": "0" * 64,
                "duration_seconds": duration,
                "sample_rate": 16_000,
            }
        )
    return {
        "schema_version": 1,
        "source_root": "/captures",
        "staging_root": str(staging_root),
        "session_count": 1,
        "stream_count": 2,
        "mic1_task_count": 1,
        "max_observed_duration_delta_seconds": 3.5,
        "source_files_verified_unchanged": True,
        "sessions": [
            {
                "pair_id": pair_id,
                "origin": "fazenda-inhame",
                "roast_num": 1,
                "source_session_dir": f"/captures/{pair_id}",
                "session_sidecar_source_path": f"/captures/{pair_id}/session.json",
                "session_sidecar_staged_relative_path": f"sessions/{pair_id}__session.json",
                "recording_sidecar_source_path": f"/captures/{pair_id}/roast.recording.json",
                "recording_sidecar_staged_relative_path": (
                    f"sessions/{pair_id}__roast.recording.json"
                ),
                "observed_duration_delta_seconds": 0.25,
                "streams": streams,
            }
        ],
    }


def _write_manifest(path: Path, manifest: CaptureManifest) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_conversion_preserves_human_pair_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_root = tmp_path / "mcp"
    data_root = tmp_path
    manifest = _manifest(staging_root)
    mic1 = manifest["sessions"][0]["streams"][0]
    audio_path = staging_root / mic1["staged_relative_path"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(conversion.librosa, "get_duration", lambda **_: 100.0)
    task = {
        "file_upload": f"deadbeef-{audio_path.name}",
        "annotations": [
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {"start": 20.0, "end": 50.0, "labels": ["first_crack"]},
                    }
                ]
            }
        ],
    }

    converted = conversion.convert_task(task, data_root, manifest)

    assert converted["pair_id"] == "a" * 32
    assert converted["mic_num"] == 1
    assert converted["audio_file"].startswith("mcp/mic1/")
    assert converted["sample_rate"] == 16_000
    assert converted["provenance"]["annotation_source"] == "human_label_studio"


@pytest.mark.parametrize(("start", "end"), [(float("nan"), 50.0), (20.0, float("nan"))])
def test_manifest_conversion_rejects_non_finite_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: float,
    end: float,
) -> None:
    """Malformed Label Studio numbers cannot silently invert whole-recording labels."""
    staging_root = tmp_path / "mcp"
    manifest = _manifest(staging_root)
    mic1 = manifest["sessions"][0]["streams"][0]
    audio_path = staging_root / mic1["staged_relative_path"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(conversion.librosa, "get_duration", lambda **_: 100.0)
    task = {
        "file_upload": f"deadbeef-{audio_path.name}",
        "annotations": [
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {"start": start, "end": end, "labels": ["first_crack"]},
                    }
                ]
            }
        ],
    }

    with pytest.raises(ValueError, match="invalid first_crack boundary"):
        conversion.convert_task(task, tmp_path, manifest)


def test_manifest_conversion_resolves_label_studio_truncated_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_root = tmp_path / "mcp"
    manifest = _manifest(staging_root)
    mic1 = manifest["sessions"][0]["streams"][0]
    staged_name = Path(mic1["staged_relative_path"]).name
    audio_path = staging_root / mic1["staged_relative_path"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(conversion.librosa, "get_duration", lambda **_: 100.0)
    truncated_name = f"{staged_name[:-12]}_EsoDqTW.wav"

    converted = conversion.convert_task(
        {"file_upload": f"deadbeef-{truncated_name}", "annotations": [{"result": []}]},
        tmp_path,
        manifest,
    )

    assert converted["pair_id"] == "a" * 32
    assert converted["audio_file"] == f"mcp/{mic1['staged_relative_path']}"


def test_manifest_conversion_rejects_unsubmitted_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfinished task cannot silently become whole-recording negative ground truth."""
    staging_root = tmp_path / "mcp"
    manifest = _manifest(staging_root)
    mic1 = manifest["sessions"][0]["streams"][0]
    audio_path = staging_root / mic1["staged_relative_path"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(conversion.librosa, "get_duration", lambda **_: 100.0)

    with pytest.raises(ValueError, match="exactly one submitted"):
        conversion.convert_task(
            {"file_upload": f"deadbeef-{audio_path.name}", "annotations": []},
            tmp_path,
            manifest,
        )


@pytest.mark.parametrize(
    "annotations",
    [None, ["malformed"], [{"was_cancelled": "false", "result": []}]],
)
def test_manifest_conversion_rejects_malformed_submission_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    annotations: object,
) -> None:
    """Malformed Label Studio submission metadata fails closed."""
    staging_root = tmp_path / "mcp"
    manifest = _manifest(staging_root)
    mic1 = manifest["sessions"][0]["streams"][0]
    audio_path = staging_root / mic1["staged_relative_path"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(conversion.librosa, "get_duration", lambda **_: 100.0)

    with pytest.raises(ValueError, match="annotation|cancellation"):
        conversion.convert_task(
            {"file_upload": f"deadbeef-{audio_path.name}", "annotations": annotations},
            tmp_path,
            manifest,
        )


def test_manifest_conversion_accepts_submitted_explicit_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submitted annotation object with an empty result remains a valid negative."""
    staging_root = tmp_path / "mcp"
    manifest = _manifest(staging_root)
    mic1 = manifest["sessions"][0]["streams"][0]
    audio_path = staging_root / mic1["staged_relative_path"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(conversion.librosa, "get_duration", lambda **_: 100.0)

    converted = conversion.convert_task(
        {"file_upload": f"deadbeef-{audio_path.name}", "annotations": [{"result": []}]},
        tmp_path,
        manifest,
    )

    assert converted["annotations"] == []


def test_manifest_conversion_rejects_inconsistent_truncated_upload(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "mcp")
    uploaded_name = f"{'a' * 32}__mic1-unrelated_EsoDqTW.wav"

    with pytest.raises(ValueError, match="does not match the manifest basename"):
        conversion.resolve_manifest_stream(uploaded_name, manifest)


@pytest.mark.parametrize(
    ("uploaded_name", "message"),
    [
        ("unknown.wav", "not present in the capture manifest"),
        ("unknown_EsoDqTW.wav", "not present in the capture manifest"),
        (f"{'b' * 32}__mic1-roast_EsoDqTW.wav", "unknown pair_id"),
    ],
)
def test_manifest_conversion_rejects_unresolvable_uploads(
    tmp_path: Path,
    uploaded_name: str,
    message: str,
) -> None:
    manifest = _manifest(tmp_path / "mcp")

    with pytest.raises(ValueError, match=message):
        conversion.resolve_manifest_stream(uploaded_name, manifest)


def test_manifest_conversion_rejects_missing_truncated_mic(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "mcp")
    manifest["sessions"][0]["streams"] = [manifest["sessions"][0]["streams"][1]]
    uploaded_name = f"{'a' * 32}__mic1-fazenda_EsoDqTW.wav"

    with pytest.raises(ValueError, match="ambiguous mic identity"):
        conversion.resolve_manifest_stream(uploaded_name, manifest)


def test_manifest_derivation_records_uncertainty_and_pair_identity(tmp_path: Path) -> None:
    staging_root = tmp_path / "mcp"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    manifest = _manifest(staging_root)
    manifest_path = staging_root / "capture_manifest.json"
    staging_root.mkdir()
    _write_manifest(manifest_path, manifest)
    session = manifest["sessions"][0]
    primary, target = session["streams"]
    target_audio = staging_root / target["staged_relative_path"]
    target_audio.parent.mkdir(parents=True)
    target_audio.write_bytes(b"stub")
    primary_label_path = labels_dir / f"{Path(primary['staged_relative_path']).stem}.json"
    primary_label_path.write_text(
        json.dumps(
            {
                "audio_file": primary["staged_relative_path"],
                "duration": 100.0,
                "sample_rate": 16_000,
                "pair_id": session["pair_id"],
                "mic_num": 1,
                "annotations": [{"start_time": 20.0, "end_time": 50.0, "label": "first_crack"}],
            }
        ),
        encoding="utf-8",
    )

    written, skipped = propagation.propagate_manifest(
        manifest_path,
        labels_dir,
        staging_root,
        overwrite=False,
        dry_run=False,
    )

    assert (written, skipped) == (1, 0)
    derived_path = labels_dir / f"{Path(target['staged_relative_path']).stem}.json"
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    assert derived["pair_id"] == session["pair_id"]
    assert derived["mic_num"] == 2
    assert derived["provenance"]["derived_from"] == primary["staged_relative_path"]
    assert derived["provenance"]["alignment_uncertainty_seconds"] is None
    assert derived["provenance"]["alignment_uncertainty_status"] == (
        "unbounded_historical_missing_stream_start_offsets"
    )
    assert derived["provenance"]["observed_pair_duration_delta_seconds"] == 0.25
    assert derived["provenance"]["alignment"] == "independent_clocks_not_sample_locked"
    assert derived["provenance"]["training_policy"] == (
        "exclude_all_derived_mic2_without_verified_alignment"
    )


def test_manifest_derivation_fails_on_missing_human_annotation(tmp_path: Path) -> None:
    staging_root = tmp_path / "mcp"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    manifest = _manifest(staging_root)
    manifest_path = staging_root / "capture_manifest.json"
    staging_root.mkdir()
    _write_manifest(manifest_path, manifest)

    with pytest.raises(FileNotFoundError, match="Missing human mic1 annotation"):
        propagation.propagate_manifest(
            manifest_path,
            labels_dir,
            staging_root,
            overwrite=False,
            dry_run=False,
        )
