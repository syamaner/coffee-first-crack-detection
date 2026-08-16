"""Tests for the fail-closed MCP full-recording holdout evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import scripts.evaluate_mcp_heldout as replay


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one test JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    """Return a test file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, duration_sec: float = 10.5) -> str:
    """Write a small mono 16 kHz PCM fixture and return its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(round(16_000 * duration_sec), dtype=np.float32)
    sf.write(path, samples, 16_000, subtype="PCM_16")
    return _sha256(path)


def _discovery_fixture(tmp_path: Path) -> dict[str, Any]:
    """Build one used dataset pair and one fresh paired holdout."""
    split_integrity = tmp_path / "split_integrity.json"
    chunk_manifest = tmp_path / "chunk_manifest.jsonl"
    dataset_manifest = tmp_path / "dataset_manifest.json"
    holdout_manifest = tmp_path / "holdout_manifest.json"
    labels_dir = tmp_path / "labels"
    sidecar = tmp_path / "fresh" / "roast.recording.json"
    mic1 = tmp_path / "fresh" / "mic1-fresh.wav"
    mic2 = tmp_path / "fresh" / "mic2-fresh.wav"
    mic1_sha = _write_wav(mic1)
    mic2_sha = _write_wav(mic2)

    _write_json(
        split_integrity,
        {
            "splits": {
                "train": {"pair_ids": ["used"], "stream_recording_count": 2},
                "validation": {"pair_ids": [], "stream_recording_count": 0},
                "test": {"pair_ids": [], "stream_recording_count": 0},
            }
        },
    )
    chunk_manifest.write_text(
        "\n".join(
            [
                json.dumps({"pair_id": "used", "recording_id": "used__mic1"}),
                json.dumps({"pair_id": "used", "recording_id": "used__mic2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        dataset_manifest,
        {
            "sessions": [
                {
                    "pair_id": "used",
                    "streams": [{"sha256": "used-1"}, {"sha256": "used-2"}],
                }
            ]
        },
    )
    _write_json(
        sidecar,
        {"session_id": "fresh", "milestones": {"beans_added": 2.5, "first_crack": 8.0}},
    )
    streams = [
        {
            "mic_num": 1,
            "label": "primary",
            "duration_seconds": 10.5,
            "sha256": mic1_sha,
            "source_path": str(mic1),
            "staged_relative_path": "mic1/fresh__mic1-fresh.wav",
        },
        {
            "mic_num": 2,
            "label": "paired",
            "duration_seconds": 10.5,
            "sha256": mic2_sha,
            "source_path": str(mic2),
            "staged_relative_path": "mic2/fresh__mic2-fresh.wav",
        },
    ]
    _write_json(
        holdout_manifest,
        {
            "sessions": [
                {
                    "pair_id": "fresh",
                    "recording_sidecar_source_path": str(sidecar),
                    "streams": streams,
                }
            ]
        },
    )
    for recording_id, mic_num in (("fresh__mic1-fresh", 1), ("fresh__mic2-fresh", 2)):
        _write_json(
            labels_dir / f"{recording_id}.json",
            {
                "pair_id": "fresh",
                "mic_num": mic_num,
                "annotations": [
                    {
                        "label": "first_crack",
                        "start_time": 7.0,
                        "end_time": 10.0,
                    }
                ],
            },
        )
    return {
        "split_integrity_path": split_integrity,
        "chunk_manifest_path": chunk_manifest,
        "dataset_capture_manifest_path": dataset_manifest,
        "holdout_capture_manifest_path": holdout_manifest,
        "label_dirs": (labels_dir,),
        "pair_ids": {"fresh"},
        "window_seconds": 10.0,
    }


def test_discovers_fresh_pair_with_authoritative_t0(tmp_path: Path) -> None:
    """A fresh pair retains both checksums, mic identities, labels, and T0."""
    fixture = _discovery_fixture(tmp_path)
    fixture["pair_ids"] = None
    recordings = replay.discover_heldout_recordings(**fixture)

    assert len(recordings) == 2
    assert [recording.mic_num for recording in recordings] == [1, 2]
    assert {recording.pair_id for recording in recordings} == {"fresh"}
    assert {recording.t0_offset_sec for recording in recordings} == {2.5}
    assert all(recording.region is not None for recording in recordings)


def test_rejects_pair_already_present_in_split(tmp_path: Path) -> None:
    """A session ID exposed to any split cannot be called a fresh holdout."""
    fixture = _discovery_fixture(tmp_path)
    split = json.loads(fixture["split_integrity_path"].read_text())
    split["splits"]["train"]["pair_ids"].append("fresh")
    split["splits"]["train"]["stream_recording_count"] = 4
    _write_json(fixture["split_integrity_path"], split)
    with fixture["chunk_manifest_path"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pair_id": "fresh", "recording_id": "fresh__mic1-fresh"}))
        handle.write("\n")
        handle.write(json.dumps({"pair_id": "fresh", "recording_id": "fresh__mic2-fresh"}))
        handle.write("\n")

    with pytest.raises(ValueError, match="already appears in a dataset split"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_source_checksum_seen_in_dataset(tmp_path: Path) -> None:
    """A fresh UUID cannot hide reuse of an already exposed source stream."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    dataset = json.loads(fixture["dataset_capture_manifest_path"].read_text())
    dataset["sessions"][0]["streams"][0]["sha256"] = holdout["sessions"][0]["streams"][0]["sha256"]
    _write_json(fixture["dataset_capture_manifest_path"], dataset)

    with pytest.raises(ValueError, match="source checksum already exposed"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_missing_t0_alignment(tmp_path: Path) -> None:
    """The evaluator never assumes that WAV time zero equals charge."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    sidecar = Path(holdout["sessions"][0]["recording_sidecar_source_path"])
    _write_json(sidecar, {"milestones": {"beans_added": None, "first_crack": None}})

    with pytest.raises(ValueError, match="authoritative recording-relative beans_added"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_recording_shorter_than_detector_window(tmp_path: Path) -> None:
    """Aborted captures shorter than one window are not valid full-roast holdouts."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    holdout["sessions"][0]["streams"][0]["duration_seconds"] = 6.0
    _write_json(fixture["holdout_capture_manifest_path"], holdout)

    with pytest.raises(ValueError, match="at least 10.00s is required"):
        replay.discover_heldout_recordings(**fixture)


@pytest.mark.parametrize(
    ("region", "detected_sec", "expected"),
    [
        (None, None, "true_negative"),
        (None, 5.0, "false_positive"),
        (replay.FirstCrackRegion(20.0, 30.0, 0.0, "human"), None, "missed"),
        (
            replay.FirstCrackRegion(20.0, 30.0, 0.0, "human"),
            5.0,
            "premature_false_alert",
        ),
        (replay.FirstCrackRegion(20.0, 30.0, 0.0, "human"), 12.0, "detected"),
    ],
)
def test_classify_outcome(
    region: replay.FirstCrackRegion | None,
    detected_sec: float | None,
    expected: str,
) -> None:
    """Recording outcomes distinguish misses and premature alerts."""
    assert (
        replay.classify_outcome(region=region, detected_sec=detected_sec, window_seconds=10.0)
        == expected
    )


def test_frozen_protocol_cannot_change_after_creation(tmp_path: Path) -> None:
    """Model/profile/cohort changes require a new output protocol lock."""
    path = tmp_path / "protocol.json"
    replay._freeze_protocol(path, {"model": "a", "threshold": 0.6})
    replay._freeze_protocol(path, {"model": "a", "threshold": 0.6})

    with pytest.raises(ValueError, match="differs from this invocation"):
        replay._freeze_protocol(path, {"model": "a", "threshold": 0.7})
