"""Tests for the fail-closed MCP full-recording holdout evaluator."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    mic2_sha = _write_wav(mic2, 10.6)

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
                json.dumps(
                    {
                        "pair_id": "used",
                        "recording_id": "used__mic1",
                        "source_audio_sha256": "1" * 64,
                    }
                ),
                json.dumps(
                    {
                        "pair_id": "used",
                        "recording_id": "used__mic2",
                        "source_audio_sha256": "2" * 64,
                    }
                ),
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
                    "streams": [{"sha256": "1" * 64}, {"sha256": "2" * 64}],
                }
            ]
        },
    )
    _write_json(
        sidecar,
        {
            "session_id": "fresh",
            "milestones": {"beans_added": 2.5, "first_crack": 8.0, "drop": 10.0},
        },
    )
    streams = [
        {
            "mic_num": 1,
            "label": "primary",
            "duration_seconds": 10.6,
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
            "source_root": str(tmp_path / "fresh"),
            "sessions": [
                {
                    "pair_id": "fresh",
                    "recording_sidecar_source_path": str(sidecar),
                    "streams": streams,
                }
            ],
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
        "minimum_pair_count": 1,
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
        handle.write(
            json.dumps(
                {
                    "pair_id": "fresh",
                    "recording_id": "fresh__mic1-fresh",
                    "source_audio_sha256": "3" * 64,
                }
            )
        )
        handle.write("\n")
        handle.write(
            json.dumps(
                {
                    "pair_id": "fresh",
                    "recording_id": "fresh__mic2-fresh",
                    "source_audio_sha256": "4" * 64,
                }
            )
        )
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


def test_rejects_source_checksum_seen_in_legacy_chunk_manifest(tmp_path: Path) -> None:
    """Legacy split exposure is detected even without an MCP capture-manifest row."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    rows = [
        json.loads(line)
        for line in fixture["chunk_manifest_path"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_audio_sha256"] = holdout["sessions"][0]["streams"][0]["sha256"]
    fixture["chunk_manifest_path"].write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source checksum already exposed"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_invalid_split_source_checksum(tmp_path: Path) -> None:
    """Exposure evidence must contain auditable source hashes."""
    fixture = _discovery_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in fixture["chunk_manifest_path"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_audio_sha256"] = "invalid"
    fixture["chunk_manifest_path"].write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid source_audio_sha256"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_conflicting_split_source_checksums(tmp_path: Path) -> None:
    """One split recording cannot resolve to multiple source files."""
    fixture = _discovery_fixture(tmp_path)
    first = json.loads(fixture["chunk_manifest_path"].read_text(encoding="utf-8").splitlines()[0])
    first["source_audio_sha256"] = "3" * 64
    with fixture["chunk_manifest_path"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(first) + "\n")

    with pytest.raises(ValueError, match="Conflicting source checksums"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_duplicate_checksum_inside_holdout_cohort(tmp_path: Path) -> None:
    """A cohort cannot count the same audio bytes as two independent streams."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    first_sha = holdout["sessions"][0]["streams"][0]["sha256"]
    holdout["sessions"][0]["streams"][1]["sha256"] = first_sha
    _write_json(fixture["holdout_capture_manifest_path"], holdout)

    with pytest.raises(ValueError, match="Duplicate source checksum within holdout cohort"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_missing_t0_alignment(tmp_path: Path) -> None:
    """The evaluator never assumes that WAV time zero equals charge."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    sidecar = Path(holdout["sessions"][0]["recording_sidecar_source_path"])
    _write_json(sidecar, {"milestones": {"beans_added": None, "first_crack": None}})

    with pytest.raises(ValueError, match="authoritative recording-relative beans_added"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_missing_drop_milestone(tmp_path: Path) -> None:
    """A holdout must prove that each capture covers the complete roast."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    sidecar = Path(holdout["sessions"][0]["recording_sidecar_source_path"])
    _write_json(sidecar, {"milestones": {"beans_added": 2.5}})

    with pytest.raises(ValueError, match="beans_added or drop milestone"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_drop_after_stream_end(tmp_path: Path) -> None:
    """A drop timestamp beyond either WAV cannot establish complete-roast evidence."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    sidecar = Path(holdout["sessions"][0]["recording_sidecar_source_path"])
    _write_json(sidecar, {"milestones": {"beans_added": 2.5, "drop": 11.0}})

    with pytest.raises(ValueError, match="drop milestone .* exceeds mic1 duration"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_drop_before_charge(tmp_path: Path) -> None:
    """Completion metadata must preserve roast milestone ordering."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    sidecar = Path(holdout["sessions"][0]["recording_sidecar_source_path"])
    _write_json(sidecar, {"milestones": {"beans_added": 2.5, "drop": 2.0}})

    with pytest.raises(ValueError, match="Invalid drop milestone"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_non_finite_stream_duration(tmp_path: Path) -> None:
    """NaN duration metadata cannot bypass complete-roast validation."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    holdout["sessions"][0]["streams"][0]["duration_seconds"] = "nan"
    _write_json(fixture["holdout_capture_manifest_path"], holdout)

    with pytest.raises(ValueError, match="Invalid duration"):
        replay.discover_heldout_recordings(**fixture)


def test_requires_decisive_minimum_physical_session_count(tmp_path: Path) -> None:
    """A one-roast smoke test cannot be reported as the final holdout cohort."""
    fixture = _discovery_fixture(tmp_path)
    fixture["minimum_pair_count"] = 6

    with pytest.raises(ValueError, match="require at least 6"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_recording_shorter_than_detector_window(tmp_path: Path) -> None:
    """Aborted captures shorter than one window are not valid full-roast holdouts."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    holdout["sessions"][0]["streams"][0]["duration_seconds"] = 6.0
    _write_json(fixture["holdout_capture_manifest_path"], holdout)

    with pytest.raises(ValueError, match="at least 10.00s is required"):
        replay.discover_heldout_recordings(**fixture)


def test_rejects_source_path_outside_manifest_root(tmp_path: Path) -> None:
    """A holdout manifest cannot make the evaluator read an unrelated file."""
    fixture = _discovery_fixture(tmp_path)
    holdout = json.loads(fixture["holdout_capture_manifest_path"].read_text())
    outside = tmp_path / "outside.wav"
    holdout["sessions"][0]["streams"][0]["source_path"] = str(outside)
    _write_json(fixture["holdout_capture_manifest_path"], holdout)

    with pytest.raises(ValueError, match="escapes source_root"):
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


def test_input_snapshot_detects_changed_exposure_evidence(tmp_path: Path) -> None:
    """Split/cohort evidence cannot mutate between discovery and completed replay."""
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    paths = {"evidence": evidence}
    snapshot = replay._snapshot_inputs(paths)
    evidence.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="evidence inputs changed"):
        replay._verify_input_snapshot(paths, snapshot)


def test_input_snapshot_rejects_missing_evidence(tmp_path: Path) -> None:
    """Every frozen exposure input must exist before discovery."""
    with pytest.raises(ValueError, match="Missing replay evidence input"):
        replay._snapshot_inputs({"missing": tmp_path / "missing.json"})


def test_git_head_fails_closed_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protocol provenance cannot silently omit the MCP source revision."""
    monkeypatch.setattr(replay.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Git executable is required"):
        replay._git_head(tmp_path)


def test_git_head_rejects_dirty_selected_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit hash is insufficient provenance when local MCP changes are present."""
    monkeypatch.setattr(replay.shutil, "which", lambda _name: "/usr/bin/git")

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        stdout = "abc123\n" if "rev-parse" in argv else " M src/detector.py\n"
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(replay.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="checkout is dirty"):
        replay._git_head(tmp_path)


def test_git_head_accepts_clean_selected_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean checkout produces an exact source revision."""
    monkeypatch.setattr(replay.shutil, "which", lambda _name: "/usr/bin/git")

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        stdout = "abc123\n" if "rev-parse" in argv else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(replay.subprocess, "run", fake_run)

    assert replay._git_head(tmp_path) == "abc123"


def test_replay_counts_all_events_and_preserves_first_notification(tmp_path: Path) -> None:
    """Replay continues after confirmation so repeated MCP alerts are observable."""
    windows = [
        SimpleNamespace(started_at_monotonic_seconds=100.0),
        SimpleNamespace(started_at_monotonic_seconds=103.0),
    ]

    class FakePipeline:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def drain_windows(self, *, max_windows: int) -> list[SimpleNamespace]:
            del max_windows
            return [windows.pop(0)] if windows else []

    events = [
        SimpleNamespace(
            detected_at_monotonic_seconds=100.0,
            confirmed_at_monotonic_seconds=101.0,
            confidence=0.9,
            confirmed_by_window_sequence_number=5,
        ),
        SimpleNamespace(
            detected_at_monotonic_seconds=103.0,
            confirmed_at_monotonic_seconds=104.0,
            confidence=0.8,
            confirmed_by_window_sequence_number=6,
        ),
    ]

    class FakeAdapter:
        def process_window_observed(
            self, _window: object, *, earliest_eligible_monotonic_seconds: float
        ) -> SimpleNamespace:
            del earliest_eligible_monotonic_seconds
            return SimpleNamespace(
                confidence=events[0].confidence,
                fc_status="confirmed",
                event=events.pop(0),
            )

    mcp = {
        "AudioConfig": lambda **kwargs: kwargs,
        "build_audio_capture_pipeline": lambda _config: FakePipeline(),
        "build_first_crack_detector_adapter": lambda *_args: FakeAdapter(),
    }
    recording = replay.HeldOutRecording(
        pair_id="pair",
        recording_id="pair__mic1",
        audio_path=tmp_path / "audio.wav",
        label_path=tmp_path / "label.json",
        mic_num=1,
        mic_label="primary",
        source_sha256="a" * 64,
        t0_offset_sec=0.0,
        drop_offset_sec=20.0,
        region=replay.FirstCrackRegion(0.0, 10.0, 0.0, "human"),
    )

    result = replay._evaluate_recording(
        mcp=mcp,
        config=object(),
        artifacts=SimpleNamespace(),
        backend=object(),
        recording=recording,
        replay_path=recording.audio_path,
        resampled=False,
        window_seconds=10.0,
        overlap=0.7,
    )

    assert result["event_count"] == 2
    assert result["processed_window_count"] == 2
    assert result["detected_sec"] == 0.0
    assert result["confirming_window_sequence"] == 5


def test_mcp_import_rejects_preloaded_ambiguous_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requested MCP checkout cannot be shadowed by a cached module."""
    source = tmp_path / "src" / "coffee_roaster_mcp"
    source.mkdir(parents=True)
    (source / "detector.py").write_text("", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "coffee_roaster_mcp", ModuleType("coffee_roaster_mcp"))

    with pytest.raises(RuntimeError, match="refusing ambiguous provenance"):
        replay._load_mcp(tmp_path / "src")
