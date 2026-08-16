"""Tests for uncertainty-aware chunking and physical-pair splitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import coffee_first_crack.data_prep.chunk_audio as chunking
from coffee_first_crack.data_prep.chunk_audio import (
    build_legacy_pair_index,
    intersects_uncertain_boundary,
    process_recording,
)
from coffee_first_crack.data_prep.dataset_splitter import (
    generate_split_report,
    group_chunks_by_pair,
    recording_level_split,
)


def test_uncertainty_guard_excludes_only_boundary_sensitive_windows() -> None:
    regions = [{"start_time": 20.0, "end_time": 50.0, "label": "first_crack"}]

    assert intersects_uncertain_boundary(10.0, 20.0, regions, 3.5)
    assert intersects_uncertain_boundary(20.0, 30.0, regions, 3.5)
    assert intersects_uncertain_boundary(40.0, 50.0, regions, 3.5)
    assert not intersects_uncertain_boundary(30.0, 40.0, regions, 3.5)
    assert not intersects_uncertain_boundary(0.0, 10.0, regions, 3.5)


def test_derived_boundary_windows_are_reported_and_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "mic2.wav").write_bytes(b"stub")
    annotation_path = tmp_path / "mic2.json"
    annotation_path.write_text(
        json.dumps(
            {
                "audio_file": "mic2.wav",
                "pair_id": "pair-1",
                "mic_num": 2,
                "annotations": [{"start_time": 20.0, "end_time": 50.0, "label": "first_crack"}],
                "provenance": {
                    "annotation_source": "derived_from_paired_mic",
                    "alignment_uncertainty_seconds": 3.5,
                    "training_policy": "exclude_windows_intersecting_boundary_guard_band",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chunking.librosa,
        "load",
        lambda *_args, **_kwargs: (np.zeros(60 * 100, dtype=np.float32), 100),
    )
    monkeypatch.setattr(chunking, "save_chunk", lambda *_args, **_kwargs: None)
    records: list[dict[str, object]] = []

    counts = process_recording(
        annotation_path,
        audio_root,
        tmp_path / "processed",
        window_size=10.0,
        sample_rate=100,
        chunk_manifest_records=records,
    )

    assert counts["excluded_uncertain"] == 4
    assert sum(record["included"] is False for record in records) == 4
    assert all(
        record.get("exclusion_reason") == "derived_boundary_alignment_uncertainty"
        for record in records
        if record["included"] is False
    )


def test_panama_style_session_names_resolve_to_one_pair(tmp_path: Path) -> None:
    sidecar = {
        "mics": [
            {"mic_num": 1, "file": "mic1-panama-hortigal-estate-roast1.wav"},
            {"mic_num": 2, "file": "mic2-panama-hortigal-estate-roast1.wav"},
        ]
    }
    (tmp_path / "panama-hortigal-estate-roast1-session.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    index = build_legacy_pair_index(tmp_path)

    assert index["mic1-panama-hortigal-estate-roast1.wav"][0] == (
        "legacy:panama-hortigal-estate-roast1"
    )
    assert (
        index["mic1-panama-hortigal-estate-roast1.wav"][0]
        == index["mic2-panama-hortigal-estate-roast1.wav"][0]
    )


def _write_pair_chunks(root: Path, pair_count: int = 12) -> Path:
    records: list[dict[str, object]] = []
    for pair_number in range(pair_count):
        pair_id = f"pair-{pair_number:02d}"
        for mic_num in (1, 2):
            filename = f"{pair_id}__mic{mic_num}_w0000.0.wav"
            path = root / "first_crack" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")
            records.append(
                {
                    "chunk_filename": filename,
                    "recording_id": f"{pair_id}__mic{mic_num}",
                    "pair_id": pair_id,
                    "mic_num": mic_num,
                    "label": "first_crack",
                    "included": True,
                }
            )
    manifest = root / "chunk_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def test_paired_streams_never_cross_splits_and_seed_is_deterministic(tmp_path: Path) -> None:
    manifest = _write_pair_chunks(tmp_path)
    groups, recordings = group_chunks_by_pair(tmp_path, manifest)

    first = recording_level_split(groups, 0.7, 0.15, 0.15, 42)
    second = recording_level_split(groups, 0.7, 0.15, 0.15, 42)

    assert first == second
    train, validation, test = map(set, first)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert all(len(streams) == 2 for streams in recordings.values())


def test_split_integrity_report_is_machine_checkable(tmp_path: Path) -> None:
    groups = {
        "pair-train": {"first_crack": [Path("train.wav")]},
        "pair-validation": {"no_first_crack": [Path("validation.wav")]},
        "pair-test": {"first_crack": [Path("test.wav")]},
    }
    recordings = {
        "pair-train": {"train-mic1", "train-mic2"},
        "pair-validation": {"validation-mic1"},
        "pair-test": {"test-mic1", "test-mic2"},
    }

    generate_split_report(
        tmp_path,
        groups,
        ["pair-train"],
        ["pair-validation"],
        ["pair-test"],
        {"first_crack": 1},
        {"no_first_crack": 1},
        {"first_crack": 1},
        recordings,
    )

    integrity = json.loads((tmp_path / "split_integrity.json").read_text(encoding="utf-8"))
    assert integrity["integrity_passed"] is True
    assert integrity["physical_session_count"] == 3
    assert integrity["stream_recording_count"] == 5
    assert integrity["pair_id_intersections"] == {
        "train_test": [],
        "train_validation": [],
        "validation_test": [],
    }
