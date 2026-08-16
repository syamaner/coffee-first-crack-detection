"""Tests for uncertainty-aware chunking and physical-pair splitting."""

from __future__ import annotations

import hashlib
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
                    "derivation_method": "verified_audio_alignment",
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


def test_unaligned_derived_mic2_excludes_every_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical timestamps remain auditable but never become training ground truth."""
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
                    "alignment_uncertainty_seconds": None,
                    "alignment_uncertainty_status": (
                        "unbounded_historical_missing_stream_start_offsets"
                    ),
                    "training_policy": ("exclude_all_derived_mic2_without_verified_alignment"),
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

    assert counts["first_crack"] == 0
    assert counts["no_first_crack"] == 0
    assert counts["excluded_uncertain"] == len(records)
    assert all(record["included"] is False for record in records)
    assert {record["exclusion_reason"] for record in records} == {
        "derived_mic2_without_verified_alignment"
    }


def test_paired_mic2_without_derived_provenance_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed MCP mic2 annotation cannot fall through as exact legacy ground truth."""
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
                "annotations": [],
                "provenance": {"annotation_source": "misspelled"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chunking.librosa,
        "load",
        lambda *_args, **_kwargs: (np.zeros(20 * 100, dtype=np.float32), 100),
    )

    with pytest.raises(ValueError, match="Paired mic2 annotation lacks"):
        process_recording(
            annotation_path,
            audio_root,
            tmp_path / "processed",
            window_size=10.0,
            sample_rate=100,
        )


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        (
            {
                "annotation_source": "derived_from_paired_mic",
                "alignment_uncertainty_seconds": 1.0,
                "alignment_uncertainty_status": (
                    "unbounded_historical_missing_stream_start_offsets"
                ),
                "training_policy": "exclude_all_derived_mic2_without_verified_alignment",
            },
            "inconsistent uncertainty",
        ),
        (
            {
                "annotation_source": "derived_from_paired_mic",
                "derivation_method": "copy_timestamps_for_audit_only",
                "alignment_uncertainty_seconds": 1.0,
                "training_policy": "exclude_windows_intersecting_boundary_guard_band",
            },
            "lacks verified finite alignment uncertainty",
        ),
    ],
)
def test_inconsistent_derived_alignment_provenance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provenance: dict[str, object],
    message: str,
) -> None:
    """Neither a false finite bound nor an unverified guard band is accepted."""
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
                "annotations": [],
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chunking.librosa,
        "load",
        lambda *_args, **_kwargs: (np.zeros(20 * 100, dtype=np.float32), 100),
    )

    with pytest.raises(ValueError, match=message):
        process_recording(
            annotation_path,
            audio_root,
            tmp_path / "processed",
            window_size=10.0,
            sample_rate=100,
        )


def test_legacy_single_mic_chunk_keeps_deterministic_pair_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy annotations without pairing metadata remain supported."""
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "legacy.wav").write_bytes(b"stub")
    annotation_path = tmp_path / "legacy.json"
    annotation_path.write_text(
        json.dumps({"audio_file": "legacy.wav", "annotations": []}), encoding="utf-8"
    )
    monkeypatch.setattr(
        chunking.librosa,
        "load",
        lambda *_args, **_kwargs: (np.zeros(20 * 100, dtype=np.float32), 100),
    )
    monkeypatch.setattr(chunking, "save_chunk", lambda *_args, **_kwargs: None)
    records: list[dict[str, object]] = []

    process_recording(
        annotation_path,
        audio_root,
        tmp_path / "processed",
        window_size=10.0,
        sample_rate=100,
        chunk_manifest_records=records,
    )

    assert {record["pair_id"] for record in records} == {"single:legacy"}


def test_non_integer_annotation_mic_number_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed pairing metadata cannot be normalized implicitly."""
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "bad.wav").write_bytes(b"stub")
    annotation_path = tmp_path / "bad.json"
    annotation_path.write_text(
        json.dumps(
            {
                "audio_file": "bad.wav",
                "pair_id": "pair-1",
                "mic_num": "2",
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chunking.librosa,
        "load",
        lambda *_args, **_kwargs: (np.zeros(20 * 100, dtype=np.float32), 100),
    )

    with pytest.raises(ValueError, match="mic_num must be an integer"):
        process_recording(
            annotation_path,
            audio_root,
            tmp_path / "processed",
            window_size=10.0,
            sample_rate=100,
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


def test_partial_session_names_resolve_to_one_pair(tmp_path: Path) -> None:
    """Explicitly retained partial recordings cannot split by microphone."""
    sidecar = {
        "mics": [
            {"mic_num": 1, "file": "mic1-brazil-roast4_partial.wav"},
            {"mic_num": 2, "file": "mic2-brazil-roast4_partial.wav"},
        ]
    }
    (tmp_path / "brazil-roast4-session_partial.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    index = build_legacy_pair_index(tmp_path)

    assert index["mic1-brazil-roast4_partial.wav"][0] == "legacy:brazil-roast4"
    assert index["mic1-brazil-roast4_partial.wav"][0] == index["mic2-brazil-roast4_partial.wav"][0]


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
                    "source_audio_sha256": hashlib.sha256(
                        f"{pair_id}__mic{mic_num}".encode()
                    ).hexdigest(),
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


def test_pair_manifest_rejects_traversing_label(tmp_path: Path) -> None:
    """An untrusted manifest label cannot escape the input or split root."""
    manifest = tmp_path / "chunk_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "chunk_filename": "chunk.wav",
                "recording_id": "recording",
                "pair_id": "pair",
                "source_audio_sha256": "0" * 64,
                "label": "../escape",
                "included": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported chunk label"):
        group_chunks_by_pair(tmp_path, manifest)


def test_pair_manifest_rejects_source_checksum_across_pair_ids(tmp_path: Path) -> None:
    """Exact copied audio cannot be assigned independently through different pair IDs."""
    manifest = _write_pair_chunks(tmp_path, pair_count=2)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    first_hash = rows[0]["source_audio_sha256"]
    target_pair = rows[-1]["pair_id"]
    for row in rows:
        if row["pair_id"] == target_pair:
            row["source_audio_sha256"] = first_hash
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Source checksum is assigned to multiple pair IDs"):
        group_chunks_by_pair(tmp_path, manifest)


def test_pair_manifest_rejects_symlinked_chunk(tmp_path: Path) -> None:
    """A manifest cannot import a chunk through a symlink to another corpus."""
    external = tmp_path / "external.wav"
    external.write_bytes(b"wav")
    label_dir = tmp_path / "first_crack"
    label_dir.mkdir()
    (label_dir / "chunk.wav").symlink_to(external)
    manifest = tmp_path / "chunk_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "chunk_filename": "chunk.wav",
                "recording_id": "recording",
                "pair_id": "pair",
                "source_audio_sha256": "0" * 64,
                "label": "first_crack",
                "included": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="regular non-symlink"):
        group_chunks_by_pair(tmp_path, manifest)


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
