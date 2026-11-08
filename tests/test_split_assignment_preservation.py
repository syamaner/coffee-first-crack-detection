"""Synthetic behavioural contracts for preserving established pair membership."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from coffee_first_crack.data_prep import dataset_splitter as splitter
from coffee_first_crack.data_prep.output_paths import require_absent_output_dir


def prior_document() -> dict[str, Any]:
    """Return synthetic legacy pair memberships with deliberately stale counts."""
    return {
        "schema_version": 1,
        "strategy": "pair_id",
        "integrity_passed": True,
        "physical_session_count": 9000,
        "splits": {
            name: {"pair_ids": [pair], "stream_recording_count": 999}
            for name, pair in zip(
                ("train", "validation", "test"),
                ("legacy-train", "legacy-val", "legacy-test"),
                strict=True,
            )
        },
    }


def corpus(root: Path, *, reverse: bool = False, changed: bool = False) -> Path:
    """Create opaque synthetic chunk bytes and the actual manifest format."""
    root.mkdir()
    records = []
    for pair in ("legacy-train", "legacy-val", "legacy-test", "new-pair"):
        for mic in range(1, 4 if changed else 3):
            stream = f"{pair}-mic{mic}"
            label = "first_crack" if (mic == 1) != changed else "no_first_crack"
            filename = f"{stream}_w0000.0.wav"
            directory = root / label
            directory.mkdir(exist_ok=True)
            (directory / filename).write_bytes(stream.encode())
            records.append(
                {
                    "chunk_filename": filename,
                    "pair_id": pair,
                    "recording_id": stream,
                    "source_audio_sha256": hashlib.sha256(stream.encode()).hexdigest(),
                    "label": label,
                    "included": True,
                }
            )
    # Excluded streams retain source identity for an assigned physical pair.
    records.append(
        {
            "chunk_filename": "excluded.wav",
            "pair_id": "legacy-train",
            "recording_id": "excluded-stream",
            "source_audio_sha256": "a" * 64,
            "label": "first_crack",
            "included": False,
        }
    )
    (root / "chunk_manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in (records[::-1] if reverse else records)) + "\n"
    )
    return root


def invoke(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    output: Path,
    prior: Path | None,
    *extra: str,
) -> None:
    """Run the real CLI parser and orchestration."""
    argv = ["splitter", "--input", str(source), "--output", str(output)]
    if prior is not None:
        argv += ["--preserve-assignments", str(prior)]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    splitter.main()


def write_prior(root: Path, document: object | None = None) -> Path:
    """Write one synthetic prior report."""
    path = root / "prior.json"
    path.write_text(json.dumps(prior_document() if document is None else document))
    return path


def hashes(root: Path) -> dict[str, str]:
    """Capture exact file contents for preservation assertions."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


@pytest.mark.parametrize("changed", [False, True])
def test_membership_counts_and_determinism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    source = corpus(tmp_path / "input", reverse=changed, changed=changed)
    document = prior_document()
    document["splits"]["test"]["pair_ids"].append("missing-pair")
    prior = write_prior(tmp_path, document)
    before = hashes(source)
    outputs = [tmp_path / "out1", tmp_path / "out2"]
    for output, seed in zip(outputs, (1, 999), strict=True):
        invoke(monkeypatch, source, output, prior, "--seed", str(seed), "--train", "-9")
        integrity = json.loads((output / "split_integrity.json").read_text())
        report = json.loads((output / "split_assignment_report.json").read_text())
        assert report["schema_version"] == 1
        assert report["assignment_mode"] == "preserved"
        assert integrity["integrity_passed"] is True
        assert integrity["physical_session_count"] == 3
        assert integrity["stream_recording_count"] == (10 if changed else 7)
        assert all(value == [] for value in integrity["pair_id_intersections"].values())
        assert (
            report["prior_split_integrity_sha256"] == hashlib.sha256(prior.read_bytes()).hexdigest()
        )
        assert report["unassigned_pair_ids"] == ["new-pair"]
        assert report["unassigned_source_stream_ids"]["new-pair"] == [
            f"new-pair-mic{mic}" for mic in range(1, 4 if changed else 3)
        ]
        assert report["missing_prior_pair_ids"] == ["missing-pair"]
        assert report["fresh_holdout"] is False
        assert report["seed_and_ratios_used"] is False
        for name, pair in (
            ("train", "legacy-train"),
            ("validation", "legacy-val"),
            ("test", "legacy-test"),
        ):
            split_dir = output / ("val" if name == "validation" else name)
            copied = list(split_dir.rglob("*.wav"))
            assert len(copied) == (3 if changed else 2)
            assert all(path.read_bytes().decode().startswith(pair + "-mic") for path in copied)
            assert integrity["splits"][name]["pair_ids"] == [pair]
            assert report["splits"][name]["chunk_count"] == len(copied)
            expected_labels = {"first_crack": 2 if changed else 1, "no_first_crack": 1}
            expected_streams = [f"{pair}-mic{mic}" for mic in range(1, 4 if changed else 3)]
            if name == "train":
                expected_streams.append("excluded-stream")
            expected_streams.sort()
            assert integrity["splits"][name] == {
                "pair_ids": [pair],
                "physical_session_count": 1,
                "stream_recording_count": len(expected_streams),
            }
            assert report["splits"][name] == {
                "pair_ids": [pair],
                "physical_session_count": 1,
                "stream_recording_count": len(expected_streams),
                "chunk_count": len(copied),
                "label_counts": expected_labels,
            }
            for label, count in expected_labels.items():
                assert len(list((split_dir / label).glob("*.wav"))) == count
            markdown = (output / "split_report.md").read_text()
            display_name = {"train": "Train", "validation": "Val", "test": "Test"}[name]
            assert (
                f"| {display_name} | {expected_labels['first_crack']} | 1 | {len(copied)} | 33.3% |"
                in markdown
            )
            assert (
                f"- {pair}: {len(expected_streams)} stream(s) [{', '.join(expected_streams)}]; "
                f"{expected_labels['first_crack']} first_crack, 1 no_first_crack"
            ) in markdown
        markdown = (output / "split_report.md").read_text()
        assert f"**Total chunks**: {9 if changed else 6}" in markdown
        assert "new-pair" not in markdown
        assert "excluded-stream" in markdown
        assert not any("new-pair" in str(path.relative_to(output)) for path in output.rglob("*"))
        assert all(not path.read_bytes().startswith(b"new-pair") for path in output.rglob("*.wav"))
    assert hashes(outputs[0]) == hashes(outputs[1])
    assert hashes(source) == before


def test_all_unassigned_and_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = corpus(tmp_path / "input")
    document = prior_document()
    for split in document["splits"].values():
        split["pair_ids"] = ["missing-" + split["pair_ids"][0]]
    prior = write_prior(tmp_path, document)
    for index in range(2):
        output = tmp_path / f"out{index}"
        invoke(monkeypatch, source, output, prior)
        integrity = json.loads((output / "split_integrity.json").read_text())
        report = json.loads((output / "split_assignment_report.json").read_text())
        assert integrity["physical_session_count"] == integrity["stream_recording_count"] == 0
        assert not list(output.rglob("*.wav"))
        assert all(split["chunk_count"] == 0 for split in report["splits"].values())
        markdown = (output / "split_report.md").read_text()
        assert "**Total chunks**: 0" in markdown
        assert "**Total streams / recordings**: 0" in markdown
        assert report["missing_prior_pair_ids"] == [
            "missing-legacy-test",
            "missing-legacy-train",
            "missing-legacy-val",
        ]
        assert report["unassigned_pair_ids"] == (
            ["legacy-test", "legacy-train", "legacy-val", "new-pair"] if index == 0 else []
        )
        assert report["unassigned_source_stream_ids"] == (
            {
                pair: sorted(
                    [f"{pair}-mic1", f"{pair}-mic2"]
                    + (["excluded-stream"] if pair == "legacy-train" else [])
                )
                for pair in ["legacy-test", "legacy-train", "legacy-val", "new-pair"]
            }
            if index == 0
            else {}
        )
        for name in ("train", "validation", "test"):
            assert integrity["splits"][name] == {
                "pair_ids": [],
                "physical_session_count": 0,
                "stream_recording_count": 0,
            }
            assert report["splits"][name] == {
                "pair_ids": [],
                "physical_session_count": 0,
                "stream_recording_count": 0,
                "chunk_count": 0,
                "label_counts": {"first_crack": 0, "no_first_crack": 0},
            }
        for name in ("Train", "Val", "Test"):
            assert f"| {name} | 0 | 0 | 0 | 0.0% |" in markdown
        assert not any(
            line.startswith("- legacy-") or line.startswith("- missing-")
            for line in markdown.splitlines()
        )
        if index == 0:
            for path in source.rglob("*.wav"):
                path.unlink()
            (source / "chunk_manifest.jsonl").write_text("")


def malformed_documents() -> list[object]:
    """Enumerate structurally invalid historical membership declarations."""
    documents: list[object] = [[], "text", 1, {}]
    for key, values in {
        "schema_version": [None, True, 1.0, "1", 2],
        "strategy": [None, "recording"],
        "integrity_passed": [None, False, 1, "true"],
        "splits": [None, [], {}, {"train": {}, "validation": {}, "test": {}, "extra": {}}],
    }.items():
        for value in values:
            document = prior_document()
            document[key] = value
            documents.append(document)
    for value in [None, {}, "pair", [1], [None], [""], ["x", "x"], ["legacy-val"]]:
        document = prior_document()
        document["splits"]["train"]["pair_ids"] = value
        documents.append(document)
    document = prior_document()
    del document["splits"]["validation"]
    documents.append(document)
    for key in ("schema_version", "strategy", "integrity_passed", "splits"):
        document = prior_document()
        del document[key]
        documents.append(document)
    for value in [None, {}, {"pair_ids": None}]:
        document = prior_document()
        document["splits"]["train"] = value
        documents.append(document)
    return documents


@pytest.mark.parametrize("document", malformed_documents())
def test_invalid_prior_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: object,
) -> None:
    source = corpus(tmp_path / "input")
    before = hashes(source)
    prior = write_prior(tmp_path, document)
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, output, prior)
    assert error.value.code == 1
    assert not output.exists()
    assert hashes(source) == before


@pytest.mark.parametrize("kind", ["empty", "nonempty", "file", "link", "dangling", "fifo"])
def test_existing_output_entries_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    output = tmp_path / "output"
    target = tmp_path / "target"
    target.write_bytes(b"target sentinel")
    if kind in {"empty", "nonempty"}:
        output.mkdir()
        if kind == "nonempty":
            (output / "sentinel").write_bytes(b"keep")
    elif kind == "file":
        output.write_bytes(b"keep")
    elif kind in {"link", "dangling"}:
        output.symlink_to(target if kind == "link" else tmp_path / "absent")
    else:
        os.mkfifo(output)
    before = hashes(source)
    identity = output.lstat()
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, output, prior)
    assert output.lstat() == identity
    assert target.read_bytes() == b"target sentinel"
    if kind == "nonempty":
        assert (output / "sentinel").read_bytes() == b"keep"
    if kind == "file":
        assert output.read_bytes() == b"keep"
    assert hashes(source) == before


def test_missing_parent_nested_and_bad_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    for output in (tmp_path / "missing" / "out", source / "out"):
        with pytest.raises(SystemExit):
            invoke(monkeypatch, source, output, prior)
        assert not output.exists()
    assert not (tmp_path / "missing").exists()
    (source / "first_crack" / "undeclared.wav").write_bytes(b"synthetic")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "malformed"])
def test_prior_file_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    source = corpus(tmp_path / "input")
    prior = tmp_path / "prior"
    if kind == "symlink":
        prior.symlink_to(write_prior(tmp_path))
    elif kind == "directory":
        prior.mkdir()
    elif kind == "fifo":
        os.mkfifo(prior)
    else:
        prior.write_bytes(b"{")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert not (tmp_path / "out").exists()


def test_creation_race_not_adopted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    output = tmp_path / "output"
    original = os.mkdir
    raced = False

    def race(path: str | Path, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        nonlocal raced
        if path == output.name and dir_fd is not None and not raced:
            raced = True
            original(path, mode, dir_fd=dir_fd)
            (output / "sentinel").write_bytes(b"raced")
        original(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", race)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, output, prior)
    assert sorted(path.name for path in output.iterdir()) == ["sentinel"]
    assert (output / "sentinel").read_bytes() == b"raced"


@pytest.mark.parametrize(
    "destination",
    ["wav", "split_report.md", "split_integrity.json", "split_assignment_report.json"],
)
def test_preplanted_destination_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    output = tmp_path / "output"
    original = os.open
    planted: list[str] = []

    def plant(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        name = str(path)
        if (
            flags & os.O_CREAT
            and not planted
            and (name.endswith(".wav") if destination == "wav" else name == destination)
        ):
            descriptor = original(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=dir_fd)
            try:
                os.write(descriptor, b"sentinel")
            finally:
                os.close(descriptor)
            planted.append(name)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", plant)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, output, prior)
    matches = list(output.rglob(planted[0]))
    assert len(matches) == 1 and matches[0].read_bytes() == b"sentinel"
    if destination in {"wav", "split_assignment_report.json"}:
        assert not (output / "split_integrity.json").exists()


def test_copy_failure_has_no_success_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    before = hashes(source)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(splitter.shutil, "copyfileobj", fail)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    for name in ("split_report.md", "split_integrity.json", "split_assignment_report.json"):
        assert not (tmp_path / "out" / name).exists()
    assert hashes(source) == before


def test_never_uses_destructive_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("legacy destructive copy invoked")

    monkeypatch.setattr(splitter, "copy_chunks", forbidden)
    monkeypatch.setattr(splitter.shutil, "rmtree", forbidden)
    invoke(monkeypatch, source, tmp_path / "out", prior)


def test_lstat_errors_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(path: Path) -> os.stat_result:
        raise PermissionError("synthetic permission error")

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError):
        require_absent_output_dir(tmp_path / "out")


def test_non_directory_parent(tmp_path: Path) -> None:
    parent = tmp_path / "file"
    parent.write_bytes(b"keep")
    with pytest.raises(NotADirectoryError):
        require_absent_output_dir(parent / "out")
    assert parent.read_bytes() == b"keep"


def test_real_cli_failure_is_nonzero(tmp_path: Path) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path, {})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            splitter.__name__,
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out"),
            "--preserve-assignments",
            str(prior),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Preservation failed" in result.stderr
    assert not (tmp_path / "out").exists()


def test_default_seeded_bytes_match_bound_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the unflagged base implementation's synthetic seeded report bytes."""
    source = corpus(tmp_path / "input")
    output = tmp_path / "out"
    invoke(
        monkeypatch,
        source,
        output,
        None,
        "--train",
        "0.5",
        "--val",
        "0.25",
        "--test",
        "0.25",
        "--seed",
        "42",
    )
    assert hashlib.sha256((output / "split_integrity.json").read_bytes()).hexdigest() == (
        "080c482623466f52cfd713f5ade02bf4be5e379379e7929538038aca507db055"
    )
    assert hashlib.sha256((output / "split_report.md").read_bytes()).hexdigest() == (
        "6775d113363200582d2198b880d03f25626428c804bed5b1ef6dcef2ddff6957"
    )
    integrity = json.loads((output / "split_integrity.json").read_text())
    assert integrity["splits"]["train"]["pair_ids"] == ["legacy-train", "new-pair"]
    assert integrity["splits"]["validation"]["pair_ids"] == ["legacy-test"]
    assert integrity["splits"]["test"]["pair_ids"] == ["legacy-val"]
    assert not (output / "split_assignment_report.json").exists()


def test_parent_changed_after_absence_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a parent replaced with a file after the absence observation."""
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "out"
    original = Path.lstat

    def replace_parent(path: Path) -> os.stat_result:
        if path == output:
            parent.rmdir()
            parent.write_bytes(b"raced parent")
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "lstat", replace_parent)
    with pytest.raises(NotADirectoryError):
        require_absent_output_dir(output)
    assert parent.read_bytes() == b"raced parent"


def test_reordered_manifest_report_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    invoke(monkeypatch, source, tmp_path / "out1", prior)
    manifest = source / "chunk_manifest.jsonl"
    manifest.write_text("\n".join(reversed(manifest.read_text().splitlines())) + "\n")
    invoke(monkeypatch, source, tmp_path / "out2", prior, "--seed", "987", "--test", "9")
    assert hashes(tmp_path / "out1") == hashes(tmp_path / "out2")


def test_json_null_rejected_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = tmp_path / "prior.json"
    prior.write_text("null")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("split_name", "pair"),
    [("train", "legacy-train"), ("validation", "legacy-val"), ("test", "legacy-test")],
)
def test_preserved_source_streams_match_existing_report_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split_name: str,
    pair: str,
) -> None:
    """Keep excluded mic2 exposure identity while copying only usable mic1."""
    from scripts.evaluate_mcp_heldout import _split_recording_ids, discover_heldout_recordings

    source = tmp_path / "input"
    label_dir = source / "first_crack"
    label_dir.mkdir(parents=True)
    mic1_bytes = b"synthetic included mic1"
    mic2_hash = hashlib.sha256(b"synthetic excluded mic2").hexdigest()
    mic1 = label_dir / "mic1_w0000.0.wav"
    mic1.write_bytes(mic1_bytes)
    records = [
        {
            "chunk_filename": filename,
            "pair_id": pair,
            "recording_id": stream,
            "source_audio_sha256": digest,
            "label": "first_crack",
            "included": included,
        }
        for filename, stream, digest, included in (
            (mic1.name, "mic1", hashlib.sha256(mic1_bytes).hexdigest(), True),
            ("mic2_w0000.0.wav", "mic2", mic2_hash, False),
        )
    ]
    manifest = source / "chunk_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    before = hashes(source)
    prior = write_prior(tmp_path)
    output = tmp_path / "out"
    invoke(monkeypatch, source, output, prior)
    pair_ids, identities, source_hashes = _split_recording_ids(
        output / "split_integrity.json",
        manifest,
    )
    assert pair_ids == {pair}
    assert identities == {(pair, "mic1"), (pair, "mic2")}
    assert source_hashes == {record["source_audio_sha256"] for record in records}
    report = json.loads((output / "split_assignment_report.json").read_text())
    integrity = json.loads((output / "split_integrity.json").read_text())
    assert integrity["stream_recording_count"] == 2
    assert integrity["splits"][split_name]["stream_recording_count"] == 2
    assert report["splits"][split_name]["stream_recording_count"] == 2
    assert report["splits"][split_name]["chunk_count"] == 1
    copied = list(output.rglob("*.wav"))
    assert len(copied) == 1
    assert copied[0].read_bytes() == mic1_bytes
    assert hashes(source) == before

    assert integrity["splits"][split_name]["pair_ids"] == [pair]
    assert report["fresh_holdout"] is False
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text(json.dumps({"sessions": []}))
    holdout_manifest = tmp_path / "holdout_manifest.json"
    holdout_manifest.write_text(
        json.dumps(
            {
                "source_root": str(source.resolve()),
                "sessions": [{"pair_id": pair}],
            }
        )
    )
    with pytest.raises(ValueError, match="already appears in a dataset split"):
        discover_heldout_recordings(
            split_integrity_path=output / "split_integrity.json",
            chunk_manifest_path=manifest,
            dataset_capture_manifest_path=dataset_manifest,
            holdout_capture_manifest_path=holdout_manifest,
            label_dirs=(),
            pair_ids={pair},
            window_seconds=10.0,
        )


def test_prior_size_limit_precedes_read_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a sparse oversized report by size, before any unbounded read."""
    assert splitter._MAX_PRIOR_BYTES == 16 * 1024 * 1024
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    with prior.open("r+b") as handle:
        handle.truncate(splitter._MAX_PRIOR_BYTES + 1)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert error.value.code == 1
    assert "exceeds 16 MiB maximum" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("grow", [False, True])
def test_prior_read_bound_and_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    grow: bool,
) -> None:
    """Inspect actual read size; growth remains valid JSON and must hit size guard."""
    from typing import BinaryIO, TextIO

    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    original_payload = prior.read_bytes()
    cap = 4096
    monkeypatch.setattr(splitter, "_MAX_PRIOR_BYTES", cap)
    original_fdopen = splitter.os.fdopen
    original_fstat = splitter.os.fstat
    prior_identity = prior.stat()
    read_sizes: list[int] = []
    returned_sizes: list[int] = []

    class ReadProbe:
        """Observe the production descriptor reader without replacing its bytes."""

        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def __enter__(self) -> ReadProbe:
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            result = self.handle.read(size)
            returned_sizes.append(len(result))
            return result

    def observe(
        descriptor: int,
        mode: str,
        *,
        closefd: bool = True,
        encoding: str | None = None,
    ) -> ReadProbe | BinaryIO | TextIO:
        metadata = original_fstat(descriptor)
        handle = original_fdopen(descriptor, mode, closefd=closefd, encoding=encoding)
        if (metadata.st_dev, metadata.st_ino) == (prior_identity.st_dev, prior_identity.st_ino):
            return ReadProbe(handle)
        return handle

    def inspect_then_grow(descriptor: int) -> os.stat_result:
        metadata = original_fstat(descriptor)
        if grow and (metadata.st_dev, metadata.st_ino) == (
            prior_identity.st_dev,
            prior_identity.st_ino,
        ):
            with prior.open("ab") as writer:
                writer.write(b" " * (cap + 1))
        return metadata

    monkeypatch.setattr(splitter.os, "fdopen", observe)
    monkeypatch.setattr(splitter.os, "fstat", inspect_then_grow)
    if grow:
        with pytest.raises(SystemExit) as error:
            invoke(monkeypatch, source, tmp_path / "out", prior)
        assert error.value.code == 1
        assert "exceeds 16 MiB maximum" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()
        assert returned_sizes == [cap + 1]
        assert json.loads(prior.read_bytes()) == json.loads(original_payload)
    else:
        invoke(monkeypatch, source, tmp_path / "out", prior)
        report = json.loads((tmp_path / "out" / "split_assignment_report.json").read_text())
        assert (
            report["prior_split_integrity_sha256"] == hashlib.sha256(original_payload).hexdigest()
        )
        assert returned_sizes == [len(original_payload)]
    assert read_sizes == [cap + 1]


@pytest.mark.parametrize("swap", ["file-link", "label-link", "fifo", "directory"])
def test_source_swap_fails_before_outside_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    """Reject link/component swaps and nonregular sources after real validation."""
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    output = tmp_path / "out"
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = b"OUTSIDE SYNTHETIC MARKER - NEVER COPY"
    for path in (source / "first_crack").glob("*.wav"):
        (outside / path.name).write_bytes(marker)
    before = hashes(outside)
    original_group = splitter.group_chunks_by_pair
    original_open = splitter.os.open
    descriptors: list[int] = []

    def track_open(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def validate_then_swap(
        input_dir: Path,
        manifest: Path,
        *,
        _bound_input: splitter._BoundChunkInput | None = None,
    ) -> tuple[dict[str, dict[str, list[Path]]], dict[str, set[str]]]:
        result = original_group(input_dir, manifest, _bound_input=_bound_input)
        label = source / "first_crack"
        victim = label / "legacy-train-mic1_w0000.0.wav"
        if swap == "label-link":
            label.rename(source / "original-label")
            label.symlink_to(outside, target_is_directory=True)
        else:
            victim.rename(source / "original-chunk")
            if swap == "file-link":
                victim.symlink_to(outside / victim.name)
            elif swap == "fifo":
                os.mkfifo(victim)
            else:
                victim.mkdir()
        return result

    monkeypatch.setattr(splitter.os, "open", track_open)
    monkeypatch.setattr(splitter, "group_chunks_by_pair", validate_then_swap)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, output, prior)
    assert error.value.code == 1
    assert hashes(outside) == before
    assert all(marker not in path.read_bytes() for path in output.rglob("*.wav"))
    for name in ("split_report.md", "split_integrity.json", "split_assignment_report.json"):
        assert not (output / name).exists()
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def tracked_descriptors(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Observe descriptor acquisition without replacing filesystem behaviour."""
    opened: list[int] = []
    original_open, original_dup = os.open, os.dup

    def track(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def duplicate(descriptor: int) -> int:
        result = original_dup(descriptor)
        opened.append(result)
        return result

    monkeypatch.setattr(os, "open", track)
    monkeypatch.setattr(os, "dup", duplicate)
    return opened


def assert_descriptors_closed(opened: list[int]) -> None:
    """Prove every observed descriptor has been closed after orchestration."""
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_input_root_swap_keeps_manifest_validation_and_copy_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    original_root = tmp_path / "original-root"
    original = os.open
    swapped = False
    before = hashes(source)

    def swap_after_open(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = original(path, flags, *args, **kwargs)
        if path == source and flags & os.O_DIRECTORY and not swapped:
            swapped = True
            source.rename(original_root)
            corpus(source)
            manifest = source / "chunk_manifest.jsonl"
            manifest.write_text(manifest.read_text().replace("legacy-", "replacement-"))
            for chunk in source.rglob("*.wav"):
                chunk.write_bytes(b"REPLACEMENT ROOT MARKER")
        return descriptor

    monkeypatch.setattr(os, "open", swap_after_open)
    opened = tracked_descriptors(monkeypatch)
    output = tmp_path / "out"
    invoke(monkeypatch, source, output, prior)
    assert swapped
    assert hashes(original_root) == before
    integrity = json.loads((output / "split_integrity.json").read_text())
    for name, pair in (
        ("train", "legacy-train"),
        ("validation", "legacy-val"),
        ("test", "legacy-test"),
    ):
        assert integrity["splits"][name]["pair_ids"] == [pair]
    assert all(b"REPLACEMENT ROOT" not in path.read_bytes() for path in output.rglob("*.wav"))
    assert len(list(output.rglob("*.wav"))) == 6
    assert_descriptors_closed(opened)


@pytest.mark.parametrize(
    "when", ["preflight", "grouping", "copy", "subdirectory", "report", "integrity"]
)
def test_output_binding_swaps_never_write_into_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    when: str,
) -> None:
    source = corpus(tmp_path / "input")
    before = hashes(source)
    prior = write_prior(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "out"
    saved_parent = tmp_path / "saved-parent"
    saved_output = tmp_path / "saved-output"
    swapped = False

    def replace_parent() -> None:
        nonlocal swapped
        if not swapped:
            parent.rename(saved_parent)
            parent.symlink_to(source, target_is_directory=True)
            swapped = True

    original_preflight = splitter.require_absent_output_dir
    original_group = splitter.group_chunks_by_pair
    original_copy = splitter.shutil.copyfileobj
    original_file = splitter._exclusive_output_file

    def preflight(path: Path) -> None:
        original_preflight(path)
        if when == "preflight":
            replace_parent()

    def grouping(
        input_dir: Path,
        manifest: Path,
        *,
        _bound_input: splitter._BoundChunkInput | None = None,
    ) -> tuple[dict[str, dict[str, list[Path]]], dict[str, set[str]]]:
        result = original_group(input_dir, manifest, _bound_input=_bound_input)
        if when == "grouping":
            replace_parent()
        return result

    def copy(reader: object, writer: object) -> None:
        nonlocal swapped
        if not swapped and when == "copy":
            replace_parent()
        elif not swapped and when == "subdirectory":
            directory = output / "train" / "first_crack"
            directory.rename(output / "train" / "original-label")
            directory.symlink_to(source / "first_crack", target_is_directory=True)
            swapped = True
        original_copy(reader, writer)

    def report_file(directory: int, name: str, handles: object) -> int:
        nonlocal swapped
        if (
            (when == "report" and name == "split_assignment_report.json")
            or (when == "integrity" and name == "split_integrity.json")
        ) and not swapped:
            output.rename(saved_output)
            output.symlink_to(source, target_is_directory=True)
            swapped = True
        return original_file(directory, name, handles)

    monkeypatch.setattr(splitter, "require_absent_output_dir", preflight)
    monkeypatch.setattr(splitter, "group_chunks_by_pair", grouping)
    monkeypatch.setattr(splitter.shutil, "copyfileobj", copy)
    monkeypatch.setattr(splitter, "_exclusive_output_file", report_file)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, output, prior)
    assert error.value.code == 1 and swapped
    assert hashes(source) == before
    assert not (source / "out").exists()
    for candidate in (output, saved_output, saved_parent / "out"):
        marker = candidate / "split_integrity.json"
        assert not marker.exists() or marker.read_bytes() == b""
    assert_descriptors_closed(opened)


@pytest.mark.parametrize("alias_kind", ["case", "private-prefix", "firmlink"])
def test_output_alias_inside_input_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    source = corpus(tmp_path / "CaseInput")
    prior = write_prior(tmp_path)
    if alias_kind == "case":
        alias = source.with_name("caseinput")
    elif alias_kind == "private-prefix":
        alias = Path(str(source).removeprefix("/private"))
    else:
        alias = Path("/System/Volumes/Data") / source.relative_to("/")
    if alias == source or not alias.exists() or not alias.samefile(source):
        pytest.skip(f"This filesystem does not provide the {alias_kind} alias")
    before = hashes(source)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, alias / "nested", prior)
    assert not (source / "nested").exists()
    assert hashes(source) == before
    assert_descriptors_closed(opened)


def test_parent_disappearance_at_exclusive_creation_is_not_recreated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    original = os.mkdir
    removed = False

    def disappear(path: str | Path, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        nonlocal removed
        if path == "out" and dir_fd is not None and not removed:
            parent.rmdir()
            removed = True
        original(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", disappear)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, parent / "out", prior)
    assert removed and not parent.exists()
    assert_descriptors_closed(opened)


@pytest.mark.parametrize("level", ["root", "split", "pair_ids"])
def test_duplicate_prior_object_keys_fail_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    level: str,
) -> None:
    source = corpus(tmp_path / "input")
    document = prior_document()
    raw = json.dumps(document, separators=(",", ":"))
    if level == "root":
        old, new = '"schema_version":1', '"schema_version":1,"schema_version":1'
    elif level == "split":
        value = json.dumps(document["splits"]["train"], separators=(",", ":"))
        old, new = f'"train":{value}', f'"train":{value},"train":{value}'
    else:
        old, new = (
            '"pair_ids":["legacy-train"]',
            '"pair_ids":["legacy-train"],"pair_ids":["legacy-train"]',
        )
    assert old in raw
    prior = tmp_path / "prior.json"
    prior.write_text(raw.replace(old, new, 1))
    before = hashes(source)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert "Duplicate prior JSON object key" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert hashes(source) == before


def test_assignment_sidecar_failure_never_publishes_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    original = splitter._exclusive_output_file

    def fail_sidecar(directory: int, name: str, handles: object) -> int:
        if name == "split_assignment_report.json":
            raise OSError("synthetic sidecar write failure")
        return original(directory, name, handles)

    monkeypatch.setattr(splitter, "_exclusive_output_file", fail_sidecar)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert not (tmp_path / "out" / "split_integrity.json").exists()
    assert_descriptors_closed(opened)


def test_second_default_fixture_report_bytes_match_bound_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin independently captured base749c bytes for reordered three-mic input."""
    source = corpus(tmp_path / "input", changed=True, reverse=True)
    output = tmp_path / "out"
    invoke(
        monkeypatch,
        source,
        output,
        None,
        "--train",
        "0.5",
        "--val",
        "0.25",
        "--test",
        "0.25",
        "--seed",
        "123",
    )
    assert hashlib.sha256((output / "split_report.md").read_bytes()).hexdigest() == (
        "42b2f80c097606a3a50124d6831a2cdb541eb9e5d02ea0982d5ec96622d44598"
    )
    assert hashlib.sha256((output / "split_integrity.json").read_bytes()).hexdigest() == (
        "e91e848e175213b6224aaa172feae79fe9802a266c910f4612c79e258de66427"
    )
    assert not (output / "split_assignment_report.json").exists()


@pytest.mark.parametrize("kind", ["external", "nested", "alias", "directory", "fifo"])
def test_bound_manifest_locations_and_regular_file_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source = corpus(tmp_path / "CaseInput")
    prior = write_prior(tmp_path)
    payload = (source / "chunk_manifest.jsonl").read_bytes()
    hidden = source / ".metadata"
    hidden.mkdir()
    (hidden / "ignored.wav").write_bytes(b"hidden synthetic chunk")
    if kind == "nested":
        manifest = hidden / "chunks.jsonl"
        manifest.write_bytes(payload)
    elif kind == "alias":
        alias = source.with_name("caseinput")
        if not alias.exists() or not alias.samefile(source):
            pytest.skip("This filesystem has no case alias")
        manifest = alias / "chunk_manifest.jsonl"
    else:
        manifest = tmp_path / "external-manifest"
        if kind == "directory":
            manifest.mkdir()
        elif kind == "fifo":
            os.mkfifo(manifest)
        else:
            manifest.write_bytes(payload)
    opened = tracked_descriptors(monkeypatch)
    if kind in {"directory", "fifo"}:
        with pytest.raises(SystemExit):
            invoke(monkeypatch, source, tmp_path / "out", prior, "--chunk-manifest", str(manifest))
        assert not (tmp_path / "out").exists()
    else:
        invoke(monkeypatch, source, tmp_path / "out", prior, "--chunk-manifest", str(manifest))
        assert len(list((tmp_path / "out").rglob("*.wav"))) == 6
        assert not any(
            path.read_bytes().startswith(b"hidden") for path in (tmp_path / "out").rglob("*.wav")
        )
    assert_descriptors_closed(opened)


def test_extra_input_link_is_rejected_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "unrelated-link").symlink_to(outside, target_is_directory=True)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, tmp_path / "out", prior)
    assert not (tmp_path / "out").exists()
    assert_descriptors_closed(opened)


def test_legacy_grouping_requires_its_manifest(tmp_path: Path) -> None:
    """The unflagged identity authority refuses a missing manifest before reading chunks."""
    source = corpus(tmp_path / "input")
    manifest = source / "chunk_manifest.jsonl"
    manifest.unlink()
    before = hashes(source)
    with pytest.raises(FileNotFoundError, match="Pair-aware chunk manifest is required"):
        splitter.group_chunks_by_pair(source, manifest)
    assert hashes(source) == before


@pytest.mark.parametrize("kind", ["missing", "file", "symlink"])
def test_legacy_grouping_rejects_nonregular_label_directory(
    tmp_path: Path,
    kind: str,
) -> None:
    """The unchanged legacy path refuses absent, non-directory and linked labels."""
    source = corpus(tmp_path / "input")
    label = source / "first_crack"
    retained = source / "retained-label"
    label.rename(retained)
    if kind == "file":
        label.write_bytes(b"synthetic non-directory sentinel")
    elif kind == "symlink":
        label.symlink_to(retained, target_is_directory=True)
    before = hashes(source)
    with pytest.raises(ValueError, match="Chunk label directory must be a regular directory"):
        splitter.group_chunks_by_pair(source, source / "chunk_manifest.jsonl")
    assert hashes(source) == before


def non_ascii_corpus(root: Path) -> Path:
    """Add non-ASCII identities without changing source chunk bytes."""
    source = corpus(root)
    manifest = source / "chunk_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["pair_id"] += "-café"
        row["recording_id"] += "-café"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return source


def test_manifest_equal_to_input_is_a_regular_file_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    before = hashes(source)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, tmp_path / "out", prior, "--chunk-manifest", str(source))
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "Preservation failed" in message and "regular non-symlink file" in message
    assert not (tmp_path / "out").exists()
    assert hashes(source) == before
    assert_descriptors_closed(opened)


@pytest.mark.parametrize("encoding", ["latin-1", "ascii"])
def test_default_non_ascii_encoding_matches_actual_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    """Match base749c's independently captured non-UTF8 default writing contract."""
    import io

    source = non_ascii_corpus(tmp_path / "input")
    output = tmp_path / "out"
    before = hashes(source)
    original = io.text_encoding

    def controlled_default(value: str | None, stacklevel: int = 2) -> str:
        return encoding if value is None or value == "locale" else original(value, stacklevel)

    monkeypatch.setattr(io, "text_encoding", controlled_default)
    arguments = ("--train", "0.5", "--val", "0.25", "--test", "0.25", "--seed", "42")
    if encoding == "ascii":
        with pytest.raises(UnicodeEncodeError):
            invoke(monkeypatch, source, output, None, *arguments)
        assert not (output / "split_integrity.json").exists()
    else:
        invoke(monkeypatch, source, output, None, *arguments)
        markdown = (output / "split_report.md").read_bytes()
        assert b"caf\xe9" in markdown and b"caf\xc3\xa9" not in markdown
        assert (
            hashlib.sha256(markdown).hexdigest()
            == "88a8da9d84a6397763df750ad5727f59761aba6405725c262bfae94b50208303"
        )
        assert (
            hashlib.sha256((output / "split_integrity.json").read_bytes()).hexdigest()
            == "4ef36641cdf15c04450ee12d5595d364c0d0a4050ed902b3f1caca45952b0be6"
        )
    assert hashes(source) == before


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_different_regular_source_identity_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    """Reject different regular inodes after validation, before copying their bytes."""
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    original = splitter.group_chunks_by_pair
    marker = b"DIFFERENT REGULAR INODE REPLACEMENT"
    replaced: list[tuple[int, int]] = []

    def replace(
        input_dir: Path,
        manifest: Path,
        *,
        _bound_input: splitter._BoundChunkInput | None = None,
    ) -> tuple[dict[str, dict[str, list[Path]]], dict[str, set[str]]]:
        result = original(input_dir, manifest, _bound_input=_bound_input)
        label = source / "first_crack"
        victim = label / "legacy-train-mic1_w0000.0.wav"
        previous = victim.stat()
        if kind == "directory":
            label.rename(source / "retained-label")
            label.mkdir()
            for path in (source / "retained-label").iterdir():
                (label / path.name).write_bytes(marker)
        else:
            victim.rename(source / "retained-chunk")
            victim.write_bytes(marker)
        current = victim.stat()
        assert (previous.st_dev, previous.st_ino) != (current.st_dev, current.st_ino)
        replaced.append((current.st_dev, current.st_ino))
        return result

    monkeypatch.setattr(splitter, "group_chunks_by_pair", replace)
    opened = tracked_descriptors(monkeypatch)
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, output, prior)
    assert error.value.code == 1 and replaced
    assert "Validated chunk filesystem identity changed" in capsys.readouterr().err
    assert all(marker not in path.read_bytes() for path in output.rglob("*.wav"))
    for name in ("split_report.md", "split_integrity.json", "split_assignment_report.json"):
        assert not (output / name).exists()
    assert_descriptors_closed(opened)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_same_inode_link_targets_still_require_no_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """No-follow policy rejects links even when hardlinked target inodes match."""
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    for path in (source / "first_crack").iterdir():
        os.link(path, outside / path.name)
    outside_before = hashes(outside)
    original = splitter.group_chunks_by_pair

    def replace(
        input_dir: Path,
        manifest: Path,
        *,
        _bound_input: splitter._BoundChunkInput | None = None,
    ) -> tuple[dict[str, dict[str, list[Path]]], dict[str, set[str]]]:
        result = original(input_dir, manifest, _bound_input=_bound_input)
        label = source / "first_crack"
        victim = label / "legacy-train-mic1_w0000.0.wav"
        expected = victim.stat()
        assert expected.st_ino == (outside / victim.name).stat().st_ino
        if kind == "directory":
            label.rename(source / "retained-label")
            label.symlink_to(outside, target_is_directory=True)
        else:
            victim.rename(source / "retained-chunk")
            victim.symlink_to(outside / victim.name)
        return result

    monkeypatch.setattr(splitter, "group_chunks_by_pair", replace)
    opened = tracked_descriptors(monkeypatch)
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, output, prior)
    assert hashes(outside) == outside_before
    for name in ("split_report.md", "split_integrity.json", "split_assignment_report.json"):
        assert not (output / name).exists()
    assert_descriptors_closed(opened)


def test_preserved_non_ascii_reports_keep_explicit_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preservation callback stays UTF8 even when legacy defaults are ASCII."""
    import io

    source = non_ascii_corpus(tmp_path / "input")
    document = prior_document()
    for split in document["splits"].values():
        split["pair_ids"] = [pair + "-café" for pair in split["pair_ids"]]
    prior = write_prior(tmp_path, document)
    original = io.text_encoding

    def controlled_default(value: str | None, stacklevel: int = 2) -> str:
        return "ascii" if value is None or value == "locale" else original(value, stacklevel)

    monkeypatch.setattr(io, "text_encoding", controlled_default)
    output = tmp_path / "out"
    invoke(monkeypatch, source, output, prior)
    markdown = (output / "split_report.md").read_bytes()
    assert b"caf\xc3\xa9" in markdown
    assert json.loads((output / "split_integrity.json").read_bytes())["integrity_passed"] is True
    assert len(list(output.rglob("*.wav"))) == 6


@pytest.mark.parametrize("placement", ["parent", "nested-parent"])
def test_preserved_manifest_parent_components_rejected_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    placement: str,
) -> None:
    """A real valid manifest must not escape and reenter the bound input root."""
    source = corpus(tmp_path / "input")
    prior = write_prior(tmp_path)
    manifest = (
        source / ".." / "input" / "chunk_manifest.jsonl"
        if placement == "parent"
        else source / "first_crack" / ".." / "chunk_manifest.jsonl"
    )
    assert manifest.is_file()
    before = hashes(source)
    opened = tracked_descriptors(monkeypatch)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, source, tmp_path / "out", prior, "--chunk-manifest", str(manifest))
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "Preservation failed" in message and "source path components" in message
    assert not (tmp_path / "out").exists()
    assert hashes(source) == before
    assert_descriptors_closed(opened)


@pytest.mark.parametrize("component", ["", ".", ".."])
def test_bound_file_opener_refuses_unsafe_components_before_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    """Reject each forbidden raw component without attempting a relative open."""
    from contextlib import ExitStack

    source = corpus(tmp_path / "input")
    root = os.open(source, os.O_RDONLY | os.O_DIRECTORY)

    def forbid(*args: object, **kwargs: object) -> int:
        pytest.fail("Unsafe components reached filesystem traversal")

    monkeypatch.setattr(os, "open", forbid)
    try:
        with ExitStack() as handles, pytest.raises(ValueError, match="source path components"):
            splitter._open_source_file(root, (component, "chunk_manifest.jsonl"), handles)
    finally:
        os.close(root)
