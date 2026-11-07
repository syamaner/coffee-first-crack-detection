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
    # A declared but excluded stream must not inflate current usable counts.
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
        assert integrity["integrity_passed"] is True
        assert integrity["physical_session_count"] == 3
        assert integrity["stream_recording_count"] == (9 if changed else 6)
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
        markdown = (output / "split_report.md").read_text()
        assert f"**Total chunks**: {9 if changed else 6}" in markdown
        assert "new-pair" not in markdown and "excluded-stream" not in markdown
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
        assert "**Total chunks**: 0" in (output / "split_report.md").read_text()
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
    original = Path.mkdir

    def race(path: Path, *args: object, **kwargs: object) -> None:
        if path == output:
            original(path)
            (path / "sentinel").write_bytes(b"raced")
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race)
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
    original = Path.open
    planted: list[Path] = []

    def plant(path: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        if (
            path.is_relative_to(output)
            and not planted
            and (path.suffix == ".wav" if destination == "wav" else path.name == destination)
        ):
            with original(path, "wb") as handle:
                handle.write(b"sentinel")
            planted.append(path)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", plant)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, source, output, prior)
    assert planted[0].read_bytes() == b"sentinel"
    if destination == "wav":
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
    assert not (tmp_path / "out" / "split_integrity.json").exists()
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
