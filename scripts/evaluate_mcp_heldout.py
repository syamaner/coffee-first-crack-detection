#!/usr/bin/env python3
"""Replay pair-safe held-out recordings through the MCP detector adapter.

This is an integration evaluator, not a second detector implementation. It loads
``coffee_roaster_mcp`` from an explicitly supplied checkout and uses its released
ONNX backend, detector adapter, detector-paced WAV pipeline, window timing, and
recent-positive confirmation rule. The agent harness profile is supplied as CLI
arguments so the recorded result states exactly what was exercised.

Legacy 44.1 kHz recordings are resampled to temporary 16 kHz PCM WAVs because the
live MCP detector consumes 16 kHz mono audio and deliberately rejects mismatched
WAV sample rates. Source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import librosa
import soundfile as sf

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class FirstCrackRegion:
    """One labelled first-crack interval on a recording's own time axis."""

    start_sec: float
    end_sec: float
    alignment_uncertainty_sec: float
    annotation_source: str


@dataclass(frozen=True)
class HeldOutRecording:
    """One full recording assigned to the pair-safe test split."""

    pair_id: str
    recording_id: str
    audio_path: Path
    label_path: Path
    mic_num: int | None
    mic_label: str | None
    source_sha256: str
    t0_offset_sec: float
    region: FirstCrackRegion | None


class ResolvedArtifactLike(Protocol):
    """Artifact attributes consumed from the selected MCP checkout."""

    local_path: Path


class ResolvedDetectorArtifactsLike(Protocol):
    """Resolved detector bundle attributes consumed by this evaluator."""

    onnx_model: ResolvedArtifactLike
    feature_extractor_config: ResolvedArtifactLike


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or fail with path context."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _git_head(repository_root: Path) -> str:
    """Return a repository HEAD using an explicitly resolved Git executable."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("Git executable is required to freeze the MCP source revision")
    # The executable is resolved above, argv is constant, and shell execution is disabled.
    return subprocess.run(  # noqa: S603
        [git_executable, "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _split_recording_ids(
    split_integrity_path: Path, chunk_manifest_path: Path
) -> tuple[set[str], set[tuple[str, str]]]:
    """Resolve all pair and stream IDs already assigned to dataset splits."""
    integrity = _read_json(split_integrity_path)
    try:
        splits = integrity["splits"]
        split_pair_ids = {
            str(pair_id) for split in splits.values() for pair_id in split["pair_ids"]
        }
        expected_streams = sum(int(split["stream_recording_count"]) for split in splits.values())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed split integrity report {split_integrity_path}") from exc

    resolved: set[tuple[str, str]] = set()
    try:
        lines = chunk_manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read chunk manifest {chunk_manifest_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            pair_id = str(row["pair_id"])
            recording_id = str(row["recording_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Malformed chunk manifest row {chunk_manifest_path}:{line_number}"
            ) from exc
        if pair_id in split_pair_ids:
            resolved.add((pair_id, recording_id))

    if len(resolved) != expected_streams:
        raise ValueError(
            "Split stream discovery disagrees with integrity report: "
            f"resolved {len(resolved)}, expected {expected_streams}"
        )
    resolved_pair_ids = {pair_id for pair_id, _ in resolved}
    if resolved_pair_ids != split_pair_ids:
        missing = sorted(split_pair_ids - resolved_pair_ids)
        extra = sorted(resolved_pair_ids - split_pair_ids)
        raise ValueError(f"Split pair mismatch; missing={missing}, extra={extra}")
    return split_pair_ids, resolved


def _used_source_hashes(capture_manifest_path: Path, split_pair_ids: set[str]) -> set[str]:
    """Resolve source-stream checksums already exposed to any dataset split."""
    manifest = _read_json(capture_manifest_path)
    checksums: set[str] = set()
    try:
        for session in manifest["sessions"]:
            if str(session["pair_id"]) not in split_pair_ids:
                continue
            checksums.update(str(stream["sha256"]) for stream in session["streams"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed capture manifest {capture_manifest_path}") from exc
    return checksums


def _resolve_label_path(recording_id: str, label_dirs: tuple[Path, ...]) -> Path:
    """Resolve exactly one annotation JSON for a held-out recording."""
    candidates = [directory / f"{recording_id}.json" for directory in label_dirs]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one label JSON for {recording_id!r}; found {matches}")
    return matches[0]


def _resolve_source_file(source_root: Path, value: object, *, field: str) -> Path:
    """Resolve a manifest source file without allowing reads outside its capture root."""
    if not isinstance(value, str):
        raise ValueError(f"Holdout manifest {field} must be an absolute path string")
    raw_path = Path(value)
    if not raw_path.is_absolute():
        raise ValueError(f"Holdout manifest {field} must be absolute: {value!r}")
    resolved = raw_path.resolve()
    if not resolved.is_relative_to(source_root):
        raise ValueError(f"Holdout manifest {field} escapes source_root: {value!r}")
    if not resolved.is_file() or raw_path.is_symlink():
        raise ValueError(f"Holdout manifest {field} is not a regular source file: {value!r}")
    return resolved


def _parse_region(label: dict[str, Any], label_path: Path) -> FirstCrackRegion | None:
    """Parse the optional single first-crack region and its provenance."""
    annotations = label.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"annotations must be a list in {label_path}")
    regions = [item for item in annotations if item.get("label") == "first_crack"]
    if len(regions) > 1:
        raise ValueError(f"Expected at most one first-crack region in {label_path}")
    if not regions:
        return None
    region = regions[0]
    try:
        start_sec = float(region["start_time"])
        end_sec = float(region["end_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed first-crack region in {label_path}") from exc
    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or start_sec < 0:
        raise ValueError(f"Invalid first-crack region in {label_path}")
    if end_sec <= start_sec:
        raise ValueError(f"first-crack end must be after start in {label_path}")

    provenance = label.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    uncertainty = float(provenance.get("alignment_uncertainty_seconds", 0.0))
    source = str(provenance.get("annotation_source", "legacy_human"))
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError(f"Invalid alignment uncertainty in {label_path}")
    return FirstCrackRegion(
        start_sec=start_sec,
        end_sec=end_sec,
        alignment_uncertainty_sec=uncertainty,
        annotation_source=source,
    )


def discover_heldout_recordings(
    *,
    split_integrity_path: Path,
    chunk_manifest_path: Path,
    dataset_capture_manifest_path: Path,
    holdout_capture_manifest_path: Path,
    label_dirs: tuple[Path, ...],
    pair_ids: set[str] | None,
    window_seconds: float,
) -> list[HeldOutRecording]:
    """Discover a fresh full-roast holdout and fail closed on prior exposure."""
    split_pair_ids, _ = _split_recording_ids(split_integrity_path, chunk_manifest_path)
    used_hashes = _used_source_hashes(dataset_capture_manifest_path, split_pair_ids)
    holdout_manifest = _read_json(holdout_capture_manifest_path)
    source_root_value = holdout_manifest.get("source_root")
    if not isinstance(source_root_value, str) or not Path(source_root_value).is_absolute():
        raise ValueError("Holdout manifest source_root must be an absolute path string")
    source_root = Path(source_root_value).resolve()
    if not source_root.is_dir():
        raise ValueError(f"Holdout manifest source_root is not a directory: {source_root}")
    recordings: list[HeldOutRecording] = []
    discovered_pairs: set[str] = set()
    try:
        sessions = holdout_manifest["sessions"]
        for session in sessions:
            pair_id = str(session["pair_id"])
            if pair_ids is not None and pair_id not in pair_ids:
                continue
            if pair_ids is None and pair_id in split_pair_ids:
                continue
            if pair_id in discovered_pairs:
                raise ValueError(f"Duplicate holdout pair ID {pair_id!r}")
            discovered_pairs.add(pair_id)
            if pair_id in split_pair_ids:
                raise ValueError(f"Holdout pair {pair_id!r} already appears in a dataset split")

            sidecar_path = _resolve_source_file(
                source_root,
                session["recording_sidecar_source_path"],
                field="recording_sidecar_source_path",
            )
            sidecar = _read_json(sidecar_path)
            milestones = sidecar.get("milestones")
            if not isinstance(milestones, dict) or milestones.get("beans_added") is None:
                raise ValueError(
                    f"Holdout pair {pair_id!r} lacks an authoritative recording-relative "
                    "beans_added milestone"
                )
            t0_offset_sec = float(milestones["beans_added"])
            if not math.isfinite(t0_offset_sec) or t0_offset_sec < 0:
                raise ValueError(f"Invalid beans_added milestone for holdout pair {pair_id!r}")

            streams = session["streams"]
            if len(streams) != 2 or {int(stream["mic_num"]) for stream in streams} != {1, 2}:
                raise ValueError(f"Holdout pair {pair_id!r} must contain exactly mic1 and mic2")
            for stream in streams:
                duration_sec = float(stream["duration_seconds"])
                if duration_sec < window_seconds:
                    raise ValueError(
                        f"Holdout pair {pair_id!r} mic{stream['mic_num']} is only "
                        f"{duration_sec:.2f}s; at least {window_seconds:.2f}s is required"
                    )
                source_sha256 = str(stream["sha256"])
                if source_sha256 in used_hashes:
                    raise ValueError(
                        f"Holdout pair {pair_id!r} reuses a source checksum already exposed "
                        "to a dataset split"
                    )
                audio_path = _resolve_source_file(
                    source_root, stream["source_path"], field="streams[].source_path"
                )
                if _sha256(audio_path) != source_sha256:
                    raise ValueError(f"Held-out source checksum changed: {audio_path}")
                recording_id = Path(str(stream["staged_relative_path"])).stem
                label_path = _resolve_label_path(recording_id, label_dirs)
                label = _read_json(label_path)
                if str(label.get("pair_id")) != pair_id:
                    raise ValueError(f"Label pair ID in {label_path} does not match {pair_id!r}")
                recordings.append(
                    HeldOutRecording(
                        pair_id=pair_id,
                        recording_id=recording_id,
                        audio_path=audio_path,
                        label_path=label_path,
                        mic_num=int(stream["mic_num"]),
                        mic_label=str(stream["label"]),
                        source_sha256=source_sha256,
                        t0_offset_sec=t0_offset_sec,
                        region=_parse_region(label, label_path),
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Malformed holdout manifest {holdout_capture_manifest_path}") from exc

    if pair_ids is not None and discovered_pairs != pair_ids:
        missing = sorted(pair_ids - discovered_pairs)
        raise ValueError(f"Requested holdout pair IDs were not found: {missing}")
    if not recordings:
        raise ValueError(
            "No fresh holdout recordings were found; provide new pair IDs absent from all splits"
        )
    return sorted(recordings, key=lambda item: (item.pair_id, item.mic_num or 0))


def classify_outcome(
    *,
    region: FirstCrackRegion | None,
    detected_sec: float | None,
    window_seconds: float,
) -> str:
    """Classify recording-level detection without hiding premature alerts."""
    if region is None:
        return "true_negative" if detected_sec is None else "false_positive"
    if detected_sec is None:
        return "missed"
    uncertainty = region.alignment_uncertainty_sec
    if detected_sec + window_seconds < region.start_sec - uncertainty:
        return "premature_false_alert"
    if detected_sec > region.end_sec + uncertainty:
        return "late_outside_region"
    return "detected"


def _prepare_audio(recording: HeldOutRecording, temp_dir: Path) -> tuple[Path, bool]:
    """Return a 16 kHz PCM path, resampling only a temporary copy when required."""
    info = sf.info(str(recording.audio_path))
    if info.samplerate == SAMPLE_RATE and info.channels == 1 and info.subtype.startswith("PCM"):
        return recording.audio_path, False
    audio, _ = librosa.load(str(recording.audio_path), sr=SAMPLE_RATE, mono=True)
    destination = temp_dir / f"{recording.recording_id}.wav"
    sf.write(str(destination), audio, SAMPLE_RATE, subtype="PCM_16")
    return destination, True


def _load_mcp(mcp_src: Path) -> dict[str, Any]:
    """Import the explicitly selected coffee-roaster-mcp checkout."""
    resolved = mcp_src.resolve()
    if not (resolved / "coffee_roaster_mcp" / "detector.py").is_file():
        raise ValueError(f"Not a coffee-roaster-mcp src directory: {resolved}")
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "coffee_roaster_mcp" or name.startswith("coffee_roaster_mcp.")
    )
    if preloaded:
        raise RuntimeError(
            "coffee_roaster_mcp was imported before source selection; "
            f"refusing ambiguous provenance: {preloaded}"
        )
    sys.path.insert(0, str(resolved))
    modules = {
        name: importlib.import_module(f"coffee_roaster_mcp.{name}")
        for name in ("artifacts", "audio", "config", "detector")
    }
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(resolved):
            raise RuntimeError(
                f"Imported coffee_roaster_mcp.{name} outside selected source tree: {module_file}"
            )

    return {
        "AudioConfig": modules["config"].AudioConfig,
        "FirstCrackConfig": modules["config"].FirstCrackConfig,
        "ResolvedArtifact": modules["artifacts"].ResolvedArtifact,
        "ResolvedDetectorArtifacts": modules["artifacts"].ResolvedDetectorArtifacts,
        "build_audio_capture_pipeline": modules["audio"].build_audio_capture_pipeline,
        "build_first_crack_detector_adapter": modules[
            "detector"
        ].build_first_crack_detector_adapter,
        "build_released_onnx_first_crack_detector_backend": (
            modules["detector"].build_released_onnx_first_crack_detector_backend
        ),
    }


def _resolved_artifacts(
    mcp: dict[str, Any], *, onnx_dir: Path, repo_id: str, revision: str | None
) -> ResolvedDetectorArtifactsLike:
    """Build MCP artifact metadata for an unpublished local candidate."""
    onnx_models = sorted(onnx_dir.glob("*.onnx"))
    if len(onnx_models) != 1:
        raise ValueError(f"Expected exactly one ONNX model in {onnx_dir}; found {onnx_models}")
    preprocessor = onnx_dir / "preprocessor_config.json"
    if not preprocessor.is_file():
        raise ValueError(f"Missing {preprocessor}")
    artifact_type = mcp["ResolvedArtifact"]
    return cast(
        ResolvedDetectorArtifactsLike,
        mcp["ResolvedDetectorArtifacts"](
            onnx_model=artifact_type(
                repo_id=repo_id,
                revision=revision,
                filename="onnx/int8/model_quantized.onnx",
                local_path=onnx_models[0].resolve(),
            ),
            feature_extractor_config=artifact_type(
                repo_id=repo_id,
                revision=revision,
                filename="onnx/int8/preprocessor_config.json",
                local_path=preprocessor.resolve(),
            ),
        ),
    )


def _evaluate_recording(
    *,
    mcp: dict[str, Any],
    config: object,
    artifacts: ResolvedDetectorArtifactsLike,
    backend: object,
    recording: HeldOutRecording,
    replay_path: Path,
    resampled: bool,
    window_seconds: float,
    overlap: float,
) -> dict[str, Any]:
    """Replay one full recording through the real MCP window and adapter path."""
    audio_config = mcp["AudioConfig"](
        source="wav",
        input_device=recording.mic_label,
        sample_rate=SAMPLE_RATE,
        wav_path=replay_path,
        replay_mode="detector_paced",
        window_seconds=window_seconds,
        overlap=overlap,
        hop_seconds=None,
    )
    pipeline = mcp["build_audio_capture_pipeline"](audio_config)
    adapter = mcp["build_first_crack_detector_adapter"](config, artifacts, backend)
    pipeline.start()

    first_window_started: float | None = None
    event: Any | None = None
    processed_windows = 0
    positive_windows = 0
    max_confidence = 0.0
    inference_latencies_ms: list[float] = []
    confirming_inference_latency_ms: float | None = None
    started_at = time.monotonic()
    try:
        while True:
            windows = pipeline.drain_windows(max_windows=1)
            if not windows:
                break
            window = windows[0]
            if first_window_started is None:
                first_window_started = float(window.started_at_monotonic_seconds)
            inference_started_at = time.perf_counter()
            observation = adapter.process_window_observed(
                window,
                earliest_eligible_monotonic_seconds=first_window_started,
            )
            inference_latency_ms = (time.perf_counter() - inference_started_at) * 1000.0
            inference_latencies_ms.append(inference_latency_ms)
            processed_windows += 1
            confidence = observation.confidence
            if confidence is not None:
                max_confidence = max(max_confidence, float(confidence))
            if observation.fc_status in {"candidate", "confirmed"}:
                positive_windows += 1
            if observation.event is not None:
                event = observation.event
                confirming_inference_latency_ms = inference_latency_ms
                break
    finally:
        pipeline.stop()

    detected_sec: float | None = None
    confirmed_sec: float | None = None
    event_confidence: float | None = None
    confirming_sequence: int | None = None
    notification_sec: float | None = None
    if event is not None:
        if first_window_started is None:
            raise RuntimeError("MCP emitted an event before the first replay window")
        detected_sec = round(float(event.detected_at_monotonic_seconds) - first_window_started, 6)
        confirmed_sec = round(float(event.confirmed_at_monotonic_seconds) - first_window_started, 6)
        event_confidence = None if event.confidence is None else float(event.confidence)
        confirming_sequence = int(event.confirmed_by_window_sequence_number)
        if confirming_inference_latency_ms is None:
            raise RuntimeError("MCP confirmation is missing its measured inference latency")
        notification_sec = confirmed_sec + confirming_inference_latency_ms / 1000.0

    region = recording.region
    label_start_after_t0 = None if region is None else region.start_sec - recording.t0_offset_sec
    event_after_t0 = None if detected_sec is None else detected_sec - recording.t0_offset_sec
    confirmed_after_t0 = None if confirmed_sec is None else confirmed_sec - recording.t0_offset_sec
    notification_after_t0 = (
        None if notification_sec is None else notification_sec - recording.t0_offset_sec
    )
    timing_error = (
        None
        if label_start_after_t0 is None or event_after_t0 is None
        else event_after_t0 - label_start_after_t0
    )
    confirmation_delay = (
        None
        if label_start_after_t0 is None or notification_after_t0 is None
        else notification_after_t0 - label_start_after_t0
    )
    return {
        "pair_id": recording.pair_id,
        "recording_id": recording.recording_id,
        "mic_num": recording.mic_num,
        "mic_label": recording.mic_label,
        "audio_path": str(recording.audio_path),
        "audio_sha256": recording.source_sha256,
        "label_path": str(recording.label_path),
        "ground_truth": None if region is None else asdict(region),
        "t0_alignment": {
            "source": "roast.recording.json:milestones.beans_added",
            "wav_offset_sec": recording.t0_offset_sec,
            "label_start_sec_after_t0": label_start_after_t0,
        },
        "resampled_to_16khz_temporary_copy": resampled,
        "detected": event is not None,
        "event_count": 1 if event is not None else 0,
        "detected_sec": detected_sec,
        "confirmed_sec": confirmed_sec,
        "notification_sec": notification_sec,
        "event_sec_after_t0": event_after_t0,
        "confirmed_sec_after_t0": confirmed_after_t0,
        "notification_sec_after_t0": notification_after_t0,
        "event_timing_error_from_label_start_sec": timing_error,
        "agent_confirmation_delay_from_label_start_sec": confirmation_delay,
        "event_confidence": event_confidence,
        "max_processed_confidence": max_confidence,
        "confirming_window_sequence": confirming_sequence,
        "processed_window_count": processed_windows,
        "positive_window_count_before_stop": positive_windows,
        "inference_latency_ms": _distribution(inference_latencies_ms),
        "confirming_inference_latency_ms": confirming_inference_latency_ms,
        "outcome": classify_outcome(
            region=region,
            detected_sec=detected_sec,
            window_seconds=window_seconds,
        ),
        "wall_seconds": time.monotonic() - started_at,
    }


def _select_deployed_stream(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select mic1 as the protocol's frozen primary live-path stream."""
    if len(results) == 1:
        return results[0]
    mic1 = [result for result in results if result["mic_num"] == 1]
    if len(mic1) != 1:
        raise ValueError(
            f"Could not select one mic1 primary stream for pair {results[0]['pair_id']}"
        )
    return mic1[0]


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build stream-level and deployed physical-session summaries."""
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_pair[str(result["pair_id"])].append(result)
    deployed = [_select_deployed_stream(pair_results) for pair_results in by_pair.values()]

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        positive = [item for item in items if item["ground_truth"] is not None]
        negative = [item for item in items if item["ground_truth"] is None]
        detected = [item for item in positive if item["outcome"] == "detected"]
        timing_errors = [
            float(item["event_timing_error_from_label_start_sec"]) for item in detected
        ]
        confirmation_delays = [
            float(item["agent_confirmation_delay_from_label_start_sec"]) for item in detected
        ]
        return {
            "count": len(items),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "detected_positive_count": len(detected),
            "detection_rate": len(detected) / len(positive) if positive else None,
            "false_positive_count": sum(item["outcome"] == "false_positive" for item in negative),
            "missed_count": sum(item["outcome"] == "missed" for item in positive),
            "premature_false_alert_count": sum(
                item["outcome"] == "premature_false_alert" for item in positive
            ),
            "late_outside_region_count": sum(
                item["outcome"] == "late_outside_region" for item in positive
            ),
            "one_event_max_passed": all(int(item["event_count"]) <= 1 for item in items),
            "timing_error_sec": _distribution(timing_errors),
            "agent_confirmation_delay_sec": _distribution(confirmation_delays),
        }

    pair_comparisons = []
    for pair_id, pair_results in sorted(by_pair.items()):
        if len(pair_results) != 2:
            continue
        detected_times = [item["event_sec_after_t0"] for item in pair_results]
        pair_comparisons.append(
            {
                "pair_id": pair_id,
                "recording_ids": [item["recording_id"] for item in pair_results],
                "both_detected": all(value is not None for value in detected_times),
                "event_time_delta_sec": (
                    None
                    if any(value is None for value in detected_times)
                    else abs(float(detected_times[0]) - float(detected_times[1]))
                ),
            }
        )
    return {
        "stream_level": summarize(results),
        "deployed_physical_session_level": summarize(deployed),
        "deployed_recording_ids": sorted(str(item["recording_id"]) for item in deployed),
        "paired_stream_comparisons": pair_comparisons,
    }


def _distribution(values: list[float]) -> dict[str, float] | None:
    """Return a compact deterministic distribution summary."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"min": min(values), "median": median, "max": max(values)}


def _write_markdown(output_path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable companion table next to the JSON report."""
    profile = report["profile"]
    aggregate = report["aggregate"]
    lines = [
        "# MCP held-out full-recording replay",
        "",
        "## Profile",
        "",
        f"- Model: `{report['model']['onnx_path']}`",
        f"- Model SHA-256: `{report['model']['onnx_sha256']}`",
        f"- MCP source: `{report['mcp']['source_path']}` at `{report['mcp']['git_head']}`",
        f"- Window/hop: {profile['window_seconds']:.1f}s / {profile['hop_seconds']:.1f}s",
        f"- Threshold: {profile['confidence_threshold']}",
        f"- Confirmation: {profile['min_positive_windows']} positives within "
        f"{profile['confirmation_window_seconds']:.1f}s",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "## Recordings",
        "",
        "| Pair | Recording | Mic | GT after T0 | Event after T0 | "
        "Notified after T0 | Error | Outcome |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["recordings"]:
        t0_alignment = item["t0_alignment"]
        gt_start = (
            "—"
            if t0_alignment["label_start_sec_after_t0"] is None
            else f"{t0_alignment['label_start_sec_after_t0']:.1f}"
        )
        detected = (
            "—" if item["event_sec_after_t0"] is None else f"{item['event_sec_after_t0']:.1f}"
        )
        confirmed = (
            "—"
            if item["notification_sec_after_t0"] is None
            else f"{item['notification_sec_after_t0']:.1f}"
        )
        error = (
            "—"
            if item["event_timing_error_from_label_start_sec"] is None
            else f"{item['event_timing_error_from_label_start_sec']:+.1f}"
        )
        lines.append(
            f"| `{item['pair_id']}` | `{item['recording_id']}` | "
            f"{item['mic_num'] or '—'} | {gt_start} | {detected} | {confirmed} | "
            f"{error} | {item['outcome']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze_protocol(path: Path, protocol: dict[str, Any]) -> None:
    """Persist or verify the immutable pre-inference protocol lock."""
    serialized = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(
                f"Frozen protocol {path} differs from this invocation; use a new cohort/output"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the complete held-out MCP replay and return its auditable report."""
    mcp = _load_mcp(args.mcp_src)
    recordings = discover_heldout_recordings(
        split_integrity_path=args.split_integrity,
        chunk_manifest_path=args.chunk_manifest,
        dataset_capture_manifest_path=args.dataset_capture_manifest,
        holdout_capture_manifest_path=args.holdout_capture_manifest,
        label_dirs=tuple(args.labels_dir),
        pair_ids=None if args.pair_id is None else set(args.pair_id),
        window_seconds=args.window_seconds,
    )
    artifacts = _resolved_artifacts(
        mcp,
        onnx_dir=args.onnx_dir,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    model_path = Path(artifacts.onnx_model.local_path)
    hop_seconds = args.window_seconds * (1.0 - args.overlap)
    mcp_root = args.mcp_src.resolve().parent
    git_head = _git_head(mcp_root)
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_inference",
        "mcp": {"source_path": str(mcp_root), "git_head": git_head},
        "model": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "precision": "int8",
            "onnx_sha256": _sha256(model_path),
            "preprocessor_sha256": _sha256(Path(artifacts.feature_extractor_config.local_path)),
        },
        "profile": {
            "sample_rate": SAMPLE_RATE,
            "window_seconds": args.window_seconds,
            "overlap": args.overlap,
            "hop_seconds": hop_seconds,
            "confidence_threshold": args.threshold,
            "min_positive_windows": args.min_positive_windows,
            "confirmation_window_seconds": args.confirmation_window,
            "onnx_threads": args.threads,
            "primary_stream": "mic1",
            "paired_robustness_stream": "mic2",
        },
        "holdout": [
            {
                "pair_id": recording.pair_id,
                "recording_id": recording.recording_id,
                "mic_num": recording.mic_num,
                "audio_sha256": recording.source_sha256,
                "label_sha256": _sha256(recording.label_path),
                "t0_offset_sec": recording.t0_offset_sec,
            }
            for recording in recordings
        ],
    }
    _freeze_protocol(args.output.with_suffix(".protocol.json"), protocol)
    first_crack_config = mcp["FirstCrackConfig"](
        mode="audio",
        repo_id=args.repo_id,
        revision=args.revision,
        precision="int8",
        local_model_dir=None,
        onnx_threads=args.threads,
        confidence_threshold=args.threshold,
        min_positive_windows=args.min_positive_windows,
        confirmation_window_seconds=args.confirmation_window,
        allow_manual_override=True,
    )
    backend = mcp["build_released_onnx_first_crack_detector_backend"](first_crack_config, artifacts)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mcp-heldout-replay-") as temp:
        temp_dir = Path(temp)
        for index, recording in enumerate(recordings, start=1):
            replay_path, resampled = _prepare_audio(recording, temp_dir)
            print(
                f"[{index}/{len(recordings)}] {recording.recording_id} "
                f"({'positive' if recording.region else 'negative'})",
                flush=True,
            )
            result = _evaluate_recording(
                mcp=mcp,
                config=first_crack_config,
                artifacts=artifacts,
                backend=backend,
                recording=recording,
                replay_path=replay_path,
                resampled=resampled,
                window_seconds=args.window_seconds,
                overlap=args.overlap,
            )
            results.append(result)
            print(
                f"  {result['outcome']}: event={result['detected_sec']}s, "
                f"confirmed={result['confirmed_sec']}s, "
                f"max_p={result['max_processed_confidence']:.4f}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "generated_at_unix_seconds": time.time(),
        "protocol_lock_path": str(args.output.with_suffix(".protocol.json").resolve()),
        "test_set": {
            "cohort": "fresh_full_recording_holdout",
            "split_integrity_path": str(args.split_integrity.resolve()),
            "chunk_manifest_path": str(args.chunk_manifest.resolve()),
            "dataset_capture_manifest_path": str(args.dataset_capture_manifest.resolve()),
            "holdout_capture_manifest_path": str(args.holdout_capture_manifest.resolve()),
            "physical_session_count": len({item.pair_id for item in recordings}),
            "stream_recording_count": len(recordings),
            "pair_ids_absent_from_all_splits": True,
            "source_checksums_absent_from_all_splits": True,
            "authoritative_t0_alignment_present": True,
        },
        "mcp": {"source_path": str(mcp_root), "git_head": git_head},
        "model": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "precision": "int8",
            "onnx_path": str(model_path.resolve()),
            "onnx_sha256": _sha256(model_path),
            "preprocessor_path": str(Path(artifacts.feature_extractor_config.local_path)),
        },
        "profile": {
            "sample_rate": SAMPLE_RATE,
            "window_seconds": args.window_seconds,
            "overlap": args.overlap,
            "hop_seconds": hop_seconds,
            "confidence_threshold": args.threshold,
            "min_positive_windows": args.min_positive_windows,
            "confirmation_window_seconds": args.confirmation_window,
            "onnx_threads": args.threads,
            "timestamp_policy": "mcp_backdate_to_earliest_qualifying_window_onset",
        },
        "recordings": results,
        "aggregate": _aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    _write_markdown(markdown_path, report)
    print(f"Wrote {args.output} and {markdown_path}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-src", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/baseline-v6-pair-aware")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-positive-windows", type=int, default=5)
    parser.add_argument("--confirmation-window", type=float, default=20.0)
    parser.add_argument(
        "--split-integrity", type=Path, default=Path("data/splits/split_integrity.json")
    )
    parser.add_argument(
        "--chunk-manifest", type=Path, default=Path("data/processed/chunk_manifest.jsonl")
    )
    parser.add_argument(
        "--dataset-capture-manifest",
        type=Path,
        default=Path("data/raw/mcp-captures/capture_manifest.json"),
    )
    parser.add_argument(
        "--holdout-capture-manifest",
        type=Path,
        default=Path("data/raw/mcp-captures/capture_manifest.json"),
    )
    parser.add_argument(
        "--pair-id",
        action="append",
        default=None,
        help="Fresh physical-session pair ID to replay; repeat for multiple sessions.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each label directory (default: data/labels/mcp and data/labels).",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.labels_dir is None:
        args.labels_dir = [Path("data/labels/mcp"), Path("data/labels")]
    if args.threads <= 0:
        parser.error("--threads must be > 0")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    evaluate(args)


if __name__ == "__main__":
    main()
