"""Convert Label Studio JSON export to per-file annotation format.

Reads a Label Studio JSON export and produces one annotation JSON per audio
file, suitable for consumption by ``chunk_audio.py``.

Usage::

    python -m coffee_first_crack.data_prep.convert_labelstudio_export \\
        --input data/labels/project-1-at-YYYY-MM-DD.json \\
        --output data/labels \\
        --data-root data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa

SAMPLE_RATE = 44100


def strip_hash_prefix(filename: str) -> str:
    """Remove the hash prefix Label Studio adds to uploaded filenames.

    Args:
        filename: Potentially prefixed filename, e.g. ``"0d93a737-roast-1.wav"``.

    Returns:
        Original filename without the hash prefix.
    """
    if "-" in filename:
        return filename.split("-", 1)[1]
    return filename


def convert_task(task: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Convert a single Label Studio task to our annotation format.

    Args:
        task: A single task dict from the Label Studio JSON export.
        data_root: Local directory containing the WAV files.

    Returns:
        Dict with keys ``audio_file``, ``duration``, ``sample_rate``,
        and ``annotations`` (list of region dicts).
    """
    file_upload = task.get("file_upload")
    if file_upload:
        hashed_name = Path(file_upload).name
    else:
        audio_path = task.get("data", {}).get("audio", "")
        hashed_name = Path(audio_path).name

    original_name = strip_hash_prefix(hashed_name)
    local_audio_path = data_root / original_name

    try:
        duration = librosa.get_duration(path=str(local_audio_path))
    except Exception:
        duration = 0.0

    annotations: list[dict[str, Any]] = []
    ann_list = task.get("annotations") or []
    for ann in ann_list:
        results = ann.get("result") or []
        for r in results:
            if r.get("type") == "labels" and "value" in r:
                val = r["value"]
                labels = val.get("labels") or []
                if not labels:
                    continue
                annotations.append(
                    {
                        "id": f"chunk_{len(annotations):03d}",
                        "start_time": float(val.get("start", 0.0)),
                        "end_time": float(val.get("end", 0.0)),
                        "label": str(labels[0]),
                        "confidence": "high",
                    }
                )

    return {
        "audio_file": original_name,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "annotations": annotations,
    }


def main() -> None:
    """CLI entry point for converting Label Studio exports."""
    parser = argparse.ArgumentParser(
        description="Convert Label Studio JSON export to per-file annotation JSONs"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to Label Studio JSON export file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write per-file annotation JSONs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Local directory where WAV files live (default: data/raw)",
    )
    args = parser.parse_args()

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("r") as f:
        exported = json.load(f)

    converted_count = 0
    for task in exported:
        converted = convert_task(task, args.data_root)
        stem = Path(converted["audio_file"]).stem
        out_path = out_dir / f"{stem}.json"
        with out_path.open("w") as f:
            json.dump(converted, f, indent=2)
        print(f"Wrote {out_path}")
        converted_count += 1

    print(f"Converted {converted_count} tasks -> {out_dir}")


if __name__ == "__main__":
    main()
