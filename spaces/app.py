"""HuggingFace Space: Coffee First Crack Detection.

Upload a 10-second coffee roasting audio clip to detect whether first crack
has occurred. Uses the syamaner/coffee-first-crack-detection AST model via
the transformers audio-classification pipeline.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import pipeline

_REPO_ID = "syamaner/coffee-first-crack-detection"
_EXAMPLE_DIR = Path("audio_examples")
_EXAMPLE_DIR.mkdir(exist_ok=True)

# Download example clips from the model repo at startup (not committed to git)
for _label in ("first_crack", "no_first_crack"):
    hf_hub_download(
        repo_id=_REPO_ID,
        filename=f"audio_examples/{_label}_sample.wav",
        local_dir=".",
    )

# Load model pipeline once at startup
_pipe = pipeline("audio-classification", model=_REPO_ID)

_DESCRIPTION = """\
Upload a **10-second** coffee roasting audio clip to check for **first crack** \
— the critical moment when beans begin to pop.

The model is an Audio Spectrogram Transformer (AST) fine-tuned on 973 labelled \
10-second chunks from 15 roast recordings (97.4% test accuracy, 100% precision).

> **Note**: designed for 10-second windows. Longer clips are trimmed to their \
first 10 seconds. For full-recording sliding-window detection see the \
[source repo](https://github.com/syamaner/coffee-first-crack-detection).

Model: [syamaner/coffee-first-crack-detection](https://huggingface.co/syamaner/coffee-first-crack-detection)
"""


def classify(audio_path: str | None) -> dict[str, float]:
    """Classify a 10-second audio clip as first_crack or no_first_crack.

    Args:
        audio_path: Path to the uploaded audio file, or None if not provided.

    Returns:
        Dict mapping label names to their predicted probabilities.
    """
    if audio_path is None:
        return {}
    results: list[dict[str, float | str]] = _pipe(audio_path)  # type: ignore[assignment]
    return {str(r["label"]): round(float(r["score"]), 4) for r in results}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(
        type="filepath",
        label="Upload Audio (WAV / MP3)",
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="Prediction",
    ),
    title="☕ Coffee First Crack Detection",
    description=_DESCRIPTION,
    examples=[
        [str(_EXAMPLE_DIR / "first_crack_sample.wav")],
        [str(_EXAMPLE_DIR / "no_first_crack_sample.wav")],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
