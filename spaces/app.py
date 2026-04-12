"""HuggingFace Space: Coffee First Crack Detection.

Upload a 10-second coffee roasting audio clip to detect whether first crack
has occurred. Uses the syamaner/coffee-first-crack-detection AST model via
the transformers audio-classification pipeline.
"""

from __future__ import annotations

import asyncio.base_events as _base_events


def _patch_asyncio_event_loop_del() -> None:
    """Suppress the specific ``ValueError: Invalid file descriptor: -1`` during GC.

    Gradio 6.x creates intermediate asyncio event loops during startup.
    When these are garbage-collected, ``BaseEventLoop.__del__`` tries to
    close an already-invalid file descriptor (-1), producing noisy but
    benign tracebacks in container logs.  This patch silences that exact
    error and re-raises any other ``ValueError`` so real errors are visible.

    Idempotent: applying the patch more than once (e.g. via
    ``importlib.reload()``) is a no-op.
    """
    original_del = getattr(_base_events.BaseEventLoop, "__del__", None)
    if original_del is None:
        return
    # Idempotency guard — avoid stacking wrappers on reload
    if getattr(original_del, "_spaces_app_patched", False):
        return

    def _patched_del(self: _base_events.BaseEventLoop) -> None:  # noqa: ANN
        try:
            original_del(self)
        except ValueError as exc:
            if str(exc) != "Invalid file descriptor: -1":
                raise

    _patched_del._spaces_app_patched = True  # type: ignore[attr-defined]
    _base_events.BaseEventLoop.__del__ = _patched_del  # type: ignore[attr-defined]


_patch_asyncio_event_loop_del()

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import pipeline as hf_pipeline

_REPO_ID = "syamaner/coffee-first-crack-detection"

# Example clips already hosted on the model repo
_EXAMPLES: dict[str, str] = {
    "First crack (10s)": "audio_examples/first_crack_sample.wav",
    "No first crack (10s)": "audio_examples/no_first_crack_sample.wav",
}

# Lazy-initialised pipeline — loaded on first inference request and cached.
# Avoids crashing the Space at startup if the Hub is temporarily unavailable.
_pipe: object = None


def _get_pipe() -> object:
    """Return the cached pipeline, initialising it on first call."""
    global _pipe  # noqa: PLW0603
    if _pipe is None:
        _pipe = hf_pipeline("audio-classification", model=_REPO_ID)
    return _pipe

_DESCRIPTION = """\
Upload a **10-second** coffee roasting audio clip to check for **first crack** \
— the critical moment when beans begin to pop.

The model is an Audio Spectrogram Transformer (AST) fine-tuned on 1,435 labelled \
10-second chunks from 21 roast recordings (97.7% test accuracy, 97.6% first-crack recall).

> **Note**: designed for 10-second windows. The feature extractor internally\
 truncates longer clips at the spectrogram level — for best results upload a\
 clip that is approximately 10 seconds. For full-recording sliding-window\
 detection see the [source repo](https://github.com/syamaner/coffee-first-crack-detection).

Model: [syamaner/coffee-first-crack-detection](https://huggingface.co/syamaner/coffee-first-crack-detection)

**Original prototype series (the earlier end-to-end build that led to this rewrite):**
- Part 1 — [Training a Neural Network to Detect Coffee First Crack from Audio](https://dev.to/syamaner/part-1-training-a-neural-network-to-detect-coffee-first-crack-from-audio-an-agentic-development-1jei)
- Part 2 — [Building MCP Servers to Control a Home Coffee Roaster](https://dev.to/syamaner/part-2-building-mcp-servers-to-control-a-home-coffee-roaster-an-agentic-development-journey-with-58ik)
- Part 3 — [From Neural Networks to Autonomous Coffee Roasting: Orchestrating MCP Servers](https://dev.to/syamaner/part-3-from-neural-networks-to-autonomous-coffee-roasting-orchestrating-mcp-servers-with-net-58pd)
"""


def load_example(choice: str | None) -> str | None:
    """Fetch a named example clip from the model repo into /tmp (Gradio-safe).

    Args:
        choice: Key from ``_EXAMPLES``, or None.

    Returns:
        Local filesystem path to the downloaded WAV file, or None.
    """
    if not choice or choice not in _EXAMPLES:
        return None
    # local_dir="/tmp" ensures the path is inside /tmp, which Gradio 6.x allows
    return hf_hub_download(repo_id=_REPO_ID, filename=_EXAMPLES[choice], local_dir="/tmp")


def classify(audio_path: str | None) -> dict[str, float]:
    """Classify a 10-second audio clip as first_crack or no_first_crack.

    Args:
        audio_path: Path to the uploaded audio file, or None if not provided.

    Returns:
        Dict mapping label names to their predicted probabilities.
    """
    if audio_path is None:
        return {}
    try:
        pipe = _get_pipe()
        results: list[dict[str, float | str]] = pipe(audio_path)  # type: ignore[assignment]
        return {str(r["label"]): round(float(r["score"]), 4) for r in results}
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Classification failed: {exc}") from exc


with gr.Blocks(title="☕ Coffee First Crack Detection") as demo:
    gr.Markdown("# ☕ Coffee First Crack Detection")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column():
            example_dd = gr.Dropdown(
                choices=list(_EXAMPLES),
                label="Try an example",
                value=None,
            )
            audio_in = gr.Audio(type="filepath", label="Upload Audio (WAV / MP3)")
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            output = gr.Label(num_top_classes=2, label="Prediction")

    example_dd.change(fn=load_example, inputs=example_dd, outputs=audio_in)
    submit_btn.click(fn=classify, inputs=audio_in, outputs=output)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
