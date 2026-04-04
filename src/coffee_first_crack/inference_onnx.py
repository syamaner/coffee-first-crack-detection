"""ONNX inference module: sliding window and live detection for Raspberry Pi 5.

Mirrors :mod:`coffee_first_crack.inference` but replaces PyTorch model inference
with ONNX Runtime.  Loads models and feature extractor from HuggingFace Hub.

Two main classes:

- ``OnnxSlidingWindowInference`` — offline processing of audio files
- ``OnnxFirstCrackDetector`` — real-time streaming (file or microphone) with
  thread-safe pop-confirmation logic

Usage::

    # File-based sliding window (loads INT8 model from HF Hub)
    python -m coffee_first_crack.inference_onnx \
        --audio data/raw/mic2-brazil-roast1.wav

    # With Pi-optimised parameters
    python -m coffee_first_crack.inference_onnx \
        --audio data/raw/mic2-brazil-roast1.wav --profile pi_inference

    # Live microphone on Pi
    python -m coffee_first_crack.inference_onnx --microphone \
        --profile pi_inference --threads 4

    # List audio devices
    python -c "import sounddevice as sd; print(sd.query_devices())"
"""

from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import yaml

from coffee_first_crack.inference import DetectionEvent, _format_time

if TYPE_CHECKING:
    import onnxruntime as rt
    from transformers import ASTFeatureExtractor

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_REPO_ID = "syamaner/coffee-first-crack-detection"
_DEFAULT_SUBFOLDER = "onnx/int8"
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
SAMPLE_RATE = 16000


# ── Config loading ────────────────────────────────────────────────────────────


def _load_profile(profile: str = "inference") -> dict:
    """Load an inference profile from ``configs/default.yaml``.

    Args:
        profile: Config key to load (``"inference"`` or ``"pi_inference"``).

    Returns:
        Dict of inference parameters.
    """
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg.get(profile, {}))


def _default_threads() -> int:
    """Return a safe default thread count for the current platform.

    ARM64 devices default to 2 threads; x86 lets ONNX Runtime auto-detect.
    """
    if platform.machine() in ("aarch64", "arm64"):
        return 2
    return 0  # 0 = let ONNX Runtime decide


# ── Model loading ─────────────────────────────────────────────────────────────


def _resolve_onnx_model(
    repo_id: str = _DEFAULT_REPO_ID,
    subfolder: str = _DEFAULT_SUBFOLDER,
) -> str:
    """Download the ONNX model from HuggingFace Hub and return the local path.

    Tries ``model_quantized.onnx`` first, then ``model.onnx``.

    Args:
        repo_id: HuggingFace Hub repository ID.
        subfolder: Subfolder within the HF repo containing the ONNX model.

    Returns:
        Local filesystem path to the downloaded ONNX model.

    Raises:
        FileNotFoundError: If no ONNX model is found in the given subfolder.
    """
    from huggingface_hub import hf_hub_download

    for filename in ("model_quantized.onnx", "model.onnx"):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
            )
            return path
        except Exception:  # noqa: BLE001
            continue

    raise FileNotFoundError(
        f"No ONNX model found in {repo_id}/{subfolder} (tried model_quantized.onnx, model.onnx)"
    )


def _load_extractor(
    repo_id: str = _DEFAULT_REPO_ID,
    subfolder: str = _DEFAULT_SUBFOLDER,
) -> ASTFeatureExtractor:
    """Load the ASTFeatureExtractor from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID.
        subfolder: Subfolder containing ``preprocessor_config.json``.

    Returns:
        An initialised ``ASTFeatureExtractor``.
    """
    from transformers import ASTFeatureExtractor

    return ASTFeatureExtractor.from_pretrained(repo_id, subfolder=subfolder)


def _create_session(
    onnx_path: str,
    threads: int = 0,
) -> rt.InferenceSession:
    """Create an ONNX Runtime inference session.

    Args:
        onnx_path: Path to the ONNX model file.
        threads: Number of intra-op threads (0 = auto).

    Returns:
        An ``onnxruntime.InferenceSession``.
    """
    import onnxruntime as rt

    sess_options = rt.SessionOptions()
    if threads > 0:
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1
    return rt.InferenceSession(
        onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )


# ── Sliding window (offline) ─────────────────────────────────────────────────


class OnnxSlidingWindowInference:
    """Process an audio file with a sliding window using ONNX Runtime.

    Loads the ONNX model and feature extractor from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID.
        subfolder: Subfolder in the repo containing the ONNX model.
        window_size: Window length in seconds (default: 10.0).
        overlap: Overlap fraction between consecutive windows (default: 0.7).
        threshold: Minimum first-crack probability to count as a pop (default: 0.6).
        min_pops: Number of positive windows required for confirmation (default: 5).
        confirmation_window: Time span in seconds over which pops are counted (default: 20.0).
        threads: ONNX Runtime intra-op threads (0 = auto).
        profile: Config profile name to load defaults from (``"inference"`` or
            ``"pi_inference"``).  Explicit kwargs override profile values.
    """

    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO_ID,
        subfolder: str = _DEFAULT_SUBFOLDER,
        window_size: float | None = None,
        overlap: float | None = None,
        threshold: float | None = None,
        min_pops: int | None = None,
        confirmation_window: float | None = None,
        threads: int | None = None,
        profile: str = "inference",
    ) -> None:
        cfg = _load_profile(profile)

        self.window_size = window_size or cfg.get("window_size", 10.0)
        self.overlap = overlap or cfg.get("overlap", 0.7)
        self.threshold = threshold or cfg.get("threshold", 0.6)
        self.min_pops = min_pops or cfg.get("min_pops", 5)
        self.confirmation_window = confirmation_window or cfg.get("confirmation_window", 20.0)
        self.sample_rate = SAMPLE_RATE
        self.window_samples = int(self.window_size * self.sample_rate)
        self.hop_samples = int(self.window_samples * (1 - self.overlap))

        default_threads = cfg.get("onnx_threads", _default_threads())
        onnx_threads = threads if threads is not None else default_threads
        onnx_path = _resolve_onnx_model(repo_id, subfolder)
        self._extractor = _load_extractor(repo_id, subfolder)
        self._session = _create_session(onnx_path, onnx_threads)
        self._input_name = self._session.get_inputs()[0].name

    def process_file(self, audio_path: str | Path) -> list[DetectionEvent]:
        """Process an audio file and return confirmed detection events.

        Args:
            audio_path: Path to the WAV file to process.

        Returns:
            List of :class:`DetectionEvent` instances (usually 0 or 1).
        """
        audio_path = Path(audio_path)
        audio, _sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        duration = len(audio) / self.sample_rate
        print(f"Processing: {audio_path.name} ({duration:.1f}s)")

        history: list[tuple[float, bool, float]] = []
        events: list[DetectionEvent] = []
        confirmed = False
        start = 0

        while start + self.window_samples <= len(audio):
            window = audio[start : start + self.window_samples]
            current_time = start / self.sample_rate

            prob = self._predict_window(window)
            is_positive = prob >= self.threshold
            history.append((current_time, is_positive, prob))

            # Count pops within confirmation window
            cutoff = current_time - self.confirmation_window
            recent_positives = sum(1 for t, pos, _ in history if t >= cutoff and pos)

            if not confirmed and recent_positives >= self.min_pops:
                confirmed = True
                first_t = next(t for t, pos, _ in history if pos and t >= cutoff)
                event = DetectionEvent(
                    timestamp_sec=first_t,
                    timestamp_str=_format_time(first_t),
                    confidence=recent_positives,
                )
                events.append(event)
                print(
                    f"FIRST CRACK detected at {event.timestamp_str} "
                    f"(confidence: {recent_positives} pops)"
                )

            start += self.hop_samples

        if not events:
            print("No first crack detected in file.")
        return events

    def _predict_window(self, window: np.ndarray) -> float:
        """Predict first-crack probability for a single audio window.

        Applies an energy-based noise gate: silent windows return 0.0.
        """
        if np.sqrt(np.mean(window**2)) < 0.01:
            return 0.0

        inputs = self._extractor(
            [window.tolist()], sampling_rate=self.sample_rate, return_tensors="np"
        )
        logits = self._session.run(None, {self._input_name: inputs["input_values"]})[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return float(probs[0, 1])


# ── Live streaming detector ──────────────────────────────────────────────────


class OnnxFirstCrackDetector:
    """Real-time first crack detector using ONNX Runtime.

    Runs inference in a background thread.  Call :meth:`start`, then poll
    :meth:`is_first_crack` in your roast loop, then :meth:`stop`.

    Args:
        audio_file: Path to WAV file (mutually exclusive with ``use_microphone``).
        use_microphone: Stream from the default (or specified) microphone.
        device_index: Sounddevice device index for microphone input.
        repo_id: HuggingFace Hub repository ID.
        subfolder: Subfolder in the repo containing the ONNX model.
        window_size: Window length in seconds.
        overlap: Overlap fraction.
        threshold: Positive classification threshold.
        sample_rate: Audio sample rate (must match model: 16 kHz).
        min_pops: Pops required within ``confirmation_window`` to confirm.
        confirmation_window: Time span in seconds for pop counting.
        threads: ONNX Runtime intra-op threads.
        profile: Config profile name to load defaults from.
    """

    def __init__(
        self,
        audio_file: str | Path | None = None,
        use_microphone: bool = False,
        device_index: int | None = None,
        repo_id: str = _DEFAULT_REPO_ID,
        subfolder: str = _DEFAULT_SUBFOLDER,
        window_size: float | None = None,
        overlap: float | None = None,
        threshold: float | None = None,
        sample_rate: int = SAMPLE_RATE,
        min_pops: int | None = None,
        confirmation_window: float | None = None,
        threads: int | None = None,
        profile: str = "inference",
    ) -> None:
        if audio_file and use_microphone:
            raise ValueError("Specify either audio_file or use_microphone, not both.")
        if not audio_file and not use_microphone:
            raise ValueError("Must specify audio_file or use_microphone.")

        cfg = _load_profile(profile)

        self.audio_file = Path(audio_file) if audio_file else None
        self.use_microphone = use_microphone
        self.device_index = device_index
        self.window_size = window_size or cfg.get("window_size", 10.0)
        self.overlap = overlap or cfg.get("overlap", 0.7)
        self.threshold = threshold or cfg.get("threshold", 0.6)
        self.sample_rate = sample_rate
        self.min_pops = min_pops or cfg.get("min_pops", 5)
        self.confirmation_window = confirmation_window or cfg.get("confirmation_window", 20.0)

        self.window_samples = int(self.window_size * sample_rate)
        self.hop_samples = int(self.window_samples * (1 - self.overlap))

        default_threads = cfg.get("onnx_threads", _default_threads())
        onnx_threads = threads if threads is not None else default_threads
        onnx_path = _resolve_onnx_model(repo_id, subfolder)
        self._extractor = _load_extractor(repo_id, subfolder)
        self._session = _create_session(onnx_path, onnx_threads)
        self._input_name = self._session.get_inputs()[0].name

        # State
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._first_crack_detected = False
        self._first_crack_time: float | None = None
        self._start_time: float | None = None
        self._audio_buffer: deque = deque(maxlen=int(sample_rate * 60))
        self._detection_history: deque = deque(maxlen=200)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start detection in a background thread."""
        if self._running:
            raise RuntimeError("Detector already running.")
        self._running = True
        self._start_time = time.time()

        if self.use_microphone:
            self._thread = threading.Thread(target=self._microphone_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._file_loop, daemon=True)
        self._thread.start()
        print("First crack detector started (ONNX).")

    def stop(self) -> None:
        """Stop detection and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._audio_buffer.clear()
        self._detection_history.clear()
        print("First crack detector stopped.")

    def is_first_crack(self) -> bool | tuple[bool, str]:
        """Check detection state.

        Returns:
            ``False`` if not yet detected, or ``(True, "MM:SS")`` when confirmed.
        """
        with self._lock:
            if not self._first_crack_detected:
                return False
            return True, _format_time(self._first_crack_time or 0.0)

    def get_elapsed_time(self) -> str | None:
        """Return elapsed time since :meth:`start` as ``"MM:SS"``."""
        if self._start_time is None:
            return None
        return _format_time(time.time() - self._start_time)

    @property
    def is_running(self) -> bool:
        """True while the detector background thread is active."""
        return self._running

    # ── Internal loops ────────────────────────────────────────────────────────

    def _file_loop(self) -> None:
        """Background thread: process audio file with sliding window."""
        try:
            audio, _ = librosa.load(str(self.audio_file), sr=self.sample_rate, mono=True)
            start = 0
            while self._running and start + self.window_samples <= len(audio):
                window = audio[start : start + self.window_samples]
                current_time = start / self.sample_rate
                prob = self._predict_window(window)
                self._update_state(prob, current_time)
                start += self.hop_samples
                time.sleep(0.05)
        except Exception as exc:
            print(f"File processing error: {exc}")
        finally:
            self._running = False

    def _microphone_loop(self) -> None:
        """Background thread: stream audio from microphone."""
        try:
            import sounddevice as sd  # lazy import — not needed without mic

            def _callback(indata: np.ndarray, _frames: int, _t: object, _status: object) -> None:
                audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                with self._lock:
                    self._audio_buffer.extend(audio_data)

            stream_kwargs: dict = {
                "samplerate": self.sample_rate,
                "channels": 1,
                "callback": _callback,
                "blocksize": int(self.sample_rate * 0.5),
            }
            if self.device_index is not None:
                stream_kwargs["device"] = self.device_index

            with sd.InputStream(**stream_kwargs):
                print(f"Microphone stream active at {self.sample_rate} Hz")
                while self._running:
                    with self._lock:
                        buf_size = len(self._audio_buffer)
                    if buf_size >= self.window_samples:
                        with self._lock:
                            window = np.array(list(self._audio_buffer)[-self.window_samples :])
                        current_time = time.time() - (self._start_time or time.time())
                        prob = self._predict_window(window)
                        self._update_state(prob, current_time)
                    time.sleep(0.5)
        except Exception as exc:
            print(f"Microphone error: {exc}")
            self._running = False

    def _predict_window(self, window: np.ndarray) -> float:
        """Predict first-crack probability with noise gate."""
        if np.sqrt(np.mean(window**2)) < 0.01:
            return 0.0

        inputs = self._extractor(
            [window.tolist()], sampling_rate=self.sample_rate, return_tensors="np"
        )
        logits = self._session.run(None, {self._input_name: inputs["input_values"]})[0]

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return float(probs[0, 1])

    def _update_state(self, prob: float, current_time: float) -> None:
        """Update detection history and check confirmation threshold."""
        with self._lock:
            is_positive = prob >= self.threshold
            self._detection_history.append((current_time, is_positive, prob))

            cutoff = current_time - self.confirmation_window
            recent_positives = sum(
                1 for t, pos, _ in self._detection_history if t >= cutoff and pos
            )

            if not self._first_crack_detected and recent_positives >= self.min_pops:
                self._first_crack_detected = True
                self._first_crack_time = next(
                    t for t, pos, _ in self._detection_history if pos and t >= cutoff
                )
                print(
                    f"FIRST CRACK at {_format_time(self._first_crack_time or 0.0)} "
                    f"(confidence: {recent_positives} pops)"
                )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for ONNX inference."""
    parser = argparse.ArgumentParser(description="Coffee first crack ONNX inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=Path, help="Path to audio file")
    group.add_argument("--microphone", action="store_true", help="Use live microphone")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=_DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {_DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=_DEFAULT_SUBFOLDER,
        help=f"HF repo subfolder (default: {_DEFAULT_SUBFOLDER})",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="inference",
        choices=["inference", "pi_inference"],
        help="Config profile to use (default: inference)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="ONNX Runtime intra-op threads (default: from profile or platform default)",
    )
    parser.add_argument("--device-index", type=int, default=None, help="Microphone device index")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold override",
    )
    args = parser.parse_args()

    if args.microphone:
        detector = OnnxFirstCrackDetector(
            use_microphone=True,
            device_index=args.device_index,
            repo_id=args.repo_id,
            subfolder=args.subfolder,
            threshold=args.threshold,
            threads=args.threads,
            profile=args.profile,
        )
    else:
        if not args.audio.exists():
            print(f"Error: audio file not found: {args.audio}")
            sys.exit(1)
        detector = OnnxFirstCrackDetector(
            audio_file=args.audio,
            repo_id=args.repo_id,
            subfolder=args.subfolder,
            threshold=args.threshold,
            threads=args.threads,
            profile=args.profile,
        )

    detector.start()
    try:
        while detector.is_running:
            result = detector.is_first_crack()
            elapsed = detector.get_elapsed_time()
            if result is False:
                print(f"[{elapsed}] Listening...")
            elif isinstance(result, tuple):
                _, ts = result
                print(f"[{elapsed}] First crack at {ts}!")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
