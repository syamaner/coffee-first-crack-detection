"""Inference module: sliding window and live microphone first crack detection.

Two main classes:
- ``SlidingWindowInference`` — offline processing of audio files
- ``FirstCrackDetector`` — real-time streaming (file or microphone) with
  thread-safe pop-confirmation logic

Usage::

    # File-based sliding window
    python -m coffee_first_crack.inference \\
        --audio data/raw/mic2-brazil-roast1.wav \\
        --model-dir syamaner/coffee-first-crack-detection

    # Live microphone
    python -m coffee_first_crack.inference --microphone \\
        --model-dir syamaner/coffee-first-crack-detection

    # List audio devices
    python -c "import sounddevice as sd; print(sd.query_devices())"
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch

from coffee_first_crack.model import FirstCrackClassifier
from coffee_first_crack.utils.device import get_device


# ── Detection event ───────────────────────────────────────────────────────────


@dataclass
class DetectionEvent:
    """A confirmed first-crack detection event.

    Attributes:
        timestamp_sec: Time (in seconds) of first confirmed pop.
        timestamp_str: Human-readable ``"MM:SS"`` string.
        confidence: Number of positive pops within the confirmation window.
    """

    timestamp_sec: float
    timestamp_str: str
    confidence: int


def _format_time(seconds: float) -> str:
    """Format seconds as ``MM:SS``."""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


# ── Sliding window (offline) ──────────────────────────────────────────────────


class SlidingWindowInference:
    """Process an audio file with a sliding window and detect first crack.

    Args:
        model_name_or_path: HuggingFace model ID or local ``save_pretrained`` dir.
        window_size: Window length in seconds (default: 10.0).
        overlap: Overlap fraction between consecutive windows (default: 0.7).
        threshold: Minimum first-crack probability to count as a pop (default: 0.6).
        min_pops: Number of positive windows required for confirmation (default: 5).
        confirmation_window: Time span in seconds over which pops are counted (default: 20.0).
        device: Target device. Defaults to auto-detected.
    """

    def __init__(
        self,
        model_name_or_path: str = "syamaner/coffee-first-crack-detection",
        window_size: float = 10.0,
        overlap: float = 0.7,
        threshold: float = 0.6,
        min_pops: int = 5,
        confirmation_window: float = 20.0,
        device: str | None = None,
    ) -> None:
        self.window_size = window_size
        self.overlap = overlap
        self.threshold = threshold
        self.min_pops = min_pops
        self.confirmation_window = confirmation_window
        self.device = device or get_device()
        self.sample_rate = 16000
        self.window_samples = int(window_size * self.sample_rate)
        self.hop_samples = int(self.window_samples * (1 - overlap))

        self._model = FirstCrackClassifier(
            model_name_or_path=model_name_or_path,
            device=self.device,
        )
        self._model.model.eval()

    def process_file(self, audio_path: Union[str, Path]) -> list[DetectionEvent]:
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
            recent_positives = sum(
                1 for t, pos, _ in history if t >= cutoff and pos
            )

            if not confirmed and recent_positives >= self.min_pops:
                confirmed = True
                first_t = next(
                    t for t, pos, _ in history if pos and t >= cutoff
                )
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

    @torch.inference_mode()
    def _predict_window(self, window: np.ndarray) -> float:
        """Predict first-crack probability for a single audio window.

        Applies an energy-based noise gate: silent windows return 0.0.
        """
        if np.sqrt(np.mean(window ** 2)) < 0.01:
            return 0.0
        tensor = torch.FloatTensor(window).unsqueeze(0)
        logits = self._model(tensor)
        return torch.softmax(logits, dim=-1)[0, 1].item()


# ── Live streaming detector ───────────────────────────────────────────────────


class FirstCrackDetector:
    """Real-time first crack detector for file or microphone input.

    Runs inference in a background thread. Call :meth:`start`, then poll
    :meth:`is_first_crack` in your roast loop, then :meth:`stop`.

    Args:
        audio_file: Path to WAV file (mutually exclusive with ``use_microphone``).
        use_microphone: Stream from the default (or specified) microphone.
        device_index: Sounddevice device index for microphone input.
        model_name_or_path: HuggingFace model ID or local checkpoint path.
        window_size: Window length in seconds.
        overlap: Overlap fraction.
        threshold: Positive classification threshold.
        sample_rate: Audio sample rate (must match model: 16 kHz).
        min_pops: Pops required within ``confirmation_window`` to confirm.
        confirmation_window: Time span in seconds for pop counting.
    """

    def __init__(
        self,
        audio_file: Optional[Union[str, Path]] = None,
        use_microphone: bool = False,
        device_index: Optional[int] = None,
        model_name_or_path: str = "syamaner/coffee-first-crack-detection",
        window_size: float = 10.0,
        overlap: float = 0.7,
        threshold: float = 0.6,
        sample_rate: int = 16000,
        min_pops: int = 5,
        confirmation_window: float = 20.0,
    ) -> None:
        if audio_file and use_microphone:
            raise ValueError("Specify either audio_file or use_microphone, not both.")
        if not audio_file and not use_microphone:
            raise ValueError("Must specify audio_file or use_microphone.")

        self.audio_file = Path(audio_file) if audio_file else None
        self.use_microphone = use_microphone
        self.device_index = device_index
        self.window_size = window_size
        self.overlap = overlap
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_pops = min_pops
        self.confirmation_window = confirmation_window

        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(self.window_samples * (1 - overlap))

        device = get_device()
        self._model = FirstCrackClassifier(
            model_name_or_path=model_name_or_path,
            device=device,
        )
        self._model.model.eval()

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._first_crack_detected = False
        self._first_crack_time: Optional[float] = None
        self._start_time: Optional[float] = None
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
        print("First crack detector started.")

    def stop(self) -> None:
        """Stop detection and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._audio_buffer.clear()
        self._detection_history.clear()
        print("First crack detector stopped.")

    def is_first_crack(self) -> Union[bool, tuple[bool, str]]:
        """Check detection state.

        Returns:
            ``False`` if not yet detected, or ``(True, "MM:SS")`` when confirmed.
        """
        with self._lock:
            if not self._first_crack_detected:
                return False
            return True, _format_time(self._first_crack_time or 0.0)

    def get_elapsed_time(self) -> Optional[str]:
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
            import sounddevice as sd  # lazy import — not needed on RPi5 without mic

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
                            window = np.array(list(self._audio_buffer)[-self.window_samples:])
                        current_time = time.time() - (self._start_time or time.time())
                        prob = self._predict_window(window)
                        self._update_state(prob, current_time)
                    time.sleep(0.5)
        except Exception as exc:
            print(f"Microphone error: {exc}")
            self._running = False

    @torch.inference_mode()
    def _predict_window(self, window: np.ndarray) -> float:
        """Predict first-crack probability with noise gate."""
        if np.sqrt(np.mean(window ** 2)) < 0.01:
            return 0.0
        tensor = torch.FloatTensor(window).unsqueeze(0)
        logits = self._model(tensor)
        return torch.softmax(logits, dim=-1)[0, 1].item()

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
                    f"FIRST CRACK at {_format_time(self._first_crack_time)} "
                    f"(confidence: {recent_positives} pops)"
                )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Coffee first crack inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=Path, help="Path to audio file")
    group.add_argument("--microphone", action="store_true", help="Use live microphone")
    parser.add_argument(
        "--model-dir", type=str, default="syamaner/coffee-first-crack-detection",
        help="HuggingFace model ID or local checkpoint directory",
    )
    parser.add_argument("--device-index", type=int, default=None, help="Microphone device index")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection threshold")
    args = parser.parse_args()

    if args.microphone:
        detector = FirstCrackDetector(
            use_microphone=True,
            device_index=args.device_index,
            model_name_or_path=args.model_dir,
            threshold=args.threshold,
        )
    else:
        if not args.audio.exists():
            print(f"Error: audio file not found: {args.audio}")
            sys.exit(1)
        detector = FirstCrackDetector(
            audio_file=args.audio,
            model_name_or_path=args.model_dir,
            threshold=args.threshold,
        )

    detector.start()
    try:
        while detector.is_running:
            result = detector.is_first_crack()
            elapsed = detector.get_elapsed_time()
            if result is False:
                print(f"[{elapsed}] Listening...")
            else:
                _, ts = result
                print(f"[{elapsed}] First crack at {ts}!")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
