from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# HF_HOME must be set before importing typhoon_asr, NeMo, torch, or HF helpers.
os.environ["HF_HOME"] = str(MODEL_DIR)

DEFAULT_MODEL_NAME = "scb10x/typhoon-asr-realtime"
TARGET_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2

logger = logging.getLogger(__name__)


class ASREngineError(RuntimeError):
    """Raised when the ASR engine cannot load or transcribe."""


@dataclass(frozen=True)
class TranscriptionMetrics:
    vad_end_to_asr_done_ms: float
    audio_duration_sec: float
    total_after_speech_end_ms: float
    asr_processing_time_ms: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    metrics: TranscriptionMetrics
    raw_result: Any


class ASREngine:
    """Singleton-friendly Typhoon ASR engine that keeps the model in memory."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.target_sample_rate = target_sample_rate
        self.device = "cpu"
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        self.load()

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return

            try:
                import torch
                import nemo.collections.asr as nemo_asr
            except Exception as exc:  # pragma: no cover - depends on local env
                raise ASREngineError(
                    "Cannot import Typhoon ASR dependencies. Activate the env that "
                    "has typhoon-asr, nemo-toolkit, and torch installed."
                ) from exc

            self.device = self._resolve_device(torch)
            logger.info("Loading Typhoon ASR model once on %s", self.device.upper())

            try:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name,
                    map_location=self.device,
                )
                if hasattr(model, "to"):
                    model = model.to(self.device)
                if hasattr(model, "eval"):
                    model.eval()
            except Exception as exc:  # pragma: no cover - model load is external
                raise ASREngineError(f"Failed to load ASR model: {exc}") from exc

            if model is None:
                raise ASREngineError("Failed to load ASR model: model is None")

            self._model = model

    def transcribe_pcm16(
        self,
        pcm: bytes | list[bytes] | tuple[bytes, ...],
        sample_rate: int = TARGET_SAMPLE_RATE,
        vad_end_time: float | None = None,
    ) -> TranscriptionResult:
        pcm_bytes = b"".join(pcm) if isinstance(pcm, (list, tuple)) else pcm
        if not pcm_bytes:
            return self._empty_result(vad_end_time)

        audio_duration_sec = len(pcm_bytes) / float(sample_rate * SAMPLE_WIDTH_BYTES)
        start_after_speech_end = vad_end_time or time.perf_counter()
        wav_path = self._write_pcm16_temp_wav(pcm_bytes, sample_rate)

        try:
            raw_result = self._transcribe_path(wav_path)
        finally:
            self._remove_temp_file(wav_path)

        done = time.perf_counter()
        return self._build_result(raw_result, start_after_speech_end, done, audio_duration_sec)

    def transcribe_file(
        self,
        audio_path: str | Path,
        vad_end_time: float | None = None,
    ) -> TranscriptionResult:
        start_after_speech_end = vad_end_time or time.perf_counter()
        prepared_path, audio_duration_sec, temp_path = self._prepare_audio_file(Path(audio_path))

        try:
            raw_result = self._transcribe_path(prepared_path)
        finally:
            if temp_path is not None:
                self._remove_temp_file(temp_path)

        done = time.perf_counter()
        return self._build_result(raw_result, start_after_speech_end, done, audio_duration_sec)

    def _resolve_device(self, torch: Any) -> str:
        requested = self.requested_device.lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is unavailable; falling back to CPU")
            return "cpu"
        if requested not in {"cpu", "cuda"}:
            raise ASREngineError("device must be one of: auto, cpu, cuda")
        return requested

    def _transcribe_path(self, wav_path: str | Path) -> Any:
        self.load()
        assert self._model is not None

        wav_path = str(wav_path)
        with self._transcribe_lock:
            try:
                return self._model.transcribe(audio=[wav_path])
            except TypeError:
                return self._model.transcribe([wav_path])

    def _write_pcm16_temp_wav(self, pcm_bytes: bytes, sample_rate: int) -> Path:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)

        return temp_path

    def _prepare_audio_file(self, input_path: Path) -> tuple[Path, float, Path | None]:
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        wav_duration = self._get_compatible_wav_duration(input_path)
        if wav_duration is not None:
            return input_path, wav_duration, None

        try:
            import librosa
            import numpy as np
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - depends on local env
            raise ASREngineError(
                "Non-16k mono PCM WAV input needs librosa and soundfile installed."
            ) from exc

        y, sr = librosa.load(str(input_path), sr=None, mono=True)
        if y is None or sr is None:
            raise ASREngineError(f"Failed to load audio file: {input_path}")

        audio_duration_sec = len(y) / float(sr) if sr else 0.0
        if sr != self.target_sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)

        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 0:
            y = y / peak

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        sf.write(str(temp_path), y, self.target_sample_rate, subtype="PCM_16")
        return temp_path, audio_duration_sec, temp_path

    def _get_compatible_wav_duration(self, input_path: Path) -> float | None:
        if input_path.suffix.lower() != ".wav":
            return None

        try:
            with wave.open(str(input_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
        except wave.Error:
            return None

        if channels != 1 or sample_width != SAMPLE_WIDTH_BYTES or sample_rate != self.target_sample_rate:
            return None

        return frames / float(sample_rate)

    def _build_result(
        self,
        raw_result: Any,
        start_after_speech_end: float,
        done: float,
        audio_duration_sec: float,
    ) -> TranscriptionResult:
        elapsed_ms = (done - start_after_speech_end) * 1000.0
        processing_time_sec = extract_processing_time_sec(raw_result)

        metrics = TranscriptionMetrics(
            vad_end_to_asr_done_ms=elapsed_ms,
            asr_processing_time_ms=(
                processing_time_sec * 1000.0 if processing_time_sec is not None else None
            ),
            audio_duration_sec=audio_duration_sec,
            total_after_speech_end_ms=elapsed_ms,
        )
        return TranscriptionResult(text=extract_text(raw_result), metrics=metrics, raw_result=raw_result)

    def _empty_result(self, vad_end_time: float | None) -> TranscriptionResult:
        now = time.perf_counter()
        start = vad_end_time or now
        metrics = TranscriptionMetrics(
            vad_end_to_asr_done_ms=(now - start) * 1000.0,
            audio_duration_sec=0.0,
            total_after_speech_end_ms=(now - start) * 1000.0,
        )
        return TranscriptionResult(text="", metrics=metrics, raw_result=None)

    def _remove_temp_file(self, path: str | Path) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", path, exc)


_engine: ASREngine | None = None
_engine_lock = threading.Lock()


def get_asr_engine(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
) -> ASREngine:
    global _engine

    with _engine_lock:
        if _engine is None:
            _engine = ASREngine(model_name=model_name, device=device)
        elif _engine.model_name != model_name or _engine.requested_device != device:
            logger.warning(
                "ASR engine is already loaded; ignoring new model/device request "
                "(model=%s, device=%s)",
                model_name,
                device,
            )

        return _engine


def extract_text(value: Any) -> str:
    return _extract_text(value, seen=set()).strip()


def _extract_text(value: Any, seen: set[int]) -> str:
    if value is None:
        return ""

    value_id = id(value)
    if value_id in seen:
        return ""
    seen.add(value_id)

    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        for key in ("text", "transcription", "transcript"):
            if key in value:
                return _extract_text(value[key], seen)
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _extract_text(item, seen)
            if text:
                return text
        return ""
    if hasattr(value, "text"):
        return _extract_text(getattr(value, "text"), seen)

    text = str(value)
    if text.startswith("<") and text.endswith(">"):
        return ""
    return text


def extract_processing_time_sec(value: Any) -> float | None:
    if isinstance(value, dict) and "processing_time" in value:
        return _to_float(value.get("processing_time"))
    if hasattr(value, "processing_time"):
        return _to_float(getattr(value, "processing_time"))
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metrics(metrics: TranscriptionMetrics) -> str:
    parts = [
        f"vad_end_to_asr_done={metrics.vad_end_to_asr_done_ms:.0f}ms",
        f"audio={metrics.audio_duration_sec:.2f}s",
    ]
    if metrics.asr_processing_time_ms is not None:
        parts.append(f"asr_processing_time={metrics.asr_processing_time_ms:.0f}ms")
    parts.append(f"total_after_speech_end={metrics.total_after_speech_end_ms:.0f}ms")
    return "metrics: " + ", ".join(parts)

