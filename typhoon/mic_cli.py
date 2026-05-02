from __future__ import annotations

import argparse
import io
import logging
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import requests
import sounddevice as sd
import webrtcvad

try:
    from .asr_engine import SAMPLE_WIDTH_BYTES, TARGET_SAMPLE_RATE
except ImportError:  # Allows: python typhoon/mic_cli.py
    from asr_engine import SAMPLE_WIDTH_BYTES, TARGET_SAMPLE_RATE


FRAME_MS = 30
DEFAULT_VAD_MODE = 2
DEFAULT_SILENCE_LIMIT_MS = 400
DEFAULT_MIN_SPEECH_MS = 300
DEFAULT_AUDIO_QUEUE_FRAMES = 400
DEFAULT_REQUEST_QUEUE_SIZE = 4
DEFAULT_SERVICE_URL = "http://127.0.0.1:8001/transcribe"
DEFAULT_REQUEST_TIMEOUT_SEC = 30.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UtteranceJob:
    chunks: list[bytes]
    sample_rate: int
    vad_end_time: float


def service_worker(
    jobs: "queue.Queue[UtteranceJob | None]",
    stop_event: threading.Event,
    service_url: str,
    request_timeout_sec: float,
) -> None:
    session = requests.Session()

    try:
        while not stop_event.is_set() or not jobs.empty():
            try:
                job = jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                jobs.task_done()
                break

            try:
                text, metrics_line = send_utterance_to_service(
                    session=session,
                    service_url=service_url,
                    request_timeout_sec=request_timeout_sec,
                    job=job,
                )
                if text:
                    print(f"USER: {text}", flush=True)
                    print(metrics_line, flush=True)
            except Exception:
                logger.exception("Failed to transcribe via service")
            finally:
                jobs.task_done()
    finally:
        session.close()


def enqueue_utterance(jobs: "queue.Queue[UtteranceJob | None]", job: UtteranceJob) -> None:
    try:
        jobs.put_nowait(job)
        return
    except queue.Full:
        logger.warning("Request queue is full; dropping the oldest pending utterance")

    try:
        dropped = jobs.get_nowait()
        if dropped is not None:
            jobs.task_done()
    except queue.Empty:
        pass

    try:
        jobs.put_nowait(job)
    except queue.Full:
        logger.warning("Request queue is still full; dropping current utterance")


def send_utterance_to_service(
    session: requests.Session,
    service_url: str,
    request_timeout_sec: float,
    job: UtteranceJob,
) -> tuple[str, str]:
    wav_bytes = pcm_chunks_to_wav_bytes(job.chunks, sample_rate=job.sample_rate)
    files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}

    response = session.post(service_url, files=files, timeout=request_timeout_sec)
    response.raise_for_status()

    payload = response.json()
    done = time.perf_counter()
    total_after_speech_end_ms = (done - job.vad_end_time) * 1000.0

    text = str(payload.get("text") or "").strip()
    metrics = payload.get("metrics") or {}
    audio_duration_sec = float(metrics.get("audio_duration_sec") or audio_duration_from_chunks(job.chunks, job.sample_rate))
    asr_processing_time_ms = metrics.get("asr_processing_time_ms")

    parts = [
        f"vad_end_to_asr_done={total_after_speech_end_ms:.0f}ms",
        f"audio={audio_duration_sec:.2f}s",
    ]
    if asr_processing_time_ms is not None:
        parts.append(f"asr_processing_time={float(asr_processing_time_ms):.0f}ms")
    metrics_line = "metrics: " + ", ".join(parts)
    return text, metrics_line


def pcm_chunks_to_wav_bytes(chunks: list[bytes], sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(chunks))
    return buffer.getvalue()


def audio_duration_from_chunks(chunks: list[bytes], sample_rate: int) -> float:
    total_bytes = sum(len(chunk) for chunk in chunks)
    return total_bytes / float(sample_rate * SAMPLE_WIDTH_BYTES)


def build_health_url(service_url: str) -> str:
    parsed = urlsplit(service_url)
    health_path = "/health"
    return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def verify_service_ready(service_url: str, request_timeout_sec: float) -> None:
    health_url = build_health_url(service_url)
    try:
        response = requests.get(health_url, timeout=min(request_timeout_sec, 5.0))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"ASR service is not reachable at {health_url}. Start uvicorn first."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Typhoon ASR microphone client")
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    parser.add_argument("--request-timeout-sec", type=float, default=DEFAULT_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=FRAME_MS)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=DEFAULT_VAD_MODE)
    parser.add_argument("--silence-limit-ms", type=int, default=DEFAULT_SILENCE_LIMIT_MS)
    parser.add_argument("--min-speech-ms", type=int, default=DEFAULT_MIN_SPEECH_MS)
    parser.add_argument("--audio-queue-frames", type=int, default=DEFAULT_AUDIO_QUEUE_FRAMES)
    parser.add_argument("--request-queue-size", type=int, default=DEFAULT_REQUEST_QUEUE_SIZE)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_devices:
        print(sd.query_devices())
        return 0

    verify_service_ready(args.service_url, args.request_timeout_sec)

    frame_samples = int(args.sample_rate * args.frame_ms / 1000)
    vad = webrtcvad.Vad(args.vad_mode)
    audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=args.audio_queue_frames)
    request_jobs: "queue.Queue[UtteranceJob | None]" = queue.Queue(maxsize=args.request_queue_size)
    stop_event = threading.Event()
    stats = {"audio_dropped": 0, "callback_status": 0}

    worker = threading.Thread(
        target=service_worker,
        args=(request_jobs, stop_event, args.service_url, args.request_timeout_sec),
        daemon=True,
    )
    worker.start()

    def audio_callback(indata: bytes, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            stats["callback_status"] += 1
        pcm = bytes(indata)
        try:
            audio_queue.put_nowait(pcm)
        except queue.Full:
            stats["audio_dropped"] += 1
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_queue.put_nowait(pcm)
            except queue.Full:
                pass

    logger.info("Connected to %s", args.service_url)
    logger.info("Listening. Press Ctrl+C to stop the microphone client.")

    speech_buf: list[bytes] = []
    silence_ms = 0
    speech_ms = 0
    speaking = False
    last_overflow_log = 0.0

    try:
        with sd.RawInputStream(
            channels=1,
            samplerate=args.sample_rate,
            blocksize=frame_samples,
            dtype="int16",
            callback=audio_callback,
            device=args.device_index,
            latency="low",
        ):
            while True:
                pcm = audio_queue.get()

                if stats["audio_dropped"] or stats["callback_status"]:
                    now = time.monotonic()
                    if now - last_overflow_log > 5:
                        logger.warning(
                            "Audio callback pressure: dropped_frames=%s callback_status_events=%s",
                            stats["audio_dropped"],
                            stats["callback_status"],
                        )
                        last_overflow_log = now

                is_speech = vad.is_speech(pcm, args.sample_rate)
                if is_speech:
                    speaking = True
                    silence_ms = 0
                    speech_ms += args.frame_ms
                    speech_buf.append(pcm)
                    continue

                if speaking:
                    silence_ms += args.frame_ms
                    speech_buf.append(pcm)

                if speaking and silence_ms >= args.silence_limit_ms:
                    if speech_ms >= args.min_speech_ms:
                        enqueue_utterance(
                            request_jobs,
                            UtteranceJob(
                                chunks=speech_buf.copy(),
                                sample_rate=args.sample_rate,
                                vad_end_time=time.perf_counter(),
                            ),
                        )

                    speech_buf.clear()
                    silence_ms = 0
                    speech_ms = 0
                    speaking = False

    except KeyboardInterrupt:
        logger.info("Stopping")
    except Exception:
        logger.exception("Microphone loop failed")
        return 1
    finally:
        stop_event.set()
        try:
            request_jobs.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
