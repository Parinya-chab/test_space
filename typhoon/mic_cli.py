from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass

import sounddevice as sd
import webrtcvad

try:
    from .asr_engine import DEFAULT_MODEL_NAME, TARGET_SAMPLE_RATE, format_metrics, get_asr_engine
except ImportError:  # Allows: python typhoon/mic_cli.py
    from asr_engine import DEFAULT_MODEL_NAME, TARGET_SAMPLE_RATE, format_metrics, get_asr_engine


FRAME_MS = 30
DEFAULT_VAD_MODE = 2
DEFAULT_SILENCE_LIMIT_MS = 400
DEFAULT_MIN_SPEECH_MS = 300
DEFAULT_AUDIO_QUEUE_FRAMES = 400
DEFAULT_ASR_QUEUE_SIZE = 4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UtteranceJob:
    chunks: list[bytes]
    sample_rate: int
    vad_end_time: float


def asr_worker(
    jobs: "queue.Queue[UtteranceJob | None]",
    stop_event: threading.Event,
    engine_device: str,
    model_name: str,
) -> None:
    engine = get_asr_engine(model_name=model_name, device=engine_device)

    while not stop_event.is_set() or not jobs.empty():
        try:
            job = jobs.get(timeout=0.1)
        except queue.Empty:
            continue

        if job is None:
            jobs.task_done()
            break

        try:
            result = engine.transcribe_pcm16(
                job.chunks,
                sample_rate=job.sample_rate,
                vad_end_time=job.vad_end_time,
            )
            if result.text:
                print(f"USER: {result.text}", flush=True)
                print(format_metrics(result.metrics), flush=True)
        except Exception:
            logger.exception("ASR failed")
        finally:
            jobs.task_done()


def enqueue_utterance(jobs: "queue.Queue[UtteranceJob | None]", job: UtteranceJob) -> None:
    try:
        jobs.put_nowait(job)
        return
    except queue.Full:
        logger.warning("ASR queue is full; dropping the oldest pending utterance")

    try:
        dropped = jobs.get_nowait()
        if dropped is not None:
            jobs.task_done()
    except queue.Empty:
        pass

    try:
        jobs.put_nowait(job)
    except queue.Full:
        logger.warning("ASR queue is still full; dropping current utterance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Typhoon ASR microphone CLI")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=FRAME_MS)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=DEFAULT_VAD_MODE)
    parser.add_argument("--silence-limit-ms", type=int, default=DEFAULT_SILENCE_LIMIT_MS)
    parser.add_argument("--min-speech-ms", type=int, default=DEFAULT_MIN_SPEECH_MS)
    parser.add_argument("--audio-queue-frames", type=int, default=DEFAULT_AUDIO_QUEUE_FRAMES)
    parser.add_argument("--asr-queue-size", type=int, default=DEFAULT_ASR_QUEUE_SIZE)
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

    frame_samples = int(args.sample_rate * args.frame_ms / 1000)
    vad = webrtcvad.Vad(args.vad_mode)
    audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=args.audio_queue_frames)
    asr_jobs: "queue.Queue[UtteranceJob | None]" = queue.Queue(maxsize=args.asr_queue_size)
    stop_event = threading.Event()
    stats = {"audio_dropped": 0, "callback_status": 0}

    # Load before opening the mic so the first utterance does not pay model-load latency.
    get_asr_engine(model_name=args.model_name, device=args.device)

    worker = threading.Thread(
        target=asr_worker,
        args=(asr_jobs, stop_event, args.device, args.model_name),
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

    logger.info("Listening. Press Ctrl+C to stop.")

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
                            asr_jobs,
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
            asr_jobs.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=5)

    return 0


if __name__ == "__main__":
    sys.exit(main())

