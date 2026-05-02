from __future__ import annotations

import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

try:
    from .asr_engine import DEFAULT_MODEL_NAME, format_metrics, get_asr_engine
except ImportError:  # Allows: uvicorn asr_service:app from the typhoon directory
    from asr_engine import DEFAULT_MODEL_NAME, format_metrics, get_asr_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_asr_engine(model_name=DEFAULT_MODEL_NAME, device="auto")
    yield


app = FastAPI(title="Typhoon ASR Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        data = await file.read()
        temp_path.write_bytes(data)

        engine = get_asr_engine()
        result = await run_in_threadpool(
            engine.transcribe_file,
            temp_path,
            time.perf_counter(),
        )
        logger.info(format_metrics(result.metrics))
        return {
            "text": result.text,
            "metrics": {
                "vad_end_to_asr_done_ms": result.metrics.vad_end_to_asr_done_ms,
                "asr_processing_time_ms": result.metrics.asr_processing_time_ms,
                "audio_duration_sec": result.metrics.audio_duration_sec,
                "total_after_speech_end_ms": result.metrics.total_after_speech_end_ms,
            },
        }
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", temp_path, exc)
