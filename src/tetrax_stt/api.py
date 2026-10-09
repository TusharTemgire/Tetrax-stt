import os
import time
import logging
import tempfile
from pathlib import Path
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tetrax_stt_api")

app = FastAPI(
    title="Tetrax STT API",
    description="Multilingual Speech-to-Text API powered by Fine-Tuned Whisper Small (Hindi, Marathi, English)",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.getenv("TETRAX_STT_MODEL", "models/tetrax-stt-v1")
FALLBACK_MODEL = "openai/whisper-small"

LANGUAGE_MAP = {
    "hi": "hindi",
    "mr": "marathi",
    "en": "english",
    "hi_in": "hindi",
    "mr_in": "marathi",
    "en_us": "english",
}

device = 0 if torch.cuda.is_available() else -1
asr_pipeline = None

def get_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        target_model = MODEL_DIR if Path(MODEL_DIR).exists() else FALLBACK_MODEL
        logger.info(f"Initializing STT ASR pipeline using model: '{target_model}' (Device CUDA: {torch.cuda.is_available()})")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=target_model,
            device=device,
            chunk_length_s=30,
        )
        logger.info("STT ASR Pipeline loaded successfully!")
    return asr_pipeline

MAX_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

@app.get("/health")
def health():
    current_model = MODEL_DIR if Path(MODEL_DIR).exists() else FALLBACK_MODEL
    logger.info(f"Health check invoked. Active model: {current_model}")
    return {
        "status": "ok",
        "active_model": current_model,
        "cuda_available": torch.cuda.is_available(),
        "supported_languages": list(LANGUAGE_MAP.keys())
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("hi"),
):
    start_time = time.time()
    lang_key = language.strip().lower()

    logger.info(f"Received transcription request. Filename: '{file.filename}', Language: '{lang_key}'")

    if lang_key not in LANGUAGE_MAP:
        logger.warning(f"Rejected request: Unsupported language code '{language}'")
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language code '{language}'. Supported languages: {list(LANGUAGE_MAP.keys())}",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        logger.warning(f"Rejected request: Unsupported file extension '{suffix}'")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio file extension '{suffix}'. Allowed extensions: {list(ALLOWED_SUFFIXES)}",
        )

    data = await file.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        logger.warning(f"Rejected request: Invalid file size ({len(data) if data else 0} bytes)")
        raise HTTPException(
            status_code=413,
            detail="Audio file must be non-empty and under 25 MB limit",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(data)
            temp_path = temp.name

        pipeline_inst = get_pipeline()
        result = pipeline_inst(
            temp_path,
            generate_kwargs={
                "language": LANGUAGE_MAP[lang_key],
                "task": "transcribe",
            }
        )

        elapsed = round(time.time() - start_time, 3)
        current_model = MODEL_DIR if Path(MODEL_DIR).exists() else FALLBACK_MODEL
        logger.info(f"Transcription successful in {elapsed}s using model '{current_model}'. Result length: {len(result['text'])} chars")

        return {
            "text": result["text"].strip(),
            "language": lang_key,
            "model": current_model,
            "processing_time_sec": elapsed
        }

    except Exception as e:
        logger.error(f"Error during audio processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Audio processing error: {str(e)}",
        )

    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
