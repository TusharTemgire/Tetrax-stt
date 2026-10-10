import io
import os
import time
import logging
import tempfile
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from tetrax_stt.audio import load_and_preprocess_audio

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

ROOT = Path(__file__).resolve().parents[2]

def resolve_model_path():
    env_model = os.getenv("TETRAX_STT_MODEL")
    candidates = []
    if env_model:
        candidates.append(Path(env_model))
    candidates.extend([
        ROOT / "models" / "tetrax-stt-v1",
        ROOT / "tetrax-stt-v1",
        Path("models/tetrax-stt-v1"),
        Path("tetrax-stt-v1"),
    ])
    for p in candidates:
        if p.exists() and (p / "config.json").exists():
            return str(p)
    return FALLBACK_MODEL

MODEL_DIR = resolve_model_path()

def get_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        target_model = resolve_model_path()
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

        audio_array, sr = load_and_preprocess_audio(temp_path)
        pipeline_inst = get_pipeline()
        result = pipeline_inst(
            {"raw": audio_array, "sampling_rate": sr},
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


@app.websocket("/v1/audio/stream")
async def websocket_stream(websocket: WebSocket, language: str = "hi"):
    """
    WebSocket endpoint for real-time live meeting transcription.
    Receives continuous PCM float32 or raw audio chunk bytes and sends live text captions back.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected for live meeting streaming (language: '{language}').")
    pipeline_inst = get_pipeline()
    lang_key = language.strip().lower()
    whisper_lang = LANGUAGE_MAP.get(lang_key, "hindi")

    audio_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Interpret incoming raw PCM 16kHz float32 bytes from browser MediaRecorder / AudioContext
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, chunk))

            # Transcribe when buffer reaches at least ~1.5 seconds of audio (24,000 samples @ 16kHz)
            if len(audio_buffer) >= 24000:
                result = pipeline_inst(
                    {"raw": audio_buffer, "sampling_rate": 16000},
                    generate_kwargs={"language": whisper_lang, "task": "transcribe"}
                )
                text = result.get("text", "").strip()

                if text:
                    await websocket.send_json({
                        "event": "transcription",
                        "text": text,
                        "language": lang_key,
                        "duration_sec": round(len(audio_buffer) / 16000.0, 2)
                    })

                # Maintain a 0.5s sliding context overlap window (8,000 samples)
                audio_buffer = audio_buffer[-8000:]

    except WebSocketDisconnect:
        logger.info("WebSocket live meeting client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {str(e)}")


@app.get("/demo", response_class=HTMLResponse)
def live_meeting_demo():
    """
    Real-time Live Meeting Captioning UI for Hard-of-Hearing Users & Live Transcripts.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tetrax Live Meeting Transcribe</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-green: #22c55e;
            --text-color: #f8fafc;
            --muted-text: #94a3b8;
        }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            width: 100%;
        }
        header {
            text-align: center;
            margin-bottom: 24px;
        }
        h1 {
            color: var(--accent);
            margin-bottom: 8px;
            font-size: 2.2rem;
        }
        p.subtitle {
            color: var(--muted-text);
            font-size: 1.1rem;
        }
        .controls {
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            display: flex;
            gap: 16px;
            align-items: center;
            justify-content: center;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
        }
        select, button {
            padding: 12px 20px;
            font-size: 1rem;
            border-radius: 8px;
            border: none;
            outline: none;
            cursor: pointer;
            font-weight: 600;
        }
        select {
            background: #334155;
            color: var(--text-color);
        }
        button.start {
            background: var(--accent-green);
            color: #000;
            transition: transform 0.2s, background 0.2s;
        }
        button.stop {
            background: #ef4444;
            color: #fff;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            background: #334155;
            color: var(--muted-text);
        }
        .status-badge.recording {
            background: rgba(34, 197, 94, 0.2);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }
        .transcript-box {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            min-height: 350px;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #334155;
            font-size: 1.4rem;
            line-height: 1.8;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }
        .live-text {
            color: var(--accent);
            font-weight: 500;
        }
        .history {
            margin-top: 12px;
            color: var(--text-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ Tetrax Real-Time Meeting Live Captioning</h1>
            <p class="subtitle">Instant accessibility & subtitles for ongoing office meetings</p>
        </header>

        <div class="controls">
            <label for="languageSelect">Meeting Language:</label>
            <select id="languageSelect">
                <option value="hi">Hindi (हिंदी)</option>
                <option value="mr">Marathi (मराठी)</option>
                <option value="en">English</option>
            </select>

            <button id="toggleBtn" class="start" onclick="toggleRecording()">▶️ Start Live Transcribe</button>

            <span id="statusBadge" class="status-badge">Disconnected</span>
        </div>

        <div class="transcript-box" id="transcriptBox">
            <div style="color: var(--muted-text); text-align: center; margin-top: 120px;">
                Click <strong>Start Live Transcribe</strong> and speak into your microphone...
            </div>
        </div>
    </div>

    <script>
        let isRecording = false;
        let ws = null;
        let audioContext = null;
        let processor = null;
        let micStream = null;

        async function toggleRecording() {
            const btn = document.getElementById('toggleBtn');
            const badge = document.getElementById('statusBadge');
            const lang = document.getElementById('languageSelect').value;
            const box = document.getElementById('transcriptBox');

            if (!isRecording) {
                try {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/v1/audio/stream?language=${lang}`);

                    ws.onopen = () => {
                        badge.innerText = '● Live Recording';
                        badge.classList.add('recording');
                        btn.innerText = '⏹️ Stop Transcribe';
                        btn.className = 'stop';
                        box.innerHTML = '<div class="history" id="historyText"></div><span class="live-text" id="liveText">Listening...</span>';
                    };

                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.event === 'transcription' && data.text) {
                            const historyText = document.getElementById('historyText');
                            historyText.innerText += (historyText.innerText ? ' ' : '') + data.text;
                            document.getElementById('liveText').innerText = '';
                            box.scrollTop = box.scrollHeight;
                        }
                    };

                    ws.onclose = () => stopRecordingState();

                    // Start Web Audio API recording
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(micStream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);

                    processor.onaudioprocess = (e) => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            const inputData = e.inputBuffer.getChannelData(0);
                            ws.send(inputData.buffer);
                        }
                    };

                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    isRecording = true;
                } catch (err) {
                    alert('Microphone access or WebSocket connection failed: ' + err.message);
                    stopRecordingState();
                }
            } else {
                stopRecordingState();
            }
        }

        function stopRecordingState() {
            isRecording = false;
            if (processor) { processor.disconnect(); processor = null; }
            if (audioContext) { audioContext.close(); audioContext = null; }
            if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
            if (ws) { ws.close(); ws = null; }

            const btn = document.getElementById('toggleBtn');
            const badge = document.getElementById('statusBadge');
            badge.innerText = 'Stopped';
            badge.classList.remove('recording');
            btn.innerText = '▶️ Start Live Transcribe';
            btn.className = 'start';
        }
    </script>
</body>
</html>"""

