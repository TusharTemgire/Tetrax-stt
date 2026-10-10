# 🎙️ Tetrax STT v1 — Model Usage Guide

`tetrax-stt-v1` is a fine-tuned OpenAI Whisper Small model trained for multilingual Speech-to-Text transcription targeting **Hindi (`hi`)**, **Marathi (`mr`)**, and **English (`en`)**.

Model Location: `models/tetrax-stt-v1`

---

## ⚡ Quick Start Options

### 1. Python Inference via `TetraxSTTModel` Wrapper

The `tetrax_stt` package includes a built-in model wrapper that automatically detects `models/tetrax-stt-v1`.

```python
import sys
sys.path.insert(0, "src")

from tetrax_stt.model import TetraxSTTModel

# Automatically loads 'models/tetrax-stt-v1' if present, or GPU/CPU device
model = TetraxSTTModel()

# Transcribe audio file (Hindi example)
result = model.transcribe(
    audio_input="data/audio/hi/test/sample.wav",
    language="hi"  # Options: 'hi', 'mr', 'en'
)

print("Transcribed Text:", result["text"])
print("Language:", result["language"])
```

---

### 2. Python Inference via Hugging Face `pipeline`

You can also load the model using standard `transformers`:

```python
import torch
from transformers import pipeline

model_path = "models/tetrax-stt-v1"
device = 0 if torch.cuda.is_available() else -1

# Load pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model_path,
    device=device,
    chunk_length_s=30
)

# Run transcription
audio_file = "path/to/your/audio.wav"
language_code = "hindi"  # Options: "hindi", "marathi", "english"

result = asr_pipeline(
    audio_file,
    generate_kwargs={
        "language": language_code,
        "task": "transcribe"
    }
)

print("Transcription:", result["text"])
```

---

### 3. Launching the Production REST API Server

Start the FastAPI server powered by Uvicorn:

```bash
# Windows / Linux / macOS
uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

Server endpoints will be active at: `http://localhost:8000`

- **Health check**: `GET http://localhost:8000/health`
- **Transcription endpoint**: `POST http://localhost:8000/v1/audio/transcriptions`

---

### 4. API Request Examples

#### cURL (Command Line):
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample_hindi.wav" \
  -F "language=hi"
```

#### Python (`requests`):
```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": ("audio.wav", open("sample_hindi.wav", "rb"), "audio/wav")}
data = {"language": "hi"}

response = requests.post(url, files=files, data=data)
print(response.json())
```
**Sample JSON Response:**
```json
{
  "text": "नमस्ते यह एक परीक्षण संदेश है",
  "language": "hi",
  "model": "models/tetrax-stt-v1",
  "processing_time_sec": 0.842
}
```

#### JavaScript / Frontend Integration:
```javascript
const formData = new FormData();
formData.append("file", audioBlob, "recording.wav");
formData.append("language", "hi"); // 'hi', 'mr', 'en'

const response = await fetch("http://localhost:8000/v1/audio/transcriptions", {
  method: "POST",
  body: formData,
});

const data = await response.json();
console.log("Transcribed output:", data.text);
```

---

## 📊 Evaluation & Benchmarking

To compute per-language Word Error Rate (WER) on the test dataset:

```bash
python scripts/evaluate.py
```

This generates a report in `reports/wer_by_language.csv`.

---

## 🛠️ Supported Specifications

| Parameter | Supported Values |
|---|---|
| **Target Languages** | `hi` (Hindi), `mr` (Marathi), `en` (English) |
| **Allowed File Formats** | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.webm` |
| **Max Audio File Size** | 25 MB |
| **Sample Rate** | 16 kHz (auto-resampled by pipeline) |
