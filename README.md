# Tetrax STT — Multilingual Speech-to-Text (Phase 1)

Multilingual Speech-to-Text pipeline fine-tuned on OpenAI **Whisper Small** for:
- 🇮🇳 **Hindi** (`hi_in` / `hi`)
- 🚩 **Marathi** (`mr_in` / `mr`)
- 🇺🇸 **English** (`en_us` / `en`)

Using the **Google FLEURS** dataset (`google/fleurs`).

---

## 📘 Quick Guides
- ⚡ **Google Colab Training Guide**: [`COLAB_GUIDE.md`](file:///d:/Tushar_repos/Tetrax-CPaaS/tetrax-stt/COLAB_GUIDE.md) | Notebook: [`tetrax_stt_colab_training.ipynb`](file:///d:/Tushar_repos/Tetrax-CPaaS/tetrax-stt/tetrax_stt_colab_training.ipynb)
- ☁️ **RunPod GPU Server Guide**: [`RUNPOD_GUIDE.md`](file:///d:/Tushar_repos/Tetrax-CPaaS/tetrax-stt/RUNPOD_GUIDE.md)

---

## 📂 Repository Structure

```
tetrax-stt/
├── configs/
│   └── languages.yaml             # Target language configurations (hi, mr, en)
├── data/
│   ├── raw/                       # Manual raw FLEURS downloads go here
│   │   ├── hi_in/
│   │   ├── mr_in/
│   │   └── en_us/
│   ├── audio/                     # Extracted 16kHz WAV audio files (gitignored)
│   └── manifests/                 # CSV manifests (train.csv, validation.csv, test.csv)
├── scripts/
│   ├── check_environment.py       # GPU & dependencies diagnostic
│   ├── download_fleurs_data.py    # Automatic dataset downloader via HuggingFace
│   ├── prepare_local_fleurs.py    # Local tar.gz & tsv dataset extractor
│   ├── prepare_dataset.py         # Manifest validation script
│   ├── train.py                   # Whisper Small fine-tuning script
│   └── evaluate.py                # Per-language Word Error Rate (WER) evaluation
├── src/
│   └── tetrax_stt/
│       ├── __init__.py
│       ├── audio.py               # Audio preprocessing (mono conversion, 16kHz resampling)
│       ├── model.py               # Model inference wrapper
│       └── api.py                 # FastAPI REST API server
├── models/                        # Saved fine-tuned model checkpoints (gitignored)
├── checkpoints/                   # Training step checkpoints (gitignored)
├── reports/                       # Per-language evaluation output reports
│   └── wer_by_language.csv
├── COLAB_GUIDE.md                 # Google Colab guide
├── RUNPOD_GUIDE.md                # RunPod cloud GPU guide
├── tetrax_stt_colab_training.ipynb# Google Colab Jupyter Notebook
├── requirements.txt               # Dependencies list
└── README.md                      # Guide documentation
```

---

## 🛠️ Step-by-Step Local & Cloud Setup Guide

### Step 1: Environment Setup & Diagnostic

1. **Create and activate Virtual Environment:**
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU / CUDA Environment:**
   ```bash
   python scripts/check_environment.py
   ```

---

### Step 2: Prepare Dataset

- **Automatic Download via HuggingFace:**
  ```bash
  python scripts/download_fleurs_data.py
  ```
- **Process Raw Files in `data/raw/`:**
  ```bash
  python scripts/prepare_local_fleurs.py
  ```
- **Validate Manifests:**
  ```bash
  python scripts/prepare_dataset.py
  ```

---

### Step 3: Fine-Tune Whisper Small Model

```bash
python scripts/train.py --max_steps 500 --batch_size 2 --grad_accum 8 --learning_rate 1e-5
```

---

### Step 4: Evaluate Model Accuracy (WER)

```bash
python scripts/evaluate.py
```

---

### Step 5: Launch FastAPI Production Server

```bash
uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000
```

#### Test Transcription Endpoint:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@data/audio/hi/test/hi_test_00000.wav" \
  -F "language=hi"
```

---

## 📄 Expected Log Outputs & Reference Examples

### 1. Expected Ideal Training Log Output (`scripts/train.py`):
```text
Loading processor and model: openai/whisper-small...
Loading CSV datasets...
Preprocessing audio features and language tokens...
Preparing audio features and language labels: 100%|██████████| 7907/7907 [00:12<00:00, 642.10it/s]
Starting fine-tuning process...

  0%|                                                    | 0/500 [00:00<?, ?it/s]
{'loss': 2.4105, 'learning_rate': 2.0000e-06, 'epoch': 0.02}
{'loss': 1.8421, 'learning_rate': 4.0000e-06, 'epoch': 0.04}
...
{'eval_loss': 0.3821, 'epoch': 0.25}
{'loss': 0.2910, 'learning_rate': 9.8000e-06, 'epoch': 1.00}
{'eval_loss': 0.2140, 'epoch': 1.00}

Saving final model to: D:\Tushar_repos\Tetrax-CPaaS\tetrax-stt\models\tetrax-stt-v1
Fine-tuning completed successfully!
```

---

### 2. Expected API Server Logs & Error Handling (`tetrax_stt/api.py`):

#### Successful Transcription Log (`200 OK`):
```text
2026-10-09 13:15:00 [INFO] tetrax_stt_api: Received transcription request. Filename: 'sample.wav', Language: 'hi'
2026-10-09 13:15:00 [INFO] tetrax_stt_api: Initializing STT ASR pipeline using model: 'models/tetrax-stt-v1' (Device CUDA: True)
2026-10-09 13:15:01 [INFO] tetrax_stt_api: STT ASR Pipeline loaded successfully!
2026-10-09 13:15:02 [INFO] tetrax_stt_api: Transcription successful in 1.42s using model 'models/tetrax-stt-v1'. Result length: 42 chars
INFO:     127.0.0.1:54321 - "POST /v1/audio/transcriptions HTTP/1.1" 200 OK
```

#### Error Example A: Unsupported Language (`422 Unprocessable Entity`)
```bash
# Request:
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=fr"

# Server Log:
2026-10-09 13:16:00 [WARNING] tetrax_stt_api: Rejected request: Unsupported language code 'fr'

# JSON Response:
{
  "detail": "Unsupported language code 'fr'. Supported languages: ['hi', 'mr', 'en', 'hi_in', 'mr_in', 'en_us']"
}
```

#### Error Example B: Unsupported Extension (`415 Unsupported Media Type`)
```bash
# Request with unsupported .txt extension:
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.txt" -F "language=hi"

# Server Log:
2026-10-09 13:16:05 [WARNING] tetrax_stt_api: Rejected request: Unsupported file extension '.txt'

# JSON Response:
{
  "detail": "Unsupported audio file extension '.txt'. Allowed extensions: ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm']"
}
```

#### Error Example C: File Size Exceeds Limit (`413 Payload Too Large`)
```bash
# Server Log:
2026-10-09 13:16:10 [WARNING] tetrax_stt_api: Rejected request: Invalid file size (28314572 bytes)

# JSON Response:
{
  "detail": "Audio file must be non-empty and under 25 MB limit"
}
```

