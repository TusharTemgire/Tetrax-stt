# 🚀 Guide: Fine-Tuning Tetrax STT on Google Colab & Deploying to GPU Server

Google Colab provides free T4 GPUs (15 GB VRAM) for fine-tuning Whisper Small. After training, you can easily export your fine-tuned model and deploy it onto any cloud or private GPU server (e.g., RunPod, AWS EC2, GCP, Lambda Labs, or custom Linux GPU instances) to host a production-ready Speech-to-Text (STT) API.

---

## 📋 Step 1: Open Google Colab & Enable GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`tetrax_stt_colab_training.ipynb`](file:///d:/Tushar_repos/Tetrax-stt/tetrax_stt_colab_training.ipynb).
3. Go to **Runtime** ➔ **Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU**.
5. Click **Save**.

---

## 💻 Step 2: Run Fine-Tuning in Google Colab

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Dependencies
```bash
!pip install -q transformers datasets[audio] accelerate evaluate jiwer soundfile librosa pandas pyyaml fastapi uvicorn python-multipart
```

### 3. Check GPU Availability
```bash
!python scripts/check_environment.py
```

### 4. Prepare Dataset
```bash
!python scripts/download_fleurs_data.py
!python scripts/prepare_dataset.py
```

### 5. Fine-Tune Whisper Small Model
```bash
!python scripts/train.py --max_steps 500 --batch_size 4 --grad_accum 4 --learning_rate 1e-5
```

### 6. Evaluate WER Score
```bash
!python scripts/evaluate.py
```

### 7. Export Trained Model to Google Drive & Download Archive
```bash
# Save model folder directly to mounted Google Drive
!cp -r models/tetrax-stt-v1 /content/drive/MyDrive/tetrax-stt-v1

# Create a portable zip archive for easy server transfer
!zip -r tetrax-stt-v1.zip models/tetrax-stt-v1
!cp tetrax-stt-v1.zip /content/drive/MyDrive/tetrax-stt-v1.zip
```

---

## 🧪 Step 3: Run Quick Local Model Inference (in Colab or Local Machine)

```python
import sys
sys.path.append('src')
from tetrax_stt.model import TetraxSTTModel

# Initialize model from fine-tuned output directory
stt_model = TetraxSTTModel('models/tetrax-stt-v1')

# Transcribe sample audio file
result = stt_model.transcribe('data/audio/hi/test/hi_test_00000.wav', language='hi')
print("Transcript:", result['text'])
```

---

## 📦 Step 4: Transfer Trained Model Weights to External GPU Server

You can transfer your fine-tuned model (`tetrax-stt-v1.zip` or folder) from Google Colab / Drive to your destination GPU server using any of the options below:

### Option A: Direct Download via `gdown` (Recommended on Target GPU Server)
On your target GPU server terminal:
```bash
# Install gdown utility
pip install gdown

# Share your zip file in Google Drive as "Anyone with link" and copy FILE_ID
# URL format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
gdown https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID -O tetrax-stt-v1.zip

# Unzip into models directory
mkdir -p models
unzip tetrax-stt-v1.zip -d .
```

### Option B: Transfer via SCP / SFTP (From local machine to GPU server)
1. Download `tetrax-stt-v1.zip` from your Google Drive to your local machine.
2. Upload to your target server using `scp`:
```bash
scp -P 22 tetrax-stt-v1.zip user@YOUR_GPU_SERVER_IP:/workspace/tetrax-stt/
```
3. On the GPU server, extract the weights:
```bash
cd /workspace/tetrax-stt
unzip tetrax-stt-v1.zip
```

### Option C: Load directly from Hugging Face Hub (If uploaded)
If you pushed your fine-tuned model to Hugging Face Hub from Colab (`model.push_to_hub("your-org/tetrax-stt-v1")`), set environment variable on the server:
```bash
export TETRAX_STT_MODEL="your-org/tetrax-stt-v1"
```

---

## 🖥️ Step 5: Set Up External GPU Server Environment

Log into your external GPU server (e.g., RunPod, AWS EC2 `g4dn`/`g5` instance, GCP, Lambda Labs, or custom Linux GPU server).

### 1. Verify CUDA & GPU Drivers
```bash
nvidia-smi
```

### 2. Clone Repository & Setup Virtual Environment
```bash
# Navigate to working directory
cd /workspace   # or /var/www or ~

# Clone repository
git clone <YOUR_REPOSITORY_URL> tetrax-stt
cd tetrax-stt

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Model Directory Structure
Ensure your fine-tuned model weights are placed inside `models/tetrax-stt-v1`:
```text
models/tetrax-stt-v1/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── preprocessor_config.json
├── tokenizer.json
└── generation_config.json
```

---

## 🚀 Step 6: Deploy & Host STT API Server on GPU Server

### 1. Launch FastAPI Server using Uvicorn
```bash
# Set path to fine-tuned model (defaults to models/tetrax-stt-v1 if omitted)
export TETRAX_STT_MODEL="models/tetrax-stt-v1"

# Run Uvicorn production server on port 8000
uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000 --workers 1
```

### 2. Run as a Persistent Service (nohup or Systemd)

#### Option A: Background Execution using `nohup`
```bash
nohup uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

#### Option B: Systemd Service (For Ubuntu / Debian Linux Servers)
Create `/etc/systemd/system/tetrax-stt.service`:
```ini
[Unit]
Description=Tetrax STT FastAPI Server
After=network.target nvidia-persistenced.service

[Service]
User=root
WorkingDirectory=/workspace/tetrax-stt
Environment="PATH=/workspace/tetrax-stt/venv/bin"
Environment="TETRAX_STT_MODEL=models/tetrax-stt-v1"
ExecStart=/workspace/tetrax-stt/venv/bin/uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Enable & start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tetrax-stt
sudo systemctl start tetrax-stt
```

---

## 🌐 Step 7: Consume & Integrate STT API Service

Once the server is running on `http://<GPU_SERVER_IP>:8000`, test and integrate it into client applications:

### 1. Verify Health Check
```bash
curl http://<GPU_SERVER_IP>:8000/health
```
**Expected Response:**
```json
{
  "status": "ok",
  "active_model": "models/tetrax-stt-v1",
  "cuda_available": true,
  "supported_languages": ["hi", "mr", "en", "hi_in", "mr_in", "en_us"]
}
```

### 2. Perform Transcription via cURL
```bash
curl -X POST "http://<GPU_SERVER_IP>:8000/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_audio.wav" \
  -F "language=hi"
```
**Expected Response:**
```json
{
  "text": "नमस्ते भारत, यह स्पीच टू टेक्स्ट रिकॉग्निशन टेस्ट है।",
  "language": "hi",
  "model": "models/tetrax-stt-v1",
  "processing_time_sec": 0.452
}
```

### 3. Node.js / TypeScript Integration (Tetrax CPaaS Backend)
```typescript
import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';

const STT_API_URL = 'http://<GPU_SERVER_IP>:8000/v1/audio/transcriptions';

export async function transcribeAudio(audioFilePath: string, language: string = 'hi') {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(audioFilePath));
  formData.append('language', language);

  const response = await axios.post(STT_API_URL, formData, {
    headers: formData.getHeaders(),
  });

  return response.data; // { text: "...", language: "hi", model: "models/tetrax-stt-v1", processing_time_sec: 0.452 }
}
```

### 4. Python Integration
```python
import requests

STT_API_URL = "http://<GPU_SERVER_IP>:8000/v1/audio/transcriptions"

def transcribe_speech(audio_path, language="hi"):
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"language": language}
        response = requests.post(STT_API_URL, files=files, data=data)
        return response.json()

# Example Usage:
res = transcribe_speech("sample_audio.wav", language="hi")
print("Transcribed Text:", res.get("text"))
```

