# 🚀 Guide: Fine-Tuning & Hosting Tetrax STT on RunPod (Cloud GPU Server)

RunPod provides high-performance cloud GPUs (RTX 3090, RTX 4090, A100, L40S) for fast fine-tuning on **AI4Bharat IndicVoices (Marathi)** and **Google FLEURS**, as well as hosting 24/7 production STT API servers.

---

## 📋 Step 1: Deploy a RunPod GPU Pod

1. Log in to [RunPod.io](https://www.runpod.io/).
2. Go to **GPU Pods** ➔ Click **Deploy Pod**.
3. Select a GPU:
   - **Recommended for Training & Production:** RTX 4090 (24 GB VRAM) or RTX 3090 (24 GB VRAM) — fast, cost-effective.
   - **High Concurrency Production / Ultra-Fast Training:** A100 (80 GB VRAM) or L40S (48 GB VRAM).
4. Select Template: **PyTorch 2.x** (e.g., `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` or latest PyTorch template).
5. Click **Customize Pod**:
   - Set **Container Disk**: `40 GB`
   - Set **Volume Disk**: `60 GB` – `100 GB` (Mount path: `/workspace` — persistent storage for datasets, extracted audio, and model checkpoints).
   - **HTTP Ports**: Expose port `8000` (for FastAPI STT server).
   - *(Optional)* **Environment Variables**: Add `HF_TOKEN` = `your_huggingface_token` (required for gated datasets like AI4Bharat IndicVoices).
6. Click **Deploy Pod**.

---

## 💻 Step 2: Connect to RunPod & Set Up Environment

1. Once the Pod status is `Running`, click **Connect**.
2. Open **Web Terminal** (or connect via **SSH** / **JupyterLab**).
3. Navigate to the persistent `/workspace` directory:
   ```bash
   cd /workspace
   ```

4. Clone your repository (or upload `tetrax-stt`):
   ```bash
   git clone YOUR_GITHUB_REPO_URL tetrax-stt
   cd tetrax-stt
   ```

5. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Authenticate with Hugging Face:
   > ⚠️ **Important:** `ai4bharat/IndicVoices` is a gated dataset on Hugging Face.
   > 1. Visit [https://huggingface.co/datasets/ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) and accept the terms of use.
   > 2. Log in with your Hugging Face access token:
   ```bash
   huggingface-cli login --token YOUR_HF_TOKEN
   # Or set the environment variable:
   export HF_TOKEN="YOUR_HF_TOKEN"
   ```

7. Verify GPU diagnostic:
   ```bash
   python scripts/check_environment.py
   ```
   *Expected output: Reports CUDA True, RTX 4090 (24.0 GiB VRAM) or equivalent.*

---

## 📦 Step 3: Prepare Dataset on RunPod

RunPod instances benefit from ultra-fast 1–10 Gbps cloud network connectivity, making dataset downloads and conversions exceptionally fast.

### Option A: AI4Bharat IndicVoices (Marathi) — *Recommended for Marathi STT*
Download the high-quality conversational Marathi speech dataset from AI4Bharat:
```bash
# Download and extract Marathi speech & build manifests:
python scripts/download_indicvoices_data.py --languages marathi
```

> **Tip (Smoke Testing / Quick Run):** To download a smaller subset for quick iteration, use `--limit`:
> ```bash
> python scripts/download_indicvoices_data.py --languages marathi --limit 500
> ```

### Option B: Combined Dataset (Google FLEURS + AI4Bharat IndicVoices)
To train a comprehensive multilingual model covering Hindi, English, and rich conversational Marathi:
```bash
# 1. Download Google FLEURS (hi, mr, en)
python scripts/download_fleurs_data.py

# 2. Append AI4Bharat IndicVoices Marathi into the same manifests
python scripts/download_indicvoices_data.py --languages marathi --append
```

### Option C: Google FLEURS Only (`hi`, `mr`, `en`)
```bash
python scripts/download_fleurs_data.py
```

### Option D: Local Raw Dataset Upload
If uploading `hi_in`, `mr_in`, `en_us` folders to `/workspace/tetrax-stt/data/raw/`:
```bash
python scripts/prepare_local_fleurs.py
```

### Validate Dataset Manifests:
Always verify dataset integrity before launching training:
```bash
python scripts/prepare_dataset.py
```

---

## 🏋️ Step 4: Run High-Performance Fine-Tuning

Because RunPod GPUs offer 24GB+ VRAM, you can train with larger batch sizes for faster convergence:

### For RTX 4090 / RTX 3090 (24 GB VRAM):
```bash
python scripts/train.py --max_steps 1000 --batch_size 8 --grad_accum 2 --learning_rate 1e-5
```

### For A100 (80 GB VRAM) / L40S (48 GB VRAM):
```bash
python scripts/train.py --max_steps 1500 --batch_size 16 --grad_accum 2 --learning_rate 1e-5
```

> The fine-tuned model checkpoint is saved permanently under:
> `/workspace/tetrax-stt/models/tetrax-stt-v1`

---

## 📊 Step 5: Evaluate Model Accuracy (WER)

Evaluate Word Error Rate (WER) across all test splits:

```bash
python scripts/evaluate.py
```

Inspect the per-language error report (including Marathi `mr`):
```bash
cat reports/wer_by_language.csv
```

---

## 🌐 Step 6: Deploy 24/7 Production STT API Server on RunPod

Start the FastAPI application with Uvicorn:

```bash
uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000
```

To run in the background (persistent production process):
```bash
nohup uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000 > /workspace/stt_server.log 2>&1 &
```

### Accessing your STT Endpoint externally:

RunPod automatically generates a public proxy URL for exposed port `8000`:
- **RunPod Proxy Endpoint:**
  `https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions`
- **Health Check:**
  `https://<YOUR_POD_ID>-8000.proxy.runpod.net/health`

### Test Transcription with Curl:

**Marathi (IndicVoices sample):**
```bash
curl -X POST https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions \
  -F "file=@data/audio/mr/test/indicvoices_mr_test_00000.wav" \
  -F "language=mr"
```

**Hindi:**
```bash
curl -X POST https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions \
  -F "file=@data/audio/hi/test/hi_test_00000.wav" \
  -F "language=hi"
```

---

## 🔗 Step 7: Connect Tetrax CPaaS Backend to RunPod STT Server

### Node.js / Express Backend (`tetrax-backend`):
```typescript
import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';

const RUNPOD_STT_URL = 'https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions';

export async function transcribeAudio(audioFilePath: string, language: string = 'mr') {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(audioFilePath));
  formData.append('language', language); // 'mr', 'hi', or 'en'

  const response = await axios.post(RUNPOD_STT_URL, formData, {
    headers: formData.getHeaders(),
  });

  return response.data; // { text: "...", language: "mr", model: "models/tetrax-stt-v1" }
}
```

### Python Backend:
```python
import requests

RUNPOD_STT_URL = "https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions"

def transcribe_speech(audio_file_path, language="mr"):
    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        data = {"language": language} # 'mr', 'hi', or 'en'
        response = requests.post(RUNPOD_STT_URL, files=files, data=data)
        return response.json()
```
