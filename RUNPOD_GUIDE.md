# 🚀 Guide: Fine-Tuning & Hosting Tetrax STT on RunPod (Cloud GPU Server)

RunPod provides high-performance cloud GPUs (RTX 3090, RTX 4090, A100, L40S) for fast fine-tuning and hosting 24/7 production STT API servers.

---

## 📋 Step 1: Deploy a RunPod GPU Pod

1. Log in to [RunPod.io](https://www.runpod.io/).
2. Go to **GPU Pods** ➔ Click **Deploy Pod**.
3. Select a GPU:
   - **Recommended for Training & Production:** RTX 4090 (24 GB VRAM) or RTX 3090 (24 GB VRAM).
   - **High Concurrency Production:** A100 (80 GB VRAM) or L40S (48 GB VRAM).
4. Select Template: **PyTorch 2.x** (e.g., `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`).
5. Click **Customize Pod**:
   - Set **Container Disk**: `30 GB`
   - Set **Volume Disk**: `50 GB` (Mount path: `/workspace` - persistent storage).
   - **HTTP Ports**: Expose port `8000` (for FastAPI STT server).
6. Click **Deploy Pod**.

---

## 💻 Step 2: Connect to RunPod & Set Up Environment

1. Once the Pod is `Running`, click **Connect**.
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

6. Verify GPU diagnostic:
   ```bash
   python scripts/check_environment.py
   ```
   *Expected output: Reports CUDA True, RTX 4090 (24.0 GiB VRAM).*

---

## 📦 Step 3: Prepare Dataset on RunPod

### Option A: Automatic HuggingFace Download (Fast speed on cloud bandwidth)
```bash
python scripts/download_fleurs_data.py
```

### Option B: Local Raw Dataset Upload
If uploading `hi_in`, `mr_in`, `en_us` folders to `/workspace/tetrax-stt/data/raw/`:
```bash
python scripts/prepare_local_fleurs.py
```

### Validate Manifests:
```bash
python scripts/prepare_dataset.py
```

---

## 🏋️ Step 4: Run High-Performance Fine-Tuning

Because RunPod GPUs have 24GB+ VRAM, you can use larger batch sizes for faster convergence:

```bash
python scripts/train.py --max_steps 1000 --batch_size 8 --grad_accum 2 --learning_rate 1e-5
```

The fine-tuned model will be saved permanently under `/workspace/tetrax-stt/models/tetrax-stt-v1`.

---

## 📊 Step 5: Evaluate Model Accuracy (WER)

```bash
python scripts/evaluate.py
```

View the per-language report:
```bash
cat reports/wer_by_language.csv
```

---

## 🌐 Step 6: Deploy 24/7 Production STT API Server on RunPod

Start the FastAPI application:

```bash
uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000
```

### Accessing your STT Endpoint externally:

RunPod automatically generates a public proxy URL for exposed port `8000`:
- **RunPod Proxy Endpoint:**
  `https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions`
- **Health Check:**
  `https://<YOUR_POD_ID>-8000.proxy.runpod.net/health`

---

## 🔗 Step 7: Connect Tetrax CPaaS Backend to RunPod STT Server

### Node.js / Express Backend (`tetrax-backend`):
```typescript
import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';

const RUNPOD_STT_URL = 'https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions';

export async function transcribeAudio(audioFilePath: string, language: string = 'hi') {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(audioFilePath));
  formData.append('language', language);

  const response = await axios.post(RUNPOD_STT_URL, formData, {
    headers: formData.getHeaders(),
  });

  return response.data; // { text: "...", language: "hi", model: "models/tetrax-stt-v1" }
}
```

### Python Backend:
```python
import requests

RUNPOD_STT_URL = "https://<YOUR_POD_ID>-8000.proxy.runpod.net/v1/audio/transcriptions"

def transcribe_speech(audio_file_path, language="hi"):
    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        data = {"language": language}
        response = requests.post(RUNPOD_STT_URL, files=files, data=data)
        return response.json()
```
