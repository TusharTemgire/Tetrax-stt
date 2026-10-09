# 🚀 Guide: Fine-Tuning Tetrax STT on Google Colab & CPaaS Integration

Google Colab provides free T4 GPUs (15 GB VRAM) which accelerates fine-tuning Whisper Small from hours down to minutes.

---

## 📋 Step 1: Open Google Colab & Enable GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`tetrax_stt_colab_training.ipynb`](file:///d:/Tushar_repos/Tetrax-CPaaS/tetrax-stt/tetrax_stt_colab_training.ipynb).
3. Go to **Runtime** ➔ **Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU**.
5. Click **Save**.

---

## 💻 Step 2: Run Notebook Cells in Colab

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

### 7. Export Model to Google Drive
```bash
!cp -r models/tetrax-stt-v1 /content/drive/MyDrive/tetrax-stt-v1
!zip -r tetrax-stt-v1.zip models/tetrax-stt-v1
```

---

## 🧪 Step 3: Run Model Inference in Python

```python
import sys
sys.path.append('src')
from tetrax_stt.model import TetraxSTTModel

# Initialize model
stt_model = TetraxSTTModel('models/tetrax-stt-v1')

# Transcribe audio file
result = stt_model.transcribe('data/audio/hi/test/hi_test_00000.wav', language='hi')
print("Transcript:", result['text'])
```

---

## 🌐 Step 4: Implement & Integrate STT Service into Tetrax CPaaS

### 1. Start FastAPI REST Server
On your production GPU server / local backend machine:
```bash
uvicorn tetrax_stt.api:app --app-dir src --host 0.0.0.0 --port 8000
```

### 2. Integrate with Node.js / TypeScript CPaaS Backend (`tetrax-backend`)

In your `tetrax-backend` service, create a speech recognition service client:

```typescript
import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';

export async function transcribeAudio(audioFilePath: string, language: string = 'hi') {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(audioFilePath));
  formData.append('language', language);

  const response = await axios.post('http://localhost:8000/v1/audio/transcriptions', formData, {
    headers: formData.getHeaders(),
  });

  return response.data; 
  // Returns: { text: "नमस्ते...", language: "hi", model: "models/tetrax-stt-v1" }
}
```

### 3. Integrate with Python CPaaS Backend

```python
import requests

def transcribe_audio_file(audio_path, language="hi"):
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"language": language}
        res = requests.post("http://localhost:8000/v1/audio/transcriptions", files=files, data=data)
        return res.json()
```
