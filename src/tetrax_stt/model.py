import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from transformers import pipeline

DEFAULT_MODEL_NAME = "openai/whisper-small"

LANGUAGE_MAP = {
    "hi": "hindi",
    "mr": "marathi",
    "en": "english",
    "hi_in": "hindi",
    "mr_in": "marathi",
    "en_us": "english",
}

ROOT = Path(__file__).resolve().parents[2]

def resolve_default_model() -> str:
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
    return DEFAULT_MODEL_NAME

class TetraxSTTModel:
    """
    Inference wrapper for Tetrax Fine-Tuned Whisper STT model.
    """
    def __init__(self, model_path_or_name: Optional[Union[str, Path]] = None, device: Optional[Union[str, int, torch.device]] = None):
        if model_path_or_name is None:
            self.model_path = resolve_default_model()
        else:
            self.model_path = str(model_path_or_name)

        if device is None:
            self.device_id = 0 if torch.cuda.is_available() else -1
        elif isinstance(device, str):
            self.device_id = 0 if ("cuda" in device and torch.cuda.is_available()) else -1
        elif isinstance(device, torch.device):
            self.device_id = 0 if (device.type == "cuda" and torch.cuda.is_available()) else -1
        else:
            self.device_id = device

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_path,
            device=self.device_id,
            chunk_length_s=30,
        )

    def transcribe(
        self,
        audio_input: Union[str, Path, Any],
        language: str = "hi",
        return_timestamps: Union[bool, str] = False
    ) -> Dict[str, Any]:
        """
        Transcribes input audio for given language.
        """
        lang_code = language.strip().lower()
        whisper_lang = LANGUAGE_MAP.get(lang_code, "hindi")

        generate_kwargs = {
            "language": whisper_lang,
            "task": "transcribe"
        }

        result = self.asr_pipeline(
            audio_input,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
        )

        return {
            "text": result.get("text", "").strip(),
            "language": lang_code,
            "chunks": result.get("chunks", None)
        }
