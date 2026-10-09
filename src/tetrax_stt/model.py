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

class TetraxSTTModel:
    """
    Inference wrapper for Tetrax Fine-Tuned Whisper STT model.
    """
    def __init__(self, model_path_or_name: Union[str, Path] = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_path = str(model_path_or_name)
        if device is None:
            self.device_id = 0 if torch.cuda.is_available() else -1
        else:
            self.device_id = 0 if device == "cuda" else -1

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
