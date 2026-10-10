import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
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
    def __init__(self, model_path_or_name: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if model_path_or_name is None:
            self.model_path = resolve_default_model()
        else:
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

        if isinstance(audio_input, (str, Path, bytes)):
            from tetrax_stt.audio import load_and_preprocess_audio
            audio_array, sr = load_and_preprocess_audio(audio_input)
            inputs = {"raw": audio_array, "sampling_rate": sr}
        else:
            inputs = audio_input

        result = self.asr_pipeline(
            inputs,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
        )

        return {
            "text": result.get("text", "").strip(),
            "language": lang_code,
            "chunks": result.get("chunks", None)
        }

    def transcribe_stream_buffer(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: str = "hi",
    ) -> str:
        """
        Transcribes an in-memory 16kHz float32 audio array chunk for real-time streaming.
        """
        if len(audio_array) == 0:
            return ""

        lang_code = language.strip().lower()
        whisper_lang = LANGUAGE_MAP.get(lang_code, "hindi")

        generate_kwargs = {
            "language": whisper_lang,
            "task": "transcribe"
        }

        # Format input dict for HuggingFace ASR pipeline raw array
        inputs = {"raw": audio_array, "sampling_rate": sample_rate}
        result = self.asr_pipeline(inputs, generate_kwargs=generate_kwargs)
        return result.get("text", "").strip()

