import io
from pathlib import Path
from typing import Union, Tuple
import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000

def load_and_preprocess_audio(
    audio_input: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Loads audio file or bytes, converts to mono, and resamples to target sample rate (16kHz).
    """
    if isinstance(audio_input, (bytes, io.BytesIO)):
        if isinstance(audio_input, bytes):
            audio_input = io.BytesIO(audio_input)
        audio_array, sr = sf.read(audio_input, dtype="float32")
    else:
        audio_path = Path(audio_input)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_array, sr = sf.read(str(audio_path), dtype="float32")

    # Convert multi-channel audio to mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample audio if sample rate differs from target (16kHz)
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio_array, sr


def get_audio_duration(audio_array: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    """Returns total duration of audio array in seconds."""
    return len(audio_array) / float(sample_rate)
