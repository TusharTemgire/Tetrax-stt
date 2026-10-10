import io
import sys
import unittest
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to sys.path for test runner compatibility
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tetrax_stt.model import resolve_default_model, LANGUAGE_MAP
from tetrax_stt.audio import load_and_preprocess_audio, get_audio_duration

class TestTetraxSTT(unittest.TestCase):
    def test_language_map(self):
        self.assertEqual(LANGUAGE_MAP["hi"], "hindi")
        self.assertEqual(LANGUAGE_MAP["mr"], "marathi")
        self.assertEqual(LANGUAGE_MAP["en"], "english")
        self.assertEqual(LANGUAGE_MAP["hi_in"], "hindi")
        self.assertEqual(LANGUAGE_MAP["mr_in"], "marathi")
        self.assertEqual(LANGUAGE_MAP["en_us"], "english")

    def test_resolve_default_model(self):
        model_path = resolve_default_model()
        self.assertIsInstance(model_path, str)
        self.assertTrue(len(model_path) > 0)

    def test_audio_preprocessing_bytes(self):
        sr = 16000
        t = np.linspace(0, 1, sr, False)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, signal, sr, format="WAV")
        buf.seek(0)
        audio_bytes = buf.read()

        audio_array, out_sr = load_and_preprocess_audio(audio_bytes, target_sr=16000)
        self.assertEqual(out_sr, 16000)
        self.assertEqual(len(audio_array), sr)
        self.assertEqual(get_audio_duration(audio_array, out_sr), 1.0)

    def test_audio_preprocessing_stereo_resample(self):
        sr = 44100
        signal = np.zeros((sr, 2), dtype=np.float32)

        buf = io.BytesIO()
        sf.write(buf, signal, sr, format="WAV")
        buf.seek(0)
        audio_bytes = buf.read()

        audio_array, out_sr = load_and_preprocess_audio(audio_bytes, target_sr=16000)
        self.assertEqual(out_sr, 16000)
        self.assertEqual(audio_array.ndim, 1)

if __name__ == "__main__":
    unittest.main()
