import csv
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import soundfile as sf
import torch

# Prevent script name (evaluate.py) from shadowing third-party 'evaluate' package in sys.path
script_dir = str(Path(__file__).resolve().parent)
sys_path_removed = False
if sys.path and Path(sys.path[0]).resolve() == Path(script_dir).resolve():
    sys.path.pop(0)
    sys_path_removed = True

# Load WER metric calculation function (HF evaluate package with fallback to jiwer)
compute_wer_fn = None
try:
    import evaluate
    if hasattr(evaluate, "load"):
        wer_metric = evaluate.load("wer")
        compute_wer_fn = lambda preds, refs: float(wer_metric.compute(predictions=preds, references=refs))
except Exception:
    pass

if compute_wer_fn is None:
    import jiwer
    compute_wer_fn = lambda preds, refs: float(jiwer.wer(refs, preds))

if sys_path_removed:
    sys.path.insert(0, script_dir)

from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "tetrax-stt-v1"
FALLBACK_MODEL = "openai/whisper-small"
TEST_CSV = ROOT / "data" / "manifests" / "test.csv"
REPORT_CSV = ROOT / "reports" / "wer_by_language.csv"

LANGUAGES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper STT model per language.")
    parser.add_argument("--model", type=str, default=None, help="Path to model directory or HuggingFace ID.")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() else FALLBACK_MODEL

    print(f"Loading model for evaluation: {model_path}")
    device_id = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device_id,
        chunk_length_s=30,
    )

    records = defaultdict(lambda: {"refs": [], "preds": []})

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test manifest not found at: {TEST_CSV}. Run scripts/download_fleurs_data.py first.")

    with TEST_CSV.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lang = row["language"].strip().lower()
            audio_path = Path(row["audio"])

            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path

            if not audio_path.exists():
                print(f"Warning: audio path {audio_path} not found, skipping...")
                continue

            audio, sample_rate = sf.read(str(audio_path), dtype="float32")

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            if lang not in LANGUAGES:
                continue

            result = asr(
                {"raw": audio, "sampling_rate": sample_rate},
                generate_kwargs={
                    "language": LANGUAGES[lang],
                    "task": "transcribe",
                },
            )

            records[lang]["refs"].append(row["text"].strip())
            records[lang]["preds"].append(result["text"].strip())

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ Evaluation Results (WER) ================")
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["language", "samples", "wer"])
        writer.writeheader()

        for lang, values in sorted(records.items()):
            if not values["refs"]:
                continue
            score = compute_wer_fn(
                preds=values["preds"],
                refs=values["refs"],
            )

            writer.writerow({
                "language": lang,
                "samples": len(values["refs"]),
                "wer": round(float(score), 4),
            })

            print(f"Language: {lang:5s} | Samples: {len(values['refs']):4d} | WER: {score:.4f}")

    print("==========================================================")
    print(f"Per-language WER report saved to: {REPORT_CSV}")

if __name__ == "__main__":
    main()

