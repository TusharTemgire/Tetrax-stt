import os
import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = ROOT / "checkpoints" / "tetrax-stt-v1"
FINAL_MODEL_DIR = ROOT / "models" / "tetrax-stt-v1"

LANGUAGE_NAMES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
}

class DataCollatorSpeechSeq2Seq:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        audio_features = [
            {"input_features": x["input_features"]}
            for x in features
        ]

        batch = self.processor.feature_extractor.pad(
            audio_features,
            return_tensors="pt",
        )

        label_features = [
            {"input_ids": x["labels"]}
            for x in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        bos = self.processor.tokenizer.bos_token_id
        if (
            labels.shape[1] > 0
            and (labels[:, 0] == bos).all()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def run_preflight_checks():
    import csv
    import sys

    print("==================================================")
    print(" Running Pre-Flight Dataset & Environment Checks")
    print("==================================================")

    # 1. Environment & GPU Check
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gib = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"GPU Device      : {gpu_name} ({vram_gib} GiB VRAM)")
    else:
        print("GPU Device      : NOT DETECTED (CPU Mode Only)")
        print("  ⚠️ WARNING: You are running PyTorch CPU build or no GPU is available.")
        print("  ⚠️ Training on CPU will be very slow. Use Google Colab or RunPod for 10x-50x faster GPU training!")

    # 2. Manifest Checks
    manifest_dir = ROOT / "data" / "manifests"
    required_cols = {"audio", "text", "language"}
    errors = []

    for split in ("train", "validation"):
        manifest_path = manifest_dir / f"{split}.csv"
        if not manifest_path.exists():
            errors.append(f"Missing manifest file: {manifest_path}. Please run dataset download/preparation first!")
            continue

        sample_count = 0
        with manifest_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not required_cols.issubset(reader.fieldnames or []):
                errors.append(f"[{split}.csv] Missing required columns. Found: {reader.fieldnames}")
                continue

            for line_idx, row in enumerate(reader, start=2):
                sample_count += 1
                lang = row.get("language", "").strip().lower()
                text = row.get("text", "").strip()
                audio_str = row.get("audio", "").strip()

                if lang not in LANGUAGE_NAMES:
                    errors.append(f"[{split}.csv:{line_idx}] Unsupported language '{lang}'. Supported: {list(LANGUAGE_NAMES.keys())}")
                if not text:
                    errors.append(f"[{split}.csv:{line_idx}] Empty transcript detected!")

                audio_path = Path(audio_str)
                if not audio_path.is_absolute():
                    audio_path = ROOT / audio_path
                if not audio_path.exists():
                    errors.append(f"[{split}.csv:{line_idx}] Audio file not found: {audio_path}")

                # Limit detailed missing file reporting to top 10 errors to avoid log spam
                if len(errors) >= 10:
                    break

        print(f"Manifest [{split}.csv]: Validated {sample_count} samples.")

    if errors:
        print("\n❌ PRE-FLIGHT CHECK FAILED WITH THE FOLLOWING ERRORS:")
        for err in errors:
            print(f"  - {err}")
        print("==================================================")
        sys.exit("\nAborting training due to failed pre-flight checks.")

    print("✅ All Pre-Flight Checks Passed! Proceeding to Model Training...\n==================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper Small on Hindi, Marathi, English STT data.")
    parser.add_argument("--max_steps", type=int, default=500, help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    args = parser.parse_args()

    # Run automated pre-flight checks BEFORE downloading or loading model
    run_preflight_checks()

    print(f"Loading processor and model: {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    data_files = {
        split: str(ROOT / "data" / "manifests" / f"{split}.csv")
        for split in ("train", "validation")
    }

    print("Loading CSV datasets...")
    dataset = load_dataset("csv", data_files=data_files)

    import soundfile as sf
    import librosa

    def prepare_example(example):
        audio_item = example["audio"]
        
        # Extract audio array and sampling rate from file path or dict
        if isinstance(audio_item, dict):
            audio_array = audio_item["array"]
            sr = audio_item["sampling_rate"]
        else:
            audio_path = Path(audio_item)
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            audio_array, sr = sf.read(str(audio_path), dtype="float32")

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        language = example["language"].strip().lower()
        text = example["text"].strip()

        if language not in LANGUAGE_NAMES:
            raise ValueError(f"Unknown language code: {language}. Expected one of {list(LANGUAGE_NAMES.keys())}")

        features = processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
        ).input_features[0]

        processor.tokenizer.set_prefix_tokens(
            language=LANGUAGE_NAMES[language],
            task="transcribe",
        )

        labels = processor.tokenizer(text).input_ids

        return {
            "input_features": features,
            "labels": labels,
        }

    print("Preprocessing audio features and language tokens...")
    encoded = dataset.map(
        prepare_example,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        desc="Preparing audio features and language labels",
    )

    collator = DataCollatorSpeechSeq2Seq(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        predict_with_generate=False,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0 if os.name == "nt" else 2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=collator,
        processing_class=processor,
    )

    print("Starting fine-tuning process...")
    trainer.train()

    print(f"Saving final model to: {FINAL_MODEL_DIR}")
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    processor.save_pretrained(str(FINAL_MODEL_DIR))

    print("\nFine-tuning completed successfully!")

if __name__ == "__main__":
    main()
