import argparse
import csv
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"

FLEURS_LANG_MAP = {
    "hi_in": {"code": "hi", "name": "Hindi"},
    "mr_in": {"code": "mr", "name": "Marathi"},
    "en_us": {"code": "en", "name": "English"},
}

SPLITS = ["train", "validation", "test"]

def main():
    parser = argparse.ArgumentParser(description="Download google/fleurs dataset for hi_in, mr_in, and en_us.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples per split per language (for quick smoke tests).")
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = {split: [] for split in SPLITS}

    for fleur_config, lang_info in FLEURS_LANG_MAP.items():
        lang_code = lang_info["code"]
        print(f"\n---> Fetching google/fleurs dataset subset: {fleur_config} ({lang_info['name']})")
        
        try:
            dataset_dict = load_dataset("google/fleurs", fleur_config)
        except Exception as e:
            print(f"Error loading {fleur_config}: {e}")
            continue

        for split in SPLITS:
            if split not in dataset_dict:
                continue

            split_ds = dataset_dict[split]
            if args.limit:
                split_ds = split_ds.select(range(min(len(split_ds), args.limit)))

            split_audio_dir = AUDIO_DIR / lang_code / split
            split_audio_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {len(split_ds)} examples for [{lang_code}] split: [{split}]...")

            for idx, item in enumerate(split_ds):
                raw_audio = item["audio"]
                text = item.get("transcription") or item.get("raw_transcription") or ""
                text = text.strip()

                if not text:
                    continue

                audio_array = raw_audio["array"]
                sampling_rate = raw_audio["sampling_rate"]

                audio_filename = f"{lang_code}_{split}_{idx:05d}.wav"
                audio_path = split_audio_dir / audio_filename

                sf.write(str(audio_path), audio_array, sampling_rate)

                # Store relative path from repository root for clean portability
                rel_audio_path = audio_path.relative_to(ROOT).as_posix()
                
                manifest_rows[split].append({
                    "audio": rel_audio_path,
                    "text": text,
                    "language": lang_code,
                })

    for split in SPLITS:
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        rows = manifest_rows[split]
        
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio", "text", "language"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nWrote {len(rows)} samples to manifest: {manifest_path}")

    print("\nDataset preparation from google/fleurs complete!")

if __name__ == "__main__":
    main()
