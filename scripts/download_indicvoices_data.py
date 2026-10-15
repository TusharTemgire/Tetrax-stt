import os
import csv
import argparse
from pathlib import Path
import soundfile as sf
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"

INDICVOICES_LANG_MAP = {
    "marathi": {"code": "mr", "name": "Marathi"},
    "hindi": {"code": "hi", "name": "Hindi"},
    "english": {"code": "en", "name": "English"},
    "mr": {"code": "mr", "name": "Marathi"},
    "hi": {"code": "hi", "name": "Hindi"},
    "en": {"code": "en", "name": "English"},
}

SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "dev": "validation",
    "test": "test",
}

def main():
    parser = argparse.ArgumentParser(description="Download and process ai4bharat/IndicVoices dataset for fine-tuning Tetrax STT.")
    parser.add_argument("--languages", type=str, default="marathi", help="Comma-separated language configs (e.g. marathi, hindi, english).")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (or set HF_TOKEN env var).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples per split per language.")
    parser.add_argument("--local-dir", type=str, default=None, help="Path to directory containing manually downloaded IndicVoices .parquet files (e.g. data/raw/indicvoices_marathi).")
    parser.add_argument("--append", action="store_true", help="Append to existing train/validation/test manifests instead of replacing them.")
    args = parser.parse_args()

    hf_token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = {"train": [], "validation": [], "test": []}

    print("==================================================================")
    print(" AI4Bharat IndicVoices Dataset Downloader & Converter for Tetrax STT")
    print("==================================================================")

    # -------------------------------------------------------------
    # MODE 1: Local Parquet files provided
    # -------------------------------------------------------------
    if args.local_dir:
        local_path = Path(args.local_dir)
        if not local_path.exists():
            print(f"❌ Error: Specified local directory does not exist: {local_path}")
            return

        parquet_files = sorted(list(local_path.rglob("*.parquet")))
        if not parquet_files:
            print(f"❌ No .parquet files found in: {local_path}")
            return

        print(f"Found {len(parquet_files)} parquet file(s) in {local_path}")

        # Group by split
        split_files = {"train": [], "validation": [], "test": []}
        for p in parquet_files:
            name_lower = p.name.lower()
            if "dev" in name_lower or "valid" in name_lower:
                split_files["validation"].append(str(p))
            elif "test" in name_lower:
                split_files["test"].append(str(p))
            else:
                split_files["train"].append(str(p))

        # Determine language code
        raw_lang = args.languages.split(",")[0].strip().lower()
        info = INDICVOICES_LANG_MAP.get(raw_lang, {"code": "mr", "name": "Marathi"})
        lang_code = info["code"]

        data_dict = {k: v for k, v in split_files.items() if v}
        try:
            from datasets import Audio
            dataset_dict = load_dataset("parquet", data_files=data_dict)
            try:
                dataset_dict = dataset_dict.cast_column("audio", Audio())
            except Exception:
                pass
        except Exception as e:
            print(f"❌ Failed to load local parquet files via datasets: {e}")
            return

        for split_canonical, split_ds in dataset_dict.items():
            if args.limit and len(split_ds) > args.limit:
                split_ds = split_ds.select(range(args.limit))

            split_audio_dir = AUDIO_DIR / lang_code / split_canonical
            split_audio_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing {len(split_ds)} samples from local parquet files for [{lang_code}] split: [{split_canonical}]...")

            count = 0
            for idx, item in enumerate(split_ds):
                raw_audio = item.get("audio")
                if not raw_audio:
                    continue

                audio_array = None
                sampling_rate = 16000
                if isinstance(raw_audio, dict) and "array" in raw_audio and raw_audio["array"] is not None:
                    audio_array = raw_audio["array"]
                    sampling_rate = raw_audio.get("sampling_rate", 16000)
                elif isinstance(raw_audio, dict) and "bytes" in raw_audio and raw_audio["bytes"]:
                    import io
                    audio_array, sampling_rate = sf.read(io.BytesIO(raw_audio["bytes"]))

                if audio_array is None:
                    continue

                text = (
                    item.get("transcription")
                    or item.get("text")
                    or item.get("normalized")
                    or item.get("verbatim")
                    or item.get("raw_transcription")
                    or ""
                ).strip()

                if not text:
                    continue

                audio_filename = f"indicvoices_{lang_code}_{split_canonical}_{idx:05d}.wav"
                audio_path = split_audio_dir / audio_filename

                sf.write(str(audio_path), audio_array, sampling_rate)
                rel_audio_path = audio_path.relative_to(ROOT).as_posix()

                manifest_rows[split_canonical].append({
                    "audio": rel_audio_path,
                    "text": text,
                    "language": lang_code,
                })
                count += 1

            print(f"  Saved {count} samples for [{lang_code} / {split_canonical}].")

    # -------------------------------------------------------------
    # MODE 2: Online download from Hugging Face Hub
    # -------------------------------------------------------------
    else:
        lang_keys = [l.strip().lower() for l in args.languages.split(",") if l.strip()]

        if not hf_token:
            print("💡 NOTE: ai4bharat/IndicVoices is a gated dataset on Hugging Face.")
            print("   If you encounter an authentication error:")
            print("   1. Accept dataset terms at: https://huggingface.co/datasets/ai4bharat/IndicVoices")
            print("   2. Pass --token YOUR_HF_TOKEN or run 'huggingface-cli login'\n")

        for raw_lang in lang_keys:
            if raw_lang not in INDICVOICES_LANG_MAP:
                print(f"Skipping unknown language key '{raw_lang}'. Supported: {list(INDICVOICES_LANG_MAP.keys())}")
                continue

            info = INDICVOICES_LANG_MAP[raw_lang]
            config_name = raw_lang if raw_lang in ("marathi", "hindi", "english") else info["name"].lower()
            lang_code = info["code"]

            print(f"\n---> Fetching IndicVoices subset: '{config_name}' ({info['name']})...")

            try:
                dataset_dict = load_dataset("ai4bharat/IndicVoices", config_name, token=hf_token)
            except Exception as e:
                print(f"❌ Failed to load IndicVoices for '{config_name}': {e}")
                print("   Ensure you have logged into Hugging Face and have access to ai4bharat/IndicVoices.")
                continue

            for split_raw, split_ds in dataset_dict.items():
                split_canonical = SPLIT_MAP.get(split_raw.lower(), "train")

                if args.limit and len(split_ds) > args.limit:
                    split_ds = split_ds.select(range(args.limit))

                split_audio_dir = AUDIO_DIR / lang_code / split_canonical
                split_audio_dir.mkdir(parents=True, exist_ok=True)

                print(f"Processing {len(split_ds)} samples for [{lang_code}] split: [{split_canonical}]...")

                count = 0
                for idx, item in enumerate(split_ds):
                    raw_audio = item.get("audio")
                    if not raw_audio:
                        continue

                    audio_array = None
                    sampling_rate = 16000
                    if isinstance(raw_audio, dict) and "array" in raw_audio and raw_audio["array"] is not None:
                        audio_array = raw_audio["array"]
                        sampling_rate = raw_audio.get("sampling_rate", 16000)
                    elif isinstance(raw_audio, dict) and "bytes" in raw_audio and raw_audio["bytes"]:
                        import io
                        audio_array, sampling_rate = sf.read(io.BytesIO(raw_audio["bytes"]))

                    if audio_array is None:
                        continue

                    text = (
                        item.get("transcription")
                        or item.get("text")
                        or item.get("normalized")
                        or item.get("verbatim")
                        or item.get("raw_transcription")
                        or ""
                    ).strip()

                    if not text:
                        continue

                    audio_filename = f"indicvoices_{lang_code}_{split_canonical}_{idx:05d}.wav"
                    audio_path = split_audio_dir / audio_filename

                    sf.write(str(audio_path), audio_array, sampling_rate)

                    rel_audio_path = audio_path.relative_to(ROOT).as_posix()

                    manifest_rows[split_canonical].append({
                        "audio": rel_audio_path,
                        "text": text,
                        "language": lang_code,
                    })
                    count += 1

                print(f"  Saved {count} samples for [{lang_code} / {split_canonical}].")

    for split in ("train", "validation", "test"):
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        rows = manifest_rows[split]

        existing_rows = []
        if args.append and manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        all_rows = existing_rows + rows
        if not all_rows and not manifest_path.exists():
            continue

        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio", "text", "language"])
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Wrote {len(all_rows)} total samples to manifest: {manifest_path}")

    print("\nIndicVoices dataset processing complete!")

if __name__ == "__main__":
    main()
