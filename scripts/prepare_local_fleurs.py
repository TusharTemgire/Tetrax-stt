import csv
import tarfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AUDIO_OUTPUT_DIR = DATA_DIR / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"

LANG_MAP = {
    "hi_in": "hi",
    "mr_in": "mr",
    "en_us": "en",
    "hi": "hi",
    "mr": "mr",
    "en": "en",
}

# Mapping split names in FLEURS (dev -> validation)
SPLIT_MAP = {
    "train": "train",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
}

def extract_tar_if_needed(tar_path: Path, extract_to: Path):
    """Extracts tar.gz archive if destination files are not yet extracted."""
    if not tar_path.exists():
        print(f"  [Warning] Archive {tar_path} not found.")
        return False

    print(f"  Extracting {tar_path.name} -> {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    return True

def process_tsv(tsv_path: Path, audio_dir: Path, target_lang: str, split_name: str, manifest_rows: dict):
    """Reads FLEURS tsv transcript file and matches audio files."""
    if not tsv_path.exists():
        print(f"  [Warning] TSV file {tsv_path} not found.")
        return

    count = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 4:
                continue

            # In FLEURS TSV format:
            # col 0: id, col 1: filename, col 2: raw_transcript, col 3: transcript
            filename = row[1].strip()
            text = row[3].strip() if len(row) > 3 and row[3].strip() else row[2].strip()

            if not text:
                continue

            # Look for audio file in audio_dir
            audio_file = audio_dir / filename
            if not audio_file.exists():
                # Check nested paths if archive extracted with subdirectories
                matching_files = list(audio_dir.rglob(filename))
                if matching_files:
                    audio_file = matching_files[0]
                else:
                    continue

            rel_path = audio_file.relative_to(ROOT).as_posix()
            manifest_rows[split_name].append({
                "audio": rel_path,
                "text": text,
                "language": target_lang
            })
            count += 1

    print(f"  Processed {count} entries from {tsv_path.name} ({target_lang} / {split_name})")

def main():
    parser = argparse.ArgumentParser(description="Process raw local FLEURS dataset files (tar.gz and tsv files).")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DIR), help="Path to directory containing local hi_in, mr_in, en_us folders.")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = {"train": [], "validation": [], "test": []}

    print(f"Scanning raw local FLEURS directory: {raw_path}")

    # Look for language subdirectories (e.g. hi_in, mr_in, en_us)
    for lang_dir in raw_path.iterdir():
        if not lang_dir.is_dir():
            continue

        folder_name = lang_dir.name.lower()
        if folder_name not in LANG_MAP:
            # Check if inner data directory exists e.g. hi_in/data/hi_in or similar
            matching_key = next((k for k in LANG_MAP if k in folder_name), None)
            if not matching_key:
                continue
            target_lang = LANG_MAP[matching_key]
        else:
            target_lang = LANG_MAP[folder_name]

        print(f"\n---> Processing local FLEURS dataset folder: {lang_dir.name} ({target_lang})")

        # Find tsv and tar.gz files within folder or nested audio subfolder
        for split_key, target_split in SPLIT_MAP.items():
            tsv_candidates = list(lang_dir.rglob(f"{split_key}.tsv"))
            tar_candidates = list(lang_dir.rglob(f"{split_key}.tar.gz"))

            if not tsv_candidates:
                continue

            tsv_file = tsv_candidates[0]
            audio_out = AUDIO_OUTPUT_DIR / target_lang / target_split

            if tar_candidates:
                tar_file = tar_candidates[0]
                extract_tar_if_needed(tar_file, audio_out)

            process_tsv(tsv_file, audio_out, target_lang, target_split, manifest_rows)

    # Write manifests
    for split, rows in manifest_rows.items():
        manifest_file = MANIFEST_DIR / f"{split}.csv"
        with manifest_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio", "text", "language"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nUpdated manifest [{split}.csv] with {len(rows)} total records -> {manifest_file}")

    print("\nLocal dataset preparation completed!")

if __name__ == "__main__":
    main()
