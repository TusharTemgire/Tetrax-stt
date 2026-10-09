import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"

LANGUAGES = {"en", "mr", "hi"}
REQUIRED_COLUMNS = {"audio", "text", "language"}

def main():
    failed = False

    for split in ("train", "validation", "test"):
        manifest = MANIFEST_DIR / f"{split}.csv"

        if not manifest.exists():
            print(f"Missing manifest: {manifest}")
            failed = True
            continue

        count = 0

        with manifest.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                print(f"{split}: missing required columns (expected audio,text,language)")
                failed = True
                continue

            for line, row in enumerate(reader, start=2):
                audio_str = row["audio"].strip()
                audio = Path(audio_str)
                if not audio.is_absolute():
                    audio = ROOT / audio

                language = row["language"].strip()
                transcript = row["text"].strip()

                if language not in LANGUAGES:
                    print(f"{split}:{line}: unknown language '{language}'")
                    failed = True

                if not transcript:
                    print(f"{split}:{line}: empty transcript")
                    failed = True

                if not audio.is_file():
                    print(f"{split}:{line}: missing audio file {audio}")
                    failed = True

                count += 1

        print(f"Manifest [{split}]: checked {count} rows successfully.")

    if failed:
        sys.exit("Dataset validation failed.")

    print("\nManifest validation passed completely.")

if __name__ == "__main__":
    main()
