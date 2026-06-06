#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def write_txt_files(
    corpus_root: Path,
    transcription_file: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    written = 0
    skipped = 0
    missing = 0

    with transcription_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, 1):
            if len(row) < 2:
                skipped += 1
                print(f"[skip] line {line_no}: invalid row")
                continue

            wav_rel = row[0].strip()
            text = ",".join(row[1:]).strip()
            wav_path = corpus_root / wav_rel

            if not wav_path.is_file():
                missing += 1
                print(f"[missing] line {line_no}: {wav_rel}")
                continue

            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists() and not overwrite:
                skipped += 1
                continue

            if not dry_run:
                txt_path.write_text(text, encoding="utf-8")
            written += 1

    return written, skipped, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate same-name .txt files beside wav files for ehehe-corpus."
    )
    parser.add_argument(
        "corpus_root",
        type=Path,
        help="ehehe-corpus root path, e.g. /usr/local/corpus/ja/ehehe-corpus",
    )
    parser.add_argument(
        "--transcriptions",
        type=Path,
        help="CSV file containing wav relative path and transcription. "
        "Default: <corpus_root>/original_transcriptions.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print summary; do not write files.",
    )
    args = parser.parse_args()

    corpus_root = args.corpus_root.resolve()
    transcription_file = (
        args.transcriptions.resolve()
        if args.transcriptions
        else corpus_root / "original_transcriptions.csv"
    )

    if not corpus_root.is_dir():
        raise SystemExit(f"Corpus root not found: {corpus_root}")
    if not transcription_file.is_file():
        raise SystemExit(f"Transcription file not found: {transcription_file}")

    written, skipped, missing = write_txt_files(
        corpus_root=corpus_root,
        transcription_file=transcription_file,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"[summary] written={written}, skipped={skipped}, missing_wav={missing}")


if __name__ == "__main__":
    main()
