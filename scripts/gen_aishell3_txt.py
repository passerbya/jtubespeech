#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


PINYIN_RE = re.compile(r"^[a-zvü:]+[1-5]$", re.IGNORECASE)


def parse_text(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None, ""

    wav_name = parts[0]
    text = "".join(token for token in parts[1:] if not PINYIN_RE.fullmatch(token))
    return wav_name, text


def build_wav_index(wav_root):
    return {path.name: path for path in wav_root.rglob("*.wav")}


def process_split(split_dir, overwrite=False, dry_run=False):
    content_path = split_dir / "content.txt"
    wav_root = split_dir / "wav"

    if not content_path.is_file():
        print(f"[skip] {content_path} not found")
        return 0, 0
    if not wav_root.is_dir():
        print(f"[skip] {wav_root} not found")
        return 0, 0

    wav_index = build_wav_index(wav_root)
    written = 0
    missing = 0

    with content_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            wav_name, text = parse_text(line)
            if not wav_name:
                continue

            wav_path = wav_index.get(wav_name)
            if wav_path is None:
                missing += 1
                print(f"[missing] {split_dir.name}:{line_no} {wav_name}")
                continue

            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists() and not overwrite:
                continue

            if not dry_run:
                txt_path.write_text(text, encoding="utf-8")
            written += 1

    print(
        f"[done] {split_dir.name}: written={written}, missing_wav={missing}, wav_files={len(wav_index)}"
    )
    return written, missing


def main():
    parser = argparse.ArgumentParser(
        description="Generate same-name .txt files for AISHELL-3 wav files from content.txt."
    )
    parser.add_argument(
        "aishell3_root",
        type=Path,
        help="AISHELL-3 root path, e.g. /usr/local/corpus/zh/AISHELL-3",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process. Default: train test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print statistics; do not write files.",
    )
    args = parser.parse_args()

    total_written = 0
    total_missing = 0
    for split in args.splits:
        written, missing = process_split(
            args.aishell3_root / split,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_written += written
        total_missing += missing

    print(f"[summary] written={total_written}, missing_wav={total_missing}")


if __name__ == "__main__":
    main()
