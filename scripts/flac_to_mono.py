#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def convert_flac_to_mono(path: Path, dry_run: bool = False) -> str:
    info = sf.info(path)
    if info.channels <= 1:
        return "skipped"

    audio, sample_rate = sf.read(path, always_2d=True)
    mono = np.mean(audio, axis=1)

    if dry_run:
        return "converted"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".flac",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        sf.write(
            tmp_path,
            mono,
            sample_rate,
            format="FLAC",
            subtype=info.subtype,
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return "converted"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert stereo/multi-channel FLAC files to mono in-place, preserving sample rate and FLAC subtype."
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("/usr/local/corpus/ja/japanese-anime-speech/data"),
        help="Directory to scan recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files; do not modify anything.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"directory not found: {root}")

    converted = 0
    skipped = 0
    failed = 0

    for flac_path in root.rglob("*.flac"):
        try:
            status = convert_flac_to_mono(flac_path, dry_run=args.dry_run)
        except Exception as exc:
            failed += 1
            print(f"[failed] {flac_path}: {exc}")
            continue

        if status == "converted":
            converted += 1
            print(f"[mono] {flac_path}")
        else:
            skipped += 1

        if converted and converted % 500 == 0:
            print(f"[progress] converted={converted}, skipped={skipped}, failed={failed}")

    action = "would_convert" if args.dry_run else "converted"
    print(f"[summary] {action}={converted}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
