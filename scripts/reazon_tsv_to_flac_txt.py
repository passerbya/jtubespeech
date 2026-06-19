#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def parse_tsv_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line:
        return None

    if "\t" in line:
        parts = line.split("\t", 1)
    else:
        parts = line.split(maxsplit=1)

    if len(parts) != 2:
        return None

    rel_path = parts[0].strip()
    text = parts[1].strip()
    if not rel_path or not text:
        return None
    return rel_path, text


def rewrite_flac_to_mono_in_place(flac_path: Path, dry_run: bool = False) -> bool:
    info = sf.info(flac_path)
    if info.channels <= 1:
        return False

    if dry_run:
        return True

    audio, sample_rate = sf.read(flac_path, always_2d=True)
    mono = np.mean(audio, axis=1)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{flac_path.stem}.",
        suffix=".flac",
        dir=str(flac_path.parent),
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
        os.replace(tmp_path, flac_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return True


def process_reazon(
    reazon_root: Path,
    tsv_path: Path,
    audio_root: Path,
    overwrite_txt: bool,
    skip_mono_convert: bool,
    dry_run: bool,
    progress_interval: int,
) -> Tuple[int, int, int, int, int]:
    written_txt = 0
    skipped_txt = 0
    converted_mono = 0
    missing_audio = 0
    failed = 0

    with tsv_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            parsed = parse_tsv_line(line)
            if parsed is None:
                failed += 1
                print(f"[skip] invalid line {line_no}")
                continue

            rel_path, text = parsed
            flac_path = audio_root / rel_path
            txt_path = flac_path.with_suffix(".txt")

            if not flac_path.is_file():
                missing_audio += 1
                print(f"[missing-audio] {flac_path}")
                continue

            try:
                if not skip_mono_convert:
                    if rewrite_flac_to_mono_in_place(flac_path, dry_run=dry_run):
                        converted_mono += 1

                if txt_path.exists() and not overwrite_txt:
                    skipped_txt += 1
                else:
                    if not dry_run:
                        txt_path.write_text(text, encoding="utf-8")
                    written_txt += 1
            except Exception as exc:
                failed += 1
                print(f"[failed] {flac_path}: {exc}")

            if line_no % progress_interval == 0:
                print(
                    f"[progress] lines={line_no}, written_txt={written_txt}, "
                    f"skipped_txt={skipped_txt}, converted_mono={converted_mono}, "
                    f"missing_audio={missing_audio}, failed={failed}"
                )

    return written_txt, skipped_txt, converted_mono, missing_audio, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write same-name txt files for ReazonSpeech flac files and convert multi-channel flac to mono in-place."
    )
    parser.add_argument(
        "reazon_root",
        type=Path,
        nargs="?",
        default=Path("/usr/local/corpus/ja/ReazonSpeech/v2"),
        help="ReazonSpeech v2 root directory.",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        help="Path to v2.tsv. Default: <reazon_root>/v2.tsv",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Directory containing relative audio paths in TSV. Default: <reazon_root>/extract",
    )
    parser.add_argument(
        "--overwrite-txt",
        action="store_true",
        help="Overwrite existing txt files.",
    )
    parser.add_argument(
        "--skip-mono-convert",
        action="store_true",
        help="Only write txt files; do not rewrite stereo/multi-channel flac files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files; do not write txt or rewrite flac.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10000,
        help="Print progress every N TSV lines. Default: 10000",
    )
    args = parser.parse_args()

    reazon_root = args.reazon_root.resolve()
    tsv_path = args.tsv.resolve() if args.tsv else reazon_root / "v2.tsv"
    audio_root = args.audio_root.resolve() if args.audio_root else reazon_root / "extract"

    if not reazon_root.is_dir():
        raise SystemExit(f"ReazonSpeech root not found: {reazon_root}")
    if not tsv_path.is_file():
        raise SystemExit(f"TSV not found: {tsv_path}")
    if not audio_root.is_dir():
        raise SystemExit(f"audio root not found: {audio_root}")

    print(f"[root] {reazon_root}")
    print(f"[tsv] {tsv_path}")
    print(f"[audio-root] {audio_root}")

    written_txt, skipped_txt, converted_mono, missing_audio, failed = process_reazon(
        reazon_root=reazon_root,
        tsv_path=tsv_path,
        audio_root=audio_root,
        overwrite_txt=args.overwrite_txt,
        skip_mono_convert=args.skip_mono_convert,
        dry_run=args.dry_run,
        progress_interval=args.progress_interval,
    )

    txt_action = "would_write_txt" if args.dry_run else "written_txt"
    mono_action = "would_convert_mono" if args.dry_run else "converted_mono"
    print(
        f"[summary] {txt_action}={written_txt}, skipped_txt={skipped_txt}, "
        f"{mono_action}={converted_mono}, missing_audio={missing_audio}, failed={failed}"
    )


if __name__ == "__main__":
    main()
