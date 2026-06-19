#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


def discover_subsets(jsut_root: Path) -> List[Path]:
    subsets: List[Path] = []
    for path in sorted(jsut_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "wav").is_dir() and (path / "transcript_utf8.txt").is_file():
            subsets.append(path)
    return subsets


def load_transcripts(path: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            if ":" not in line:
                print(f"[skip] {path}:{line_no} invalid transcript line")
                continue
            utt_id, text = line.split(":", 1)
            utt_id = utt_id.strip()
            text = text.strip()
            if utt_id:
                transcripts[utt_id] = text
    return transcripts


def flac_subtype_from_wav(wav_path: Path) -> str:
    subtype = sf.info(wav_path).subtype
    if subtype in {"PCM_16", "PCM_24"}:
        return subtype
    return "PCM_16"


def write_mono_flac(wav_path: Path, flac_path: Path) -> None:
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    if audio.shape[1] == 1:
        mono = audio[:, 0]
    else:
        mono = np.mean(audio, axis=1)

    flac_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        flac_path,
        mono,
        sample_rate,
        format="FLAC",
        subtype=flac_subtype_from_wav(wav_path),
    )


def process_subset(
    subset_dir: Path,
    output_dir_name: str,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int, int, int]:
    transcript_path = subset_dir / "transcript_utf8.txt"
    wav_dir = subset_dir / "wav"
    output_dir = subset_dir / output_dir_name

    transcripts = load_transcripts(transcript_path)
    written = 0
    skipped = 0
    missing_wav = 0
    failed = 0

    for utt_id, text in transcripts.items():
        wav_path = wav_dir / f"{utt_id}.wav"
        flac_path = output_dir / f"{utt_id}.flac"
        txt_path = output_dir / f"{utt_id}.txt"

        if not wav_path.is_file():
            missing_wav += 1
            print(f"[missing-wav] {wav_path}")
            continue

        if flac_path.exists() and txt_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            if not dry_run:
                write_mono_flac(wav_path, flac_path)
                txt_path.write_text(text, encoding="utf-8")
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[failed] {wav_path}: {exc}")

        if written and written % 500 == 0:
            print(f"[progress] {subset_dir.name}: written={written}")

    print(
        f"[done] {subset_dir.name}: written={written}, skipped={skipped}, "
        f"missing_wav={missing_wav}, failed={failed}"
    )
    return written, skipped, missing_wav, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSUT wav files to mono flac and write same-name txt transcripts."
    )
    parser.add_argument(
        "jsut_root",
        type=Path,
        nargs="?",
        default=Path("/usr/local/corpus/ja/jsut_ver1.1"),
        help="JSUT root directory. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="flac",
        help="Output directory name inside each subset. Default: flac",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flac/txt files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files; do not write anything.",
    )
    args = parser.parse_args()

    jsut_root = args.jsut_root.resolve()
    if not jsut_root.is_dir():
        raise SystemExit(f"JSUT root not found: {jsut_root}")

    subsets = discover_subsets(jsut_root)
    if not subsets:
        raise SystemExit(f"no JSUT subsets found under: {jsut_root}")

    print("[subsets] " + ", ".join(path.name for path in subsets))

    total_written = 0
    total_skipped = 0
    total_missing_wav = 0
    total_failed = 0

    for subset_dir in subsets:
        written, skipped, missing_wav, failed = process_subset(
            subset_dir,
            output_dir_name=args.output_dir_name,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_written += written
        total_skipped += skipped
        total_missing_wav += missing_wav
        total_failed += failed

    action = "would_write" if args.dry_run else "written"
    print(
        f"[summary] {action}={total_written}, skipped={total_skipped}, "
        f"missing_wav={total_missing_wav}, failed={total_failed}"
    )


if __name__ == "__main__":
    main()
