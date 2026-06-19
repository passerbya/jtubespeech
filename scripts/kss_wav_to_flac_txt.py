#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


DEFAULT_ROOT = Path("/usr/local/corpus/ko/KSS_Dataset")
DEFAULT_TRANSCRIPT = Path("origin/transcript.v.1.3.txt")
DEFAULT_SOURCE_DIR = Path("origin/kss")
DEFAULT_OUTPUT_DIR = Path("flac")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def normalize_rel_wav_path(raw_path: str) -> Path:
    rel_path = Path(raw_path.strip().lstrip("/\\"))
    if not rel_path.parts or rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"unsafe transcript path: {raw_path}")
    if rel_path.suffix.lower() != ".wav":
        raise ValueError(f"non-wav transcript path: {raw_path}")
    return rel_path


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

    sf.write(
        flac_path,
        mono,
        sample_rate,
        format="FLAC",
        subtype=flac_subtype_from_wav(wav_path),
    )


def load_tasks(
    transcript_path: Path,
    source_dir: Path,
    output_dir: Path,
) -> Tuple[List[Tuple[Path, Path, Path, str]], int, int, int]:
    tasks: List[Tuple[Path, Path, Path, str]] = []
    malformed = 0
    missing_text = 0
    duplicate = 0
    seen = set()

    with transcript_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 3:
                malformed += 1
                print(f"[malformed] line={line_no}: expected at least 3 fields")
                continue

            try:
                rel_wav_path = normalize_rel_wav_path(parts[0])
            except ValueError as exc:
                malformed += 1
                print(f"[malformed] line={line_no}: {exc}")
                continue

            text = clean_text(parts[2])
            if not text:
                missing_text += 1
                print(f"[missing-text] line={line_no}: {rel_wav_path}")
                continue

            if rel_wav_path in seen:
                duplicate += 1
                print(f"[duplicate] line={line_no}: {rel_wav_path}")
                continue
            seen.add(rel_wav_path)

            wav_path = source_dir / rel_wav_path
            flac_path = (output_dir / rel_wav_path).with_suffix(".flac")
            txt_path = flac_path.with_suffix(".txt")
            tasks.append((wav_path, flac_path, txt_path, text))

    return tasks, malformed, missing_text, duplicate


def convert_one(
    task: Tuple[Path, Path, Path, str],
    overwrite: bool,
    delete_wav: bool,
    dry_run: bool,
) -> str:
    wav_path, flac_path, txt_path, text = task
    need_flac = overwrite or not flac_path.exists()
    need_txt = overwrite or not txt_path.exists()

    if not need_flac and not need_txt:
        if delete_wav and not dry_run and wav_path.exists():
            wav_path.unlink()
        return "skipped"

    if need_flac and not wav_path.is_file():
        return "missing_wav"

    if dry_run:
        return "written"

    flac_path.parent.mkdir(parents=True, exist_ok=True)
    if need_flac:
        write_mono_flac(wav_path, flac_path)
    if need_txt:
        txt_path.write_text(text + "\n", encoding="utf-8")
    if delete_wav and wav_path.exists() and flac_path.exists():
        wav_path.unlink()
    return "written"


def process_dataset(
    dataset_root: Path,
    transcript_path: Path,
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
    delete_wav: bool,
    progress_interval: int,
    dry_run: bool,
) -> Tuple[int, int, int, int, int, int, int]:
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    if not transcript_path.is_file():
        raise SystemExit(f"transcript not found: {transcript_path}")
    if not source_dir.is_dir():
        raise SystemExit(f"source wav dir not found: {source_dir}")

    tasks, malformed, missing_text, duplicate = load_tasks(
        transcript_path,
        source_dir,
        output_dir,
    )

    written = 0
    skipped = 0
    missing_wav = 0
    failed = 0
    for task in tasks:
        try:
            status = convert_one(task, overwrite, delete_wav, dry_run)
        except Exception as exc:
            failed += 1
            print(f"[failed] {task[0]}: {exc}")
            continue

        if status == "written":
            written += 1
        elif status == "skipped":
            skipped += 1
        elif status == "missing_wav":
            missing_wav += 1
            print(f"[missing-wav] {task[0]}")

        done = written + skipped + missing_wav + failed
        if progress_interval > 0 and done and done % progress_interval == 0:
            print(f"[progress] done={done}/{len(tasks)}")

    action = "would_write" if dry_run else "written"
    print(
        f"[done] {action}={written}, skipped={skipped}, "
        f"missing_wav={missing_wav}, missing_text={missing_text}, "
        f"malformed={malformed}, duplicate={duplicate}, failed={failed}"
    )
    return written, skipped, missing_wav, missing_text, malformed, duplicate, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert KSS wav files to mono flac and write same-stem txt "
            "transcripts from transcript.v.1.3.txt."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Dataset root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help=f"Transcript file. Default: <dataset_root>/{DEFAULT_TRANSCRIPT}",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=f"Source wav dir. Default: <dataset_root>/{DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output flac dir. Default: <dataset_root>/{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flac/txt files.",
    )
    parser.add_argument(
        "--delete-wav",
        action="store_true",
        help="Delete source wav files after successful conversion or skip.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Print progress every N processed entries. Use 0 to disable. Default: 500",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count work; do not write or delete files.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    transcript_path = (
        args.transcript.resolve()
        if args.transcript is not None
        else dataset_root / DEFAULT_TRANSCRIPT
    )
    source_dir = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else dataset_root / DEFAULT_SOURCE_DIR
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_root / DEFAULT_OUTPUT_DIR
    )

    process_dataset(
        dataset_root=dataset_root,
        transcript_path=transcript_path,
        source_dir=source_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        delete_wav=args.delete_wav,
        progress_interval=max(0, args.progress_interval),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
