#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


DEFAULT_ROOT = Path("/usr/local/corpus/ko/conversation_child")
DEFAULT_SPLITS = ("Training", "Validation")
UNZIP_DIR = Path("unzip")
UTTERANCE_INFO_KEY = "발화정보"
SPECIAL_MARKERS = ("/", "*", "+")
BAD_PREFIXES = {
    "o/": "other_speaker",
    "n/": "noise_or_background",
    "u/": "unintelligible",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def scandir_generator(path: Path) -> Iterable[Path]:
    with os.scandir(path) as it:
        for entry in it:
            entry_path = Path(entry.path)
            if entry.is_file():
                yield entry_path
            elif entry.is_dir():
                yield from scandir_generator(entry_path)


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def should_filter_text(text: str) -> Tuple[bool, str]:
    stripped = text.strip()
    lower = stripped.lower()
    for prefix, reason in BAD_PREFIXES.items():
        if lower.find(prefix) != -1:
            return True, reason
    if any(marker in stripped for marker in SPECIAL_MARKERS):
        return True, "special_marker"
    return False, ""


def remove_if_exists(path: Path, dry_run: bool) -> None:
    if path.exists() and not dry_run:
        path.unlink()


def wav_candidates(json_path: Path, record: dict) -> List[Path]:
    utterance_info = record.get(UTTERANCE_INFO_KEY) or {}
    file_name = utterance_info.get("fileNm")
    if not file_name:
        return []
    wav_path = json_path.parent / str(file_name)
    if wav_path.suffix.lower() == ".wavp" and not wav_path.is_file():
        wav_path = wav_path.with_suffix(".wav")
    return [wav_path]


def find_wav_path(json_path: Path, record: dict) -> Optional[Path]:
    candidates = wav_candidates(json_path, record)
    for wav_path in candidates:
        if wav_path.is_file():
            return wav_path
    return candidates[0] if candidates else None


def text_from_record(record: dict) -> str:
    utterance_info = record.get(UTTERANCE_INFO_KEY) or {}
    text = utterance_info.get("stt")
    if text is None:
        return ""
    return clean_text(str(text).strip())


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


def collect_tasks(
        input_dir: Path,
        delete_filtered_wav: bool,
        dry_run: bool,
) -> Tuple[List[Tuple[Path, Path, Path, str]], int, int, int, int]:
    tasks: List[Tuple[Path, Path, Path, str]] = []
    malformed = 0
    missing_wav = 0
    missing_text = 0
    filtered = 0

    for json_path in scandir_generator(input_dir):
        if json_path.suffix.lower() != ".json":
            continue

        record = load_json(json_path)
        if not isinstance(record, dict):
            malformed += 1
            print(f"[malformed] {json_path}: expected JSON object")
            continue

        wav_path = find_wav_path(json_path, record)
        if wav_path is None or not wav_path.is_file():
            missing_wav += 1
            expected = wav_path if wav_path is not None else json_path.with_suffix(".wav")
            print(f"[missing-wav] {json_path}: expected {expected}")
            continue

        text = text_from_record(record)
        if not text:
            missing_text += 1
            print(f"[missing-text] {json_path}: {wav_path.name}")
            continue

        flac_path = wav_path.with_suffix(".flac")
        txt_path = wav_path.with_suffix(".txt")
        should_filter, reason = should_filter_text(text)
        if should_filter:
            filtered += 1
            remove_if_exists(flac_path, dry_run=dry_run)
            remove_if_exists(txt_path, dry_run=dry_run)
            if delete_filtered_wav:
                remove_if_exists(wav_path, dry_run=dry_run)
            print(f"[filtered:{reason}] {json_path}: {wav_path.name}")
            continue

        tasks.append((wav_path, flac_path, txt_path, text))

    return tasks, malformed, missing_wav, missing_text, filtered


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

    if need_flac:
        write_mono_flac(wav_path, flac_path)
    if need_txt:
        txt_path.write_text(text + "\n", encoding="utf-8")
    if delete_wav and wav_path.exists() and flac_path.exists() and txt_path.exists():
        wav_path.unlink()
    return "written"


def process_input_dir(
        input_dir: Path,
        overwrite: bool,
        delete_wav: bool,
        progress_interval: int,
        dry_run: bool,
) -> Tuple[int, int, int, int, int, int, int]:
    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")

    tasks, malformed, missing_wav, missing_text, filtered = collect_tasks(
        input_dir,
        delete_filtered_wav=delete_wav,
        dry_run=dry_run,
    )

    written = 0
    skipped = 0
    failed = 0
    runtime_missing_wav = 0
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
            runtime_missing_wav += 1
            print(f"[missing-wav] {task[0]}")

        done = written + skipped + failed + runtime_missing_wav
        if progress_interval > 0 and done and done % progress_interval == 0:
            print(f"[progress] {input_dir.name}: done={done}/{len(tasks)}")

    missing_wav += runtime_missing_wav
    action = "would_write" if dry_run else "written"
    print(
        f"[done] {input_dir}: {action}={written}, skipped={skipped}, "
        f"filtered={filtered}, missing_wav={missing_wav}, "
        f"missing_text={missing_text}, malformed={malformed}, failed={failed}"
    )
    return written, skipped, filtered, missing_wav, missing_text, malformed, failed


def input_dirs_from_args(dataset_root: Path, splits: List[str]) -> List[Path]:
    if (dataset_root / UNZIP_DIR).is_dir():
        return [dataset_root / UNZIP_DIR]
    if any(dataset_root.glob("*.json")):
        return [dataset_root]
    return [dataset_root / split / UNZIP_DIR for split in splits]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert conversation_general wav files to mono flac in-place and "
            "write same-stem txt files from 발화정보.stt."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Dataset root, split dir, or unzip dir. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split directories to process when dataset_root is the dataset root. "
             "Default: Training Validation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flac/txt files.",
    )
    parser.add_argument(
        "--delete-wav",
        action="store_true",
        help="Delete source wav files after successful conversion, skip, or filter.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Print progress every N processed tasks. Use 0 to disable. Default: 500",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count work; do not write or delete files.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    input_dirs = input_dirs_from_args(dataset_root, args.splits)

    total_written = 0
    total_skipped = 0
    total_filtered = 0
    total_missing_wav = 0
    total_missing_text = 0
    total_malformed = 0
    total_failed = 0
    for input_dir in input_dirs:
        written, skipped, filtered, missing_wav, missing_text, malformed, failed = process_input_dir(
            input_dir=input_dir,
            overwrite=args.overwrite,
            delete_wav=args.delete_wav,
            progress_interval=max(0, args.progress_interval),
            dry_run=args.dry_run,
        )
        total_written += written
        total_skipped += skipped
        total_filtered += filtered
        total_missing_wav += missing_wav
        total_missing_text += missing_text
        total_malformed += malformed
        total_failed += failed

    action = "would_write" if args.dry_run else "written"
    print(
        f"[summary] {action}={total_written}, skipped={total_skipped}, "
        f"filtered={total_filtered}, missing_wav={total_missing_wav}, "
        f"missing_text={total_missing_text}, malformed={total_malformed}, "
        f"failed={total_failed}"
    )


if __name__ == "__main__":
    main()
