#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


DEFAULT_ROOT = Path("/usr/local/corpus/ko/emotion_for_tts")
SOURCE_DIR = Path("01.원천데이터") / "unzip"
LABEL_DIR = Path("02.라벨링데이터") / "unzip"
SPECIAL_MARKERS = ("*", "+")
# TTS filtering based on the corpus transcription guideline:
# o/: other speaker, n/: noise/background, u/: unintelligible utterance.
# Any remaining '*' or '+' marks a transcription special case.
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


def iter_records(data: object) -> Iterable[dict]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        yield data


def scandir_generator(path: Path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(Path(entry.path))


def build_wav_index(source_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for audio_path in scandir_generator(source_dir):
        suffix = audio_path.suffix.lower()
        if suffix != ".wav" and suffix != ".flac":
            continue
        wav_name = audio_path.with_suffix(".wav").name
        existing = index.get(wav_name)
        if existing is not None:
            print(f"[duplicate-wav] {wav_name}: keeping {existing}, ignoring {audio_path}")
            continue
        index[wav_name] = audio_path.with_suffix(".wav")
    return index


def text_from_sentence(sentence: dict, text_field: str) -> Optional[str]:
    voice_piece = sentence.get("voice_piece") or {}
    if text_field in {"tr", "ptr"}:
        text = voice_piece.get(text_field)
    else:
        text = sentence.get(text_field)
    if text is None:
        return None
    return str(text).strip()


def clean_text(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def replace_pause_slashes(text: str) -> str:
    return re.sub(r"\s*/\s*", ", ", text)


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
    split_dir: Path,
    text_field: str,
    delete_filtered_wav: bool,
    dry_run: bool,
) -> Tuple[List[Tuple[Path, Path, Path, str]], int, int, int]:
    label_dir = split_dir / LABEL_DIR
    source_dir = split_dir / SOURCE_DIR
    if not label_dir.is_dir():
        raise SystemExit(f"label dir not found: {label_dir}")
    if not source_dir.is_dir():
        raise SystemExit(f"source wav dir not found: {source_dir}")

    wav_index = build_wav_index(source_dir)
    tasks: List[Tuple[Path, Path, Path, str]] = []
    missing_wav = 0
    missing_text = 0
    filtered = 0

    for json_path in scandir_generator(label_dir):
        if json_path.suffix.lower() != ".json":
            continue
        for record in iter_records(load_json(json_path)):
            for sentence in record.get("sentences", []):
                voice_piece = sentence.get("voice_piece") or {}
                wav_name = voice_piece.get("filename")
                if not wav_name:
                    missing_wav += 1
                    print(f"[missing-filename] {json_path}: {sentence.get('id', '<unknown>')}")
                    continue

                text = text_from_sentence(sentence, text_field)
                if not text:
                    missing_text += 1
                    print(f"[missing-text] {json_path}: {wav_name}")
                    continue

                text = clean_text(text)
                should_filter, reason = should_filter_text(text)
                if should_filter and reason != "special_marker":
                    filtered += 1
                    wav_basename = Path(wav_name).with_suffix(".wav").name
                    wav_path = wav_index.get(wav_basename, source_dir / wav_basename)
                    flac_path = wav_path.with_suffix(".flac")
                    txt_path = wav_path.with_suffix(".txt")
                    remove_if_exists(flac_path, dry_run=dry_run)
                    remove_if_exists(txt_path, dry_run=dry_run)
                    if delete_filtered_wav:
                        remove_if_exists(wav_path, dry_run=dry_run)
                    print(f"[filtered:{reason}] {json_path}: {wav_name}")
                    continue

                text = replace_pause_slashes(text)
                should_filter, reason = should_filter_text(text)
                if should_filter:
                    filtered += 1
                    wav_basename = Path(wav_name).with_suffix(".wav").name
                    wav_path = wav_index.get(wav_basename, source_dir / wav_basename)
                    flac_path = wav_path.with_suffix(".flac")
                    txt_path = wav_path.with_suffix(".txt")
                    remove_if_exists(flac_path, dry_run=dry_run)
                    remove_if_exists(txt_path, dry_run=dry_run)
                    if delete_filtered_wav:
                        remove_if_exists(wav_path, dry_run=dry_run)
                    print(f"[filtered:{reason}] {json_path}: {wav_name}")
                    continue

                wav_path = wav_index.get(Path(wav_name).with_suffix(".wav").name)
                if wav_path is None:
                    missing_wav += 1
                    print(f"[missing-wav] {json_path}: {wav_name}")
                    continue

                flac_path = wav_path.with_suffix(".flac")
                txt_path = wav_path.with_suffix(".txt")

                tasks.append((wav_path, flac_path, txt_path, text))

    return tasks, missing_wav, missing_text, filtered


def convert_one(
    task: Tuple[Path, Path, Path, str],
    overwrite: bool,
    keep_wav: bool,
    dry_run: bool,
) -> str:
    wav_path, flac_path, txt_path, text = task
    if flac_path.exists() and txt_path.exists() and not overwrite:
        if not keep_wav and not dry_run and wav_path.exists():
            wav_path.unlink()
        return "skipped"

    if dry_run:
        return "written"

    write_mono_flac(wav_path, flac_path)
    txt_path.write_text(text + "\n", encoding="utf-8")
    if not keep_wav:
        wav_path.unlink()
    return "written"


def process_split(
    split_dir: Path,
    text_field: str,
    overwrite: bool,
    keep_wav: bool,
    delete_filtered_wav: bool,
    progress_interval: int,
    dry_run: bool,
) -> Tuple[int, int, int, int, int]:
    tasks, missing_wav, missing_text, filtered = collect_tasks(
        split_dir,
        text_field,
        delete_filtered_wav,
        dry_run,
    )

    written = 0
    skipped = 0
    failed = 0
    for task in tasks:
        try:
            status = convert_one(task, overwrite, keep_wav, dry_run)
        except Exception as exc:
            failed += 1
            print(f"[failed] {exc}")
            continue
        if status == "written":
            written += 1
        elif status == "skipped":
            skipped += 1

        done = written + skipped + failed
        if progress_interval > 0 and done and done % progress_interval == 0:
            print(f"[progress] {split_dir.name}: done={done}/{len(tasks)}")

    action = "would_write" if dry_run else "written"
    print(
        f"[done] {split_dir.name}: {action}={written}, skipped={skipped}, "
        f"filtered={filtered}, missing_wav={missing_wav}, "
        f"missing_text={missing_text}, failed={failed}"
    )
    return written, skipped, filtered, missing_wav + missing_text, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Korean emotion/style TTS sentence wav files to mono flac and "
            "write same-stem txt transcripts from label JSON files."
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
        "--splits",
        nargs="+",
        default=["Training", "Validation"],
        help="Split directories to process. Default: Training Validation",
    )
    parser.add_argument(
        "--text-field",
        choices=["tr", "ptr", "origin_text"],
        default="tr",
        help="Transcript field for txt. Default: tr",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flac/txt files.",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep source wav files after successful conversion or skip.",
    )
    parser.add_argument(
        "--delete-filtered-wav",
        action="store_true",
        help="Delete source wav files for filtered transcripts.",
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
        help="Only count work; do not write files.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    total_written = 0
    total_skipped = 0
    total_filtered = 0
    total_missing = 0
    total_failed = 0
    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"split dir not found: {split_dir}")
        written, skipped, filtered, missing, failed = process_split(
            split_dir=split_dir,
            text_field=args.text_field,
            overwrite=args.overwrite,
            keep_wav=args.keep_wav,
            delete_filtered_wav=args.delete_filtered_wav,
            progress_interval=max(0, args.progress_interval),
            dry_run=args.dry_run,
        )
        total_written += written
        total_skipped += skipped
        total_filtered += filtered
        total_missing += missing
        total_failed += failed

    action = "would_write" if args.dry_run else "written"
    print(
        f"[summary] {action}={total_written}, skipped={total_skipped}, "
        f"filtered={total_filtered}, missing={total_missing}, "
        f"failed={total_failed}"
    )


if __name__ == "__main__":
    main()
