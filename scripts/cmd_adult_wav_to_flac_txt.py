#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import soundfile as sf


SPECIAL_MARKERS = ("/", "*", "+")
# TTS filtering based on the corpus transcription guideline:
# o/: other speaker, n/: noise/background, u/: unintelligible utterance.
# Any remaining '/', '*', or '+' marks a transcription special case.
BAD_PREFIXES = {
    "o/": "other_speaker",
    "n/": "noise_or_background",
    "u/": "unintelligible",
}


def find_label_text(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "LabelText" and isinstance(value, str):
                return value
        for value in obj.values():
            found = find_label_text(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_label_text(item)
            if found is not None:
                return found
    return None


def find_json_for_wav(wav_path: Path) -> Optional[Path]:
    json_name = wav_path.with_suffix(".json").name
    candidates = [
        wav_path.with_suffix(".json"),
        wav_path.parent.parent / json_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_label_text(json_path: Path) -> Optional[str]:
    with json_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return find_label_text(data)


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


def remove_if_exists(path: Path, dry_run: bool) -> None:
    if path.exists() and not dry_run:
        path.unlink()


def scandir_generator(path: Path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(Path(entry.path))


def process_wav(
    wav_path: Path,
    overwrite: bool,
    delete_wav: bool,
    delete_filtered_wav: bool,
    dry_run: bool,
) -> Tuple[str, Optional[str]]:
    json_path = find_json_for_wav(wav_path)
    if json_path is None:
        return "missing_json", f"[missing-json] {wav_path}"

    text = load_label_text(json_path)
    if text is None:
        return "missing_text", f"[missing-text] {json_path}"

    text = clean_text(text)
    if not text:
        return "filtered", f"[filtered:empty] {wav_path}"

    should_filter, reason = should_filter_text(text)
    flac_path = wav_path.with_suffix(".flac")
    txt_path = wav_path.with_suffix(".txt")

    if should_filter:
        if delete_filtered_wav:
            remove_if_exists(wav_path, dry_run=dry_run)
        remove_if_exists(flac_path, dry_run=dry_run)
        remove_if_exists(txt_path, dry_run=dry_run)
        return "filtered", f"[filtered:{reason}] {wav_path}"

    if flac_path.exists() and txt_path.exists() and not overwrite:
        if delete_wav:
            remove_if_exists(wav_path, dry_run=dry_run)
        return "skipped", None

    if not dry_run:
        write_mono_flac(wav_path, flac_path)
        txt_path.write_text(text, encoding="utf-8")
        if delete_wav:
            wav_path.unlink()

    return "written", None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Korean cmd_adult wav files to mono flac, write same-name txt, and filter non-TTS labels."
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("/usr/local/corpus/ko/cmd_adult"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flac/txt files.",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Do not delete wav files after successful conversion.",
    )
    parser.add_argument(
        "--delete-filtered-wav",
        action="store_true",
        help="Also delete wav files filtered by o/, n/, u/, /, *, or +.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts; do not write or delete files.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N wav files. Default: 1000",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    written = 0
    skipped = 0
    filtered = 0
    missing_json = 0
    missing_text = 0
    failed = 0

    delete_wav = not args.keep_wav
    print(f"[root] {root}")

    for index, wav_path in enumerate(scandir_generator(root), 1):
        if wav_path.suffix.lower() != ".wav":
            continue

        try:
            status, message = process_wav(
                wav_path=wav_path,
                overwrite=args.overwrite,
                delete_wav=delete_wav,
                delete_filtered_wav=args.delete_filtered_wav,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failed += 1
            print(f"[failed] {wav_path}: {exc}")
            continue

        if status == "written":
            written += 1
        elif status == "skipped":
            skipped += 1
        elif status == "filtered":
            filtered += 1
        elif status == "missing_json":
            missing_json += 1
        elif status == "missing_text":
            missing_text += 1

        if message:
            print(message)

        if index % args.progress_interval == 0:
            print(
                f"[progress] seen={index}, written={written}, skipped={skipped}, "
                f"filtered={filtered}, missing_json={missing_json}, "
                f"missing_text={missing_text}, failed={failed}"
            )

    action = "would_write" if args.dry_run else "written"
    print(
        f"[summary] {action}={written}, skipped={skipped}, filtered={filtered}, "
        f"missing_json={missing_json}, missing_text={missing_text}, failed={failed}"
    )


if __name__ == "__main__":
    main()
