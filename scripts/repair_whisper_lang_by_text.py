#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_ROOT = Path("/usr/local/corpus/ja/ReazonSpeech/v2/extract")


def scandir_generator(path: Path) -> Iterable[Path]:
    with os.scandir(path) as it:
        for entry in it:
            entry_path = Path(entry.path)
            if entry.is_file():
                yield entry_path
            elif entry.is_dir():
                yield from scandir_generator(entry_path)


def iter_lang_files(root: Path):
    if root.is_file():
        if root.name.endswith(".lang.txt"):
            yield root
        elif root.suffix == ".flac":
            yield root.with_suffix(".lang.txt")
        else:
            raise ValueError(f"input file must be .lang.txt or .flac: {root}")
        return

    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")

    for lang_path in scandir_generator(root):
        if lang_path.name.endswith(".lang.txt"):
            yield lang_path


def whisper_path_for_lang_path(lang_path: Path) -> Path:
    name = lang_path.name
    if not name.endswith(".lang.txt"):
        raise ValueError(f"not a .lang.txt path: {lang_path}")
    return lang_path.with_name(name[: -len(".lang.txt")] + ".whisper.txt")


def read_text(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        return f.read(max_chars)


def char_script(ch: str) -> str | None:
    code = ord(ch)
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "katakana"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "hangul"
    if (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2EBEF
    ):
        return "han"

    name = unicodedata.name(ch, "")
    if "LATIN" in name:
        return "latin"
    return None


def count_scripts(text: str) -> Counter:
    counts = Counter()
    for ch in unicodedata.normalize("NFKC", text):
        script = char_script(ch)
        if script is not None:
            counts[script] += 1
    return counts


def detect_ja_ko(text: str, min_letters: int) -> tuple[str, Counter, int]:
    counts = count_scripts(text)
    total = sum(counts.values())
    if total < min_letters:
        return "unknown", counts, total

    kana = counts["hiragana"] + counts["katakana"]
    hangul = counts["hangul"]

    # Japanese speech can contain common Chinese characters, so kana is the
    # strongest signal. Check it before Hangul to tolerate short Korean tails
    # that Whisper sometimes appends, for example "이 시각 세계였습니다."
    if kana >= max(1, total * 0.08):
        return "ja", counts, total
    if hangul >= max(1, total * 0.20):
        return "ko", counts, total
    if counts["han"] >= max(2, total * 0.30):
        return "zh", counts, total
    return "unknown", counts, total


def write_language(lang_path: Path, language: str) -> None:
    tmp_path = lang_path.with_suffix(lang_path.suffix + ".tmp")
    tmp_path.write_text(language, encoding="utf-8")
    tmp_path.replace(lang_path)


def format_counts(counts: Counter, total: int) -> str:
    if total <= 0:
        return "letters=0"
    keys = ("hiragana", "katakana", "hangul", "han", "latin")
    parts = [f"letters={total}"]
    parts.extend(f"{key}={counts[key]}" for key in keys if counts[key])
    return " ".join(parts)


def process_one(lang_path: Path, args) -> str:
    if not lang_path.exists():
        print(f"[MISS_LANG] {lang_path}", flush=True)
        return "missing_lang"

    current_lang = lang_path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    if args.from_lang and current_lang != args.from_lang:
        return "skip_from_lang"

    whisper_path = whisper_path_for_lang_path(lang_path)
    if not whisper_path.exists():
        print(f"[MISS_WHISPER] {lang_path} whisper={whisper_path}", flush=True)
        return "missing_whisper"

    text = read_text(whisper_path, args.max_chars)
    detected, counts, total = detect_ja_ko(text, args.min_letters)
    detail = format_counts(counts, total)

    if detected != args.to_lang:
        print(
            f"[KEEP] {lang_path} current={current_lang} detected={detected} {detail}",
            flush=True,
        )
        return "keep"

    if args.dry_run:
        print(
            f"[DRY_FIX] {lang_path} {current_lang}->{args.to_lang} detected={detected} {detail}",
            flush=True,
        )
        return "dry_fix"

    write_language(lang_path, args.to_lang)
    print(
        f"[FIX] {lang_path} {current_lang}->{args.to_lang} detected={detected} {detail}",
        flush=True,
    )
    return "fixed"


def main():
    parser = argparse.ArgumentParser(
        description="Repair wrong Whisper .lang.txt codes after checking paired .whisper.txt text."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory to scan, or one .lang.txt/.flac file to repair.",
    )
    parser.add_argument("--from-lang", default="ko", help="Only repair files whose current language code matches this.")
    parser.add_argument("--to-lang", default="ja", help="Language code to write when text detection matches.")
    parser.add_argument("--max-chars", type=int, default=4096, help="Read at most this many chars from .whisper.txt.")
    parser.add_argument("--min-letters", type=int, default=2, help="Minimum script letters required for detection.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N .lang.txt files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be changed without writing files.")
    args = parser.parse_args()

    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be > 0")
    if args.min_letters <= 0:
        raise SystemExit("--min-letters must be > 0")
    if not args.to_lang:
        raise SystemExit("--to-lang must not be empty")

    stats = Counter()
    lang_files = iter_lang_files(args.root)
    if tqdm is not None:
        lang_files = tqdm(lang_files, desc="repair_lang")

    for idx, lang_path in enumerate(lang_files, start=1):
        if args.limit > 0 and idx > args.limit:
            break
        stats[process_one(lang_path, args)] += 1

    print("done.", flush=True)
    for key, count in sorted(stats.items()):
        print(f"{key}: {count}", flush=True)


if __name__ == "__main__":
    main()
