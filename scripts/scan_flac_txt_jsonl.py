#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


def iter_flac_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".flac":
            yield path


def load_scp(scp_path: Path):
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Path(line)


def read_language_code(lang_path: Path) -> str:
    for line in lang_path.read_text(encoding="utf-8-sig").splitlines():
        lang = line.strip()
        if lang:
            return lang
    return ""


def same_language(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def txt_path_for(flac_path: Path, target_lang: str) -> Path | None:
    """Find the text paired with an audio file.

    When ``target_lang`` is provided, Whisper's language sidecar is the
    source of truth: the sidecar must match before selecting ``.whisper.txt``.
    The caller checks that the selected text file exists and is non-empty.
    The regular text fallbacks are kept for the language-agnostic mode.
    """

    target_lang = target_lang.strip()
    if target_lang:
        lang_path = flac_path.with_suffix(".lang.txt")
        if not lang_path.is_file():
            return None
        lang = read_language_code(lang_path)
        if not lang or not same_language(lang, target_lang):
            return None
        return flac_path.with_suffix(".whisper.txt")

    txt_path = flac_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path

    normalized_path = flac_path.with_suffix(".normalized.txt")
    if normalized_path.exists():
        return normalized_path

    original_path = flac_path.with_suffix(".original.txt")
    if original_path.exists():
        return original_path

    vctk_txt_path = Path(
        str(txt_path).replace("wav48_silence_trimmed", "txt").replace("wav48", "txt")
    )
    strip_stem_regex = re.compile(r"_mic[0-9]+$")
    vctk_txt_path = vctk_txt_path.with_stem(
        strip_stem_regex.sub("", vctk_txt_path.stem)
    )
    if vctk_txt_path.exists():
        return vctk_txt_path

    # 用于后续报 missing txt
    return vctk_txt_path


def has_text(path: Path) -> bool:
    return bool(path.read_text(encoding="utf-8").strip())


def main():
    parser = argparse.ArgumentParser(
        description="Build [flac, txt] jsonl by scanning files that have matching .flac and .txt pairs."
    )
    parser.add_argument("--root", type=Path, default=None, help="Root directory containing .flac files.")
    parser.add_argument(
        "--scp",
        type=Path,
        default=None,
        help="Optional .scp file containing .flac paths. If set, only these files are checked.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .jsonl path.")
    parser.add_argument("--limit", type=int, default=0, help="Only check first N .flac files, useful for testing.")
    parser.add_argument(
        "--lang",
        default="",
        help="Keep matching .lang.txt codes (e.g. zh/en/ja) and use .whisper.txt only.",
    )
    args = parser.parse_args()

    flac_root = args.root
    if flac_root is not None and not flac_root.is_dir():
        raise SystemExit(f"root is not a directory: {flac_root}")
    if args.scp is not None and not args.scp.is_file():
        raise SystemExit(f"scp is not a file: {args.scp}")
    if args.root is None and args.scp is None:
        parser.error("at least one of --root or --scp is required")

    if args.scp is None:
        flac_files = sorted(iter_flac_files(flac_root))
    else:
        flac_files = list(load_scp(args.scp))
    if args.limit > 0:
        flac_files = flac_files[: args.limit]

    selected = 0
    stats = {
        "scanned_flac": len(flac_files),
        "missing_flac": 0,
        "language_filtered": 0,
        "missing_txt": 0,
        "empty_txt": 0,
        "selected": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for flac_path in tqdm(flac_files, desc="scan_flac_txt"):
            if not flac_path.is_file():
                stats["missing_flac"] += 1
                continue
            txt_path = txt_path_for(flac_path, args.lang)
            if txt_path is None:
                stats["language_filtered"] += 1
                continue
            if not txt_path.is_file():
                stats["missing_txt"] += 1
                continue
            if not has_text(txt_path):
                stats["empty_txt"] += 1
                continue

            out.write(json.dumps([str(flac_path), str(txt_path)], ensure_ascii=False) + "\n")
            selected += 1

    stats["selected"] = selected
    tmp_path.replace(args.output)

    print(f"done. files={len(flac_files)} selected={selected} output={args.output}", flush=True)
    for key in sorted(stats):
        if stats[key]:
            print(f"{key}: {stats[key]}", flush=True)


if __name__ == "__main__":
    main()
