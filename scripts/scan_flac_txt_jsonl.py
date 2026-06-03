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


def txt_path_for(
    flac_path: Path,
    flac_root: Path,
    txt_root: Path | None,
    txt_suffix: str,
    strip_stem_regex: re.Pattern[str] | None,
) -> Path:
    if txt_root is None:
        base_path = flac_path
    else:
        rel_path = flac_path.relative_to(flac_root)
        base_path = txt_root / rel_path

    stem = base_path.stem
    if strip_stem_regex is not None:
        stem = strip_stem_regex.sub("", stem)
    return base_path.with_name(stem + txt_suffix)


def has_text(path: Path) -> bool:
    return bool(path.read_text(encoding="utf-8").strip())


def main():
    parser = argparse.ArgumentParser(
        description="Build [flac, txt] jsonl by scanning files that have matching .flac and .txt pairs."
    )
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing .flac files.")
    parser.add_argument(
        "--scp",
        type=Path,
        default=None,
        help="Optional .scp file containing .flac paths. If set, only these files are checked.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .jsonl path.")
    parser.add_argument(
        "--txt-root",
        type=Path,
        default=None,
        help="Optional separate root for .txt files, matched by relative path from --root.",
    )
    parser.add_argument(
        "--txt-suffix",
        default=".txt",
        help="Text filename suffix replacing .flac, for example .txt, .normalized.txt, or .original.txt.",
    )
    parser.add_argument(
        "--strip-stem-regex",
        default=None,
        help="Optional regex removed from the .flac stem before building txt name, for example _mic[0-9]+$.",
    )
    parser.add_argument(
        "--skip-empty-txt",
        action="store_true",
        help="Skip pairs whose .txt file exists but is empty after stripping whitespace.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only check first N .flac files, useful for testing.")
    args = parser.parse_args()

    flac_root = args.root
    txt_root = args.txt_root
    if not flac_root.is_dir():
        raise SystemExit(f"root is not a directory: {flac_root}")
    if args.scp is not None and not args.scp.is_file():
        raise SystemExit(f"scp is not a file: {args.scp}")
    if txt_root is not None and not txt_root.is_dir():
        raise SystemExit(f"txt-root is not a directory: {txt_root}")
    if not args.txt_suffix.startswith("."):
        raise SystemExit(f"txt-suffix must start with '.': {args.txt_suffix}")
    try:
        strip_stem_regex = re.compile(args.strip_stem_regex) if args.strip_stem_regex else None
    except re.error as exc:
        raise SystemExit(f"bad strip-stem-regex: {exc}") from exc

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
        "missing_txt": 0,
        "empty_txt": 0,
        "selected": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for flac_path in tqdm(flac_files, desc="scan_flac_txt"):
            if not flac_path.exists():
                stats["missing_flac"] += 1
                continue
            txt_path = txt_path_for(flac_path, flac_root, txt_root, args.txt_suffix, strip_stem_regex)
            if not txt_path.exists():
                stats["missing_txt"] += 1
                continue
            if args.skip_empty_txt and not has_text(txt_path):
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
