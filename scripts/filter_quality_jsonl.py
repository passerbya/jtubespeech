#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm


_NEED_TN_RE = re.compile(
    r"[0-9]"
    r"|&"
    r"|[\u2070-\u209F\u20A0-\u20CF\u0024\u00A2-\u00A5\u0E3F\u17DB\u2103\u2109\u00B0]"
)


def need_text_normalization(text: str) -> bool:
    return bool(_NEED_TN_RE.search(text))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def txt_path_for(flac_path: Path) -> Path:
    return flac_path.with_suffix(".txt")


def whisper_path_for(flac_path: Path) -> Path:
    return flac_path.with_suffix(".whisper.txt")


def qwen_path_for(flac_path: Path) -> Path:
    return flac_path.with_suffix(".qwen.txt")


def load_scp(scp_path: Path):
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Path(line)


def load_flac_txt_jsonl(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json at {jsonl_path}:{line_no}: {exc}") from exc
            if not isinstance(item, list) or len(item) < 2:
                raise ValueError(f"bad item at {jsonl_path}:{line_no}: expected [flac, txt]")
            yield Path(item[0]), Path(item[1])


def comparable_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    drop_chars = set(string.punctuation)
    drop_chars.update(" \t\r\n，。！？、；：：“”‘’（）【】《》〈〉「」『』…—-·")
    return "".join(ch for ch in text if ch not in drop_chars)


def similarity(a: str, b: str) -> float:
    left = comparable_text(a)
    right = comparable_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def select_quality_pair(
    flac_path: Path,
    txt_path: Path,
    whisper_threshold: float,
    qwen_threshold: float,
    allow_missing_whisper: bool,
):
    if not txt_path.exists():
        return None, "missing_txt", 0.0

    txt = read_text(txt_path)
    if not txt:
        return None, "empty_txt", 0.0

    if need_text_normalization(txt):
        qwen_path = qwen_path_for(flac_path)
        if not qwen_path.exists():
            return None, "missing_qwen", 0.0
        score = similarity(txt, read_text(qwen_path))
        if score >= qwen_threshold:
            return [str(flac_path), str(qwen_path)], "qwen_ok", score
        return None, "qwen_low_similarity", score

    whisper_path = whisper_path_for(flac_path)
    if not whisper_path.exists():
        if allow_missing_whisper:
            return [str(flac_path), str(txt_path)], "whisper_ok", 1
        return None, "missing_whisper", 0.0
    score = similarity(txt, read_text(whisper_path))
    if score >= whisper_threshold:
        return [str(flac_path), str(txt_path)], "whisper_ok", score
    return None, "whisper_low_similarity", score


def main():
    parser = argparse.ArgumentParser(
        description="Build high-quality [flac, text] jsonl from a .scp flac list or an existing [flac, txt] jsonl."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--scp", type=Path, help=".scp file containing .flac paths.")
    input_group.add_argument("--jsonl", type=Path, help="Input flac_txt.jsonl whose items are [flac, txt].")
    parser.add_argument("--output", type=Path, required=True, help="Output .jsonl path.")
    parser.add_argument("--whisper-threshold", type=float, default=0.95)
    parser.add_argument("--qwen-threshold", type=float, default=0.80)
    parser.add_argument(
        "--allow-missing-whisper",
        action="store_true",
        help="If .whisper.txt is missing for non-TN text, keep [flac, txt] directly.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only check first N files, useful for testing.")
    args = parser.parse_args()

    if args.scp is not None:
        input_items = [(flac_path, txt_path_for(flac_path)) for flac_path in load_scp(args.scp)]
    else:
        input_items = list(load_flac_txt_jsonl(args.jsonl))
    if args.limit > 0:
        input_items = input_items[: args.limit]

    selected = 0
    stats = {
        "missing_flac": 0,
        "missing_txt": 0,
        "empty_txt": 0,
        "missing_whisper": 0,
        "whisper_low_similarity": 0,
        "whisper_ok": 0,
        "missing_qwen": 0,
        "qwen_low_similarity": 0,
        "qwen_ok": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for flac_path, txt_path in tqdm(input_items, desc="filter_quality"):
            if not flac_path.exists():
                stats["missing_flac"] += 1
                continue

            item, status, _score = select_quality_pair(
                flac_path,
                txt_path,
                whisper_threshold=args.whisper_threshold,
                qwen_threshold=args.qwen_threshold,
                allow_missing_whisper=args.allow_missing_whisper,
            )
            stats[status] = stats.get(status, 0) + 1
            if item is None:
                continue

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            selected += 1

    tmp_path.replace(args.output)

    print(f"done. files={len(input_items)} selected={selected} output={args.output}", flush=True)
    for key in sorted(stats):
        if stats[key]:
            print(f"{key}: {stats[key]}", flush=True)


if __name__ == "__main__":
    main()
