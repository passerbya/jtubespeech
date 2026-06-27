#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm


_NEED_TN_RE = re.compile(
    r"[0-9]"
    r"|&"
    r"|[\u2070-\u209F\u20A0-\u20CF\u0024\u00A2-\u00A5\u0E3F\u17DB\u2103\u2109\u00B0]"
)
SAFE_LANG_RE = re.compile(r"[^A-Za-z0-9_-]+")


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


def language_path_for(txt_path: Path) -> Path:
    return txt_path.with_suffix(".lang.txt")


def language_path_candidates(flac_path: Path, txt_path: Path):
    seen = set()
    for lang_path in (language_path_for(txt_path), flac_path.with_suffix(".lang.txt")):
        if lang_path in seen:
            continue
        seen.add(lang_path)
        yield lang_path


def read_language_code(lang_path: Path) -> str:
    for line in lang_path.read_text(encoding="utf-8-sig").splitlines():
        lang = line.strip()
        if lang:
            return lang
    return ""


def language_code_for(flac_path: Path, txt_path: Path) -> tuple[str | None, Path | None]:
    for lang_path in language_path_candidates(flac_path, txt_path):
        if lang_path.exists():
            return read_language_code(lang_path), lang_path
    return None, None


def safe_lang_for_filename(lang: str) -> str:
    safe_lang = SAFE_LANG_RE.sub("_", lang).strip("._-")
    return safe_lang or "unknown"


def output_path_for_language(output_path: Path, lang: str) -> Path:
    safe_lang = safe_lang_for_filename(lang)
    return output_path.with_name(f"{output_path.stem}.{safe_lang}{output_path.suffix}")


class SplitJsonlWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.handles = {}
        self.output_paths = {}
        self.tmp_paths = {}

    def write(self, item, lang: str | None):
        key = safe_lang_for_filename(lang) if lang else ""
        out = self._handle_for(key)
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        out.flush()

    def ensure_base_output(self):
        self._handle_for("")

    def _handle_for(self, key: str):
        if key not in self.handles:
            output_path = self.output_path if not key else output_path_for_language(self.output_path, key)
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            self.output_paths[key] = output_path
            self.tmp_paths[key] = tmp_path
            self.handles[key] = tmp_path.open("w", encoding="utf-8")
        return self.handles[key]

    def commit(self) -> dict[str, Path]:
        for out in self.handles.values():
            out.close()
        self.handles.clear()
        for key, tmp_path in self.tmp_paths.items():
            tmp_path.replace(self.output_paths[key])
        return self.output_paths

    def abort(self):
        for out in self.handles.values():
            out.close()
        self.handles.clear()
        for tmp_path in self.tmp_paths.values():
            if tmp_path.exists():
                tmp_path.unlink()


def load_scp(scp_path: Path):
    with scp_path.open("r", encoding="utf-8-sig") as f:
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
    output_counts = Counter()
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
        "missing_lang": 0,
        "empty_lang": 0,
        "lang_ok": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = SplitJsonlWriter(args.output)
    try:
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

            lang, lang_path = language_code_for(flac_path, txt_path)
            if lang_path is None:
                stats["missing_lang"] += 1
            elif not lang:
                stats["empty_lang"] += 1
            else:
                stats["lang_ok"] += 1

            writer.write(item, lang)
            output_counts[safe_lang_for_filename(lang) if lang else "base"] += 1
            selected += 1

        if selected == 0:
            writer.ensure_base_output()
    except Exception:
        writer.abort()
        raise

    output_paths = writer.commit()

    if len(output_paths) == 1:
        output_text = next(iter(output_paths.values()))
    else:
        output_text = f"{len(output_paths)} files"
    print(f"done. files={len(input_items)} selected={selected} output={output_text}", flush=True)
    for key, path in sorted(output_paths.items()):
        count_key = key or "base"
        print(f"output[{count_key}]: {output_counts[count_key]} -> {path}", flush=True)
    for key in sorted(stats):
        if stats[key]:
            print(f"{key}: {stats[key]}", flush=True)


if __name__ == "__main__":
    main()
