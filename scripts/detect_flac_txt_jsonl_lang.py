#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

try:
    import langid
except ImportError:  # pragma: no cover - optional dependency
    langid = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


LANGID_ALIAS = {
    "an": "es",
    "jw": "jv",
}

SCRIPT_TO_LANG = {
    "hangul": "ko",
    "thai": "th",
    "lao": "lo",
    "khmer": "km",
    "myanmar": "my",
    "hebrew": "he",
    "greek": "el",
    "armenian": "hy",
    "georgian": "ka",
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "oriya": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "sinhala": "si",
    "ethiopic": "am",
}

ARABIC_LANGS = {"ar", "fa", "ur", "ps", "sd", "ug"}
CYRILLIC_LANGS = {"ru", "uk", "bg", "mk", "mn", "sr", "be", "kk", "ky", "tg"}
DEVANAGARI_LANGS = {"hi", "ne", "mr", "sa"}

VIETNAMESE_RE = re.compile(r"[\u0102\u0103\u0110\u0111\u01A0\u01A1\u01AF\u01B0\u1EA0-\u1EF9]")
LATIN_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
LATIN_STOPWORDS = {
    "en": {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "do",
        "for",
        "have",
        "i",
        "in",
        "is",
        "it",
        "not",
        "of",
        "on",
        "that",
        "the",
        "this",
        "to",
        "we",
        "what",
        "with",
        "you",
    },
    "es": {
        "como",
        "con",
        "de",
        "el",
        "en",
        "es",
        "esta",
        "la",
        "las",
        "lo",
        "los",
        "me",
        "no",
        "para",
        "por",
        "que",
        "se",
        "si",
        "un",
        "una",
        "y",
    },
    "fr": {
        "ce",
        "dans",
        "de",
        "des",
        "est",
        "et",
        "il",
        "je",
        "la",
        "le",
        "les",
        "nous",
        "pas",
        "pour",
        "que",
        "qui",
        "un",
        "une",
        "vous",
    },
    "de": {
        "auf",
        "das",
        "den",
        "der",
        "die",
        "du",
        "ein",
        "eine",
        "es",
        "ich",
        "in",
        "ist",
        "mit",
        "nicht",
        "sie",
        "und",
        "wir",
        "zu",
    },
    "pt": {
        "a",
        "as",
        "com",
        "de",
        "e",
        "ela",
        "ele",
        "em",
        "esta",
        "eu",
        "nao",
        "o",
        "os",
        "para",
        "por",
        "que",
        "um",
        "uma",
        "voce",
    },
    "it": {
        "che",
        "con",
        "di",
        "e",
        "gli",
        "il",
        "in",
        "io",
        "la",
        "le",
        "lo",
        "non",
        "per",
        "sono",
        "tu",
        "un",
        "una",
    },
    "id": {
        "ada",
        "dan",
        "dari",
        "di",
        "dia",
        "ini",
        "itu",
        "kamu",
        "ke",
        "saya",
        "tidak",
        "untuk",
        "yang",
    },
    "nl": {
        "dat",
        "de",
        "die",
        "een",
        "en",
        "het",
        "ik",
        "is",
        "jij",
        "met",
        "niet",
        "op",
        "van",
        "voor",
    },
}


SAFE_LANG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_lang_for_filename(lang: str) -> str:
    safe_lang = SAFE_LANG_RE.sub("_", lang).strip("._-")
    return safe_lang or "unknown"


def output_path_for(jsonl_path: Path, lang: str) -> Path:
    safe_lang = safe_lang_for_filename(lang)
    return jsonl_path.with_name(f"{jsonl_path.stem}.{safe_lang}.jsonl")


def iter_flac_txt_jsonl(jsonl_path: Path):
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
            yield line_no, Path(item[0]), Path(item[1])


def read_sample_text(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        return f.read(max_chars)


def char_script(ch: str) -> str | None:
    code = ord(ch)
    if (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2EBEF
    ):
        return "han"
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "katakana"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "hangul"
    if 0x0E00 <= code <= 0x0E7F:
        return "thai"
    if 0x0E80 <= code <= 0x0EFF:
        return "lao"
    if 0x1780 <= code <= 0x17FF:
        return "khmer"
    if 0x1000 <= code <= 0x109F or 0xA9E0 <= code <= 0xA9FF or 0xAA60 <= code <= 0xAA7F:
        return "myanmar"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
        return "arabic"
    if 0x0590 <= code <= 0x05FF:
        return "hebrew"
    if 0x0400 <= code <= 0x052F:
        return "cyrillic"
    if 0x0900 <= code <= 0x097F:
        return "devanagari"
    if 0x0980 <= code <= 0x09FF:
        return "bengali"
    if 0x0A00 <= code <= 0x0A7F:
        return "gurmukhi"
    if 0x0A80 <= code <= 0x0AFF:
        return "gujarati"
    if 0x0B00 <= code <= 0x0B7F:
        return "oriya"
    if 0x0B80 <= code <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= code <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= code <= 0x0CFF:
        return "kannada"
    if 0x0D00 <= code <= 0x0D7F:
        return "malayalam"
    if 0x0D80 <= code <= 0x0DFF:
        return "sinhala"
    if 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
        return "greek"
    if 0x0530 <= code <= 0x058F:
        return "armenian"
    if 0x10A0 <= code <= 0x10FF or 0x1C90 <= code <= 0x1CBF:
        return "georgian"
    if 0x1200 <= code <= 0x137F:
        return "ethiopic"

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


def langid_classify(text: str) -> str | None:
    if langid is None:
        return None
    lang, _score = langid.classify(text.lower())
    return LANGID_ALIAS.get(lang, lang)


def latin_fallback_classify(text: str) -> str | None:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    tokens = LATIN_WORD_RE.findall(folded)
    if not tokens:
        return None

    scores = {
        lang: sum(1 for token in tokens if token in stopwords)
        for lang, stopwords in LATIN_STOPWORDS.items()
    }
    lang, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return None
    return lang


def pick_langid_for_group(text: str, allowed: set[str]) -> str | None:
    lang = langid_classify(text)
    if lang in allowed:
        return lang
    return None


def detect_language(text: str, min_letters: int) -> str:
    text = text.strip()
    if not text:
        return "empty_txt"

    counts = count_scripts(text)
    total = sum(counts.values())
    if total < min_letters:
        return "unknown"

    kana_count = counts["hiragana"] + counts["katakana"]
    if kana_count >= max(2, total * 0.08):
        return "ja"
    if counts["hangul"] >= max(2, total * 0.20):
        return "ko"
    if counts["han"] >= max(2, total * 0.30):
        return "zh"

    top_script, top_count = counts.most_common(1)[0]
    top_ratio = top_count / total

    if top_script in SCRIPT_TO_LANG and top_ratio >= 0.30:
        return SCRIPT_TO_LANG[top_script]

    if top_script == "arabic" and top_ratio >= 0.30:
        return pick_langid_for_group(text, ARABIC_LANGS) or "ar"
    if top_script == "cyrillic" and top_ratio >= 0.30:
        return pick_langid_for_group(text, CYRILLIC_LANGS) or "ru"
    if top_script == "devanagari" and top_ratio >= 0.30:
        return pick_langid_for_group(text, DEVANAGARI_LANGS) or "hi"

    if counts["latin"] >= max(2, total * 0.50):
        if VIETNAMESE_RE.search(text):
            return "vi"
        return langid_classify(text) or latin_fallback_classify(text) or "latin"

    lang = langid_classify(text)
    return lang or "unknown"


class LanguageJsonlWriter:
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.handles = {}
        self.tmp_paths = {}
        self.output_paths = {}

    def write(self, lang: str, flac_path: Path, txt_path: Path):
        out = self._handle_for(lang)
        out.write(json.dumps([str(flac_path), str(txt_path)], ensure_ascii=False) + "\n")
        out.flush()

    def _handle_for(self, lang: str):
        if lang not in self.handles:
            output_path = output_path_for(self.jsonl_path, lang)
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            self.output_paths[lang] = output_path
            self.tmp_paths[lang] = tmp_path
            self.handles[lang] = tmp_path.open("w", encoding="utf-8")
        return self.handles[lang]

    def commit(self) -> dict[str, Path]:
        for out in self.handles.values():
            out.close()
        self.handles.clear()
        for lang, tmp_path in self.tmp_paths.items():
            tmp_path.replace(self.output_paths[lang])
        return self.output_paths

    def abort(self):
        for out in self.handles.values():
            out.close()
        self.handles.clear()
        for tmp_path in self.tmp_paths.values():
            if tmp_path.exists():
                tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Detect language for txt files referenced by a [flac, txt] jsonl."
    )
    parser.add_argument("jsonl_pos", nargs="?", type=Path, help="Input jsonl path. Each line should be [flac, txt].")
    parser.add_argument(
        "--jsonl",
        dest="jsonl_opt",
        type=Path,
        default=None,
        help="Input jsonl path. Same as the positional argument.",
    )
    parser.add_argument("--max-chars", type=int, default=4096, help="Read at most this many chars per txt.")
    parser.add_argument(
        "--min-letters",
        type=int,
        default=2,
        help="Return unknown when the sampled text has fewer script letters than this.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N items, useful for testing.")
    args = parser.parse_args()

    jsonl_path = args.jsonl_opt or args.jsonl_pos
    if jsonl_path is None:
        raise SystemExit("jsonl path is required")
    if not jsonl_path.is_file():
        raise SystemExit(f"jsonl is not a file: {jsonl_path}")
    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be > 0")
    if args.min_letters <= 0:
        raise SystemExit("--min-letters must be > 0")

    stats = Counter()
    records = iter_flac_txt_jsonl(jsonl_path)
    if tqdm is not None:
        records = tqdm(records, desc="detect_lang")

    writer = LanguageJsonlWriter(jsonl_path)
    try:
        for idx, (_line_no, flac_path, txt_path) in enumerate(records, start=1):
            if args.limit > 0 and idx > args.limit:
                break
            if not txt_path.exists():
                lang = "missing_txt"
            else:
                text = read_sample_text(txt_path, args.max_chars)
                lang = detect_language(text, args.min_letters)

            writer.write(lang, flac_path, txt_path)
            stats[lang] += 1
    except Exception:
        writer.abort()
        raise

    output_paths = writer.commit()

    print(f"done. input={jsonl_path}", flush=True)
    for lang, count in sorted(stats.items()):
        print(f"{lang}: {count} -> {output_paths[lang]}", flush=True)


if __name__ == "__main__":
    main()
