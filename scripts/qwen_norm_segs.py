#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import os
import re
import string
import traceback
import unicodedata
from difflib import SequenceMatcher
from itertools import islice
from pathlib import Path
from typing import Iterable

from torch.multiprocessing import Process, Queue
from tqdm import tqdm

from qwen3_asr_oss import transcribe_file


_NEED_TN_RE = re.compile(
    r"[0-9]"
    r"|&"
    r"|[\u2070-\u209F\u20A0-\u20CF\u0024\u00A2-\u00A5\u0E3F\u17DB\u2103\u2109\u00B0]"
)


def scandir_generator(path: Path) -> Iterable[Path]:
    with os.scandir(path) as it:
        for entry in it:
            entry_path = Path(entry.path)
            if entry.is_file():
                yield entry_path
            elif entry.is_dir():
                yield from scandir_generator(entry_path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def need_text_normalization(text: str) -> bool:
    return bool(_NEED_TN_RE.search(text))


def txt_path_for(flac_path: Path) -> Path:
    return flac_path.with_suffix(".txt")


def whisper_path_for(flac_path: Path) -> Path:
    return flac_path.with_suffix(".whisper.txt")


def output_path_for(flac_path: Path) -> Path:
    return flac_path.with_suffix(".qwen.txt")


def iter_flac_files(root: Path):
    root = Path(root)
    if not root.is_dir():
        print(f"[WARN] skip missing root dir: {root}", flush=True)
        return
    for flac_path in scandir_generator(root):
        if flac_path.suffix == ".flac":
            yield flac_path


def load_scp(scp_path: Path):
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Path(line)


def collect_files(paths, limit: int):
    if limit > 0:
        return list(islice(paths, limit))
    return list(paths)


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


def should_run_qwen(flac_path: Path, threshold: float, allow_missing_whisper: bool) -> tuple[bool, str, float]:
    txt_path = txt_path_for(flac_path)
    whisper_path = whisper_path_for(flac_path)
    if not txt_path.exists():
        return False, f"missing txt: {txt_path}", 0.0

    text = read_text(txt_path)
    if not text:
        return False, "empty txt", 0.0
    if not need_text_normalization(text):
        return False, "no normalization needed", 0.0
    if not whisper_path.exists():
        if allow_missing_whisper:
            return True, "missing whisper, tn only", 1.0
        return False, f"missing whisper: {whisper_path}", 0.0

    whisper_text = read_text(whisper_path)
    score = similarity(text, whisper_text)
    if score < threshold:
        return False, f"similarity {score:.3f} < {threshold:.3f}", score
    return True, "", score


def process_one(flac_path: Path, threshold: float, overwrite: bool, allow_missing_whisper: bool):
    flac_path = Path(flac_path)
    out_path = output_path_for(flac_path)
    if out_path.exists() and not overwrite:
        return "skip", flac_path, out_path, 1.0, "exists", None

    ok, reason, score = should_run_qwen(flac_path, threshold, allow_missing_whisper)
    if not ok:
        return "skip", flac_path, out_path, score, reason, None

    text = transcribe_file(flac_path, verbose=False).strip()
    if not text:
        return "err", flac_path, out_path, score, "empty qwen result", None

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(out_path)
    return "ok", flac_path, out_path, score, "", None


def worker(
    num: int,
    task_queue: Queue,
    done_queue: Queue,
    threshold: float,
    overwrite: bool,
    allow_missing_whisper: bool,
):
    print(f"qwen_worker {num} started", flush=True)
    for flac_path in iter(task_queue.get, "STOP"):
        try:
            done_queue.put(process_one(Path(flac_path), threshold, overwrite, allow_missing_whisper))
        except Exception:
            done_queue.put(("err", Path(flac_path), output_path_for(Path(flac_path)), 0.0, "", traceback.format_exc()))
    done_queue.put(("STOP", None, None, 0.0, "", None))
    print(f"qwen_worker {num} stopped", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run qwen3-asr-flash-filetrans on segment flacs that need text normalization."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/usr/local/corpus/4th_biz/zh/segs"),
        help="The segs directory to scan.",
    )
    parser.add_argument("--scp", type=Path, default=None, help="Optional .scp file of .flac files.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum txt/whisper similarity.")
    parser.add_argument(
        "--allow-missing-whisper",
        action="store_true",
        help="If .whisper.txt is missing, run Qwen when .txt needs text normalization.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run even when .qwen.txt already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N files, useful for testing.")
    args = parser.parse_args()

    if args.scp:
        flac_files = collect_files(load_scp(args.scp), args.limit)
    else:
        flac_files = collect_files(iter_flac_files(args.root), args.limit)

    if not flac_files:
        print(f"No .flac files found. root={args.root}", flush=True)
        return

    task_files = [
        flac_path for flac_path in flac_files
        if args.overwrite or not output_path_for(flac_path).exists()
    ]
    already_done = len(flac_files) - len(task_files)
    if not task_files:
        print(f"done. files={len(flac_files)} processed=0 skipped={already_done} errors=0", flush=True)
        return

    task_queue = Queue()
    done_queue = Queue()
    workers = []
    for i in range(args.workers):
        p = Process(
            target=worker,
            args=(i, task_queue, done_queue, args.threshold, args.overwrite, args.allow_missing_whisper),
        )
        p.start()
        workers.append(p)

    for flac_path in task_files:
        task_queue.put(str(flac_path))
    for _ in workers:
        task_queue.put("STOP")

    processed = 0
    skipped = already_done
    errors = 0
    stopped_workers = 0
    with tqdm(total=len(task_files), desc="qwen_norm") as pbar:
        while stopped_workers < len(workers):
            status, flac_path, out_path, score, reason, error_text = done_queue.get()
            if status == "STOP":
                stopped_workers += 1
                continue

            if status == "ok":
                processed += 1
                tqdm.write(f"[OK] {flac_path} -> {out_path} sim={score:.3f}")
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                tqdm.write(f"[ERR] failed: {flac_path} sim={score:.3f} {reason}")
                if error_text:
                    tqdm.write(error_text)
            pbar.update(1)

    for p in workers:
        p.join()

    print(
        f"done. files={len(flac_files)} processed={processed} skipped={skipped} errors={errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
