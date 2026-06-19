#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
from pathlib import Path

import soundfile as sf
from tqdm import tqdm
from torch.multiprocessing import Process, Queue


def scandir_generator(path: Path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(Path(entry.path))


def find_dataset_roots(root):
    root = Path(root)
    if (root / "txt").is_dir() and (root / "flac").is_dir():
        yield root
        return

    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        if current.name in {"txt", "flac", "segments"}:
            dirnames[:] = []
            continue
        if (current / "txt").is_dir() and (current / "flac").is_dir():
            yield current
            dirnames[:] = []


def iter_txt_files(dataset_root):
    txt_root = dataset_root / "txt"
    for txt_path in scandir_generator(txt_root):
        if txt_path.suffix == ".txt":
            yield txt_path


def parse_segments(txt_path, min_duration):
    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            columns = line.split("\t")
            if len(columns) < 2:
                print(f"[WARN] bad subtitle columns: {txt_path}:{line_no}: {line}")
                continue
            try:
                start = float(columns[0])
                end = float(columns[1])
            except ValueError:
                print(f"[WARN] bad subtitle time: {txt_path}:{line_no}: {line}")
                continue
            if end <= start:
                print(f"[WARN] bad time range: {txt_path}:{line_no}: {line}")
                continue
            if end - start < min_duration:
                continue
            text = "\t".join(columns[2:]).strip() if len(columns) > 2 else ""
            if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                text = text[1:-1]
            elif text.startswith('"'):
                text = text[1:]
            segments.append((start, end, text))
    return segments


def remove_bad_flac(flac_path):
    try:
        flac_path.unlink()
        return " deleted source flac"
    except FileNotFoundError:
        return " source flac already missing"
    except Exception as exc:
        return f" failed to delete source flac: {exc}"


def segment_one(args_tuple):
    txt_path, dataset_root, min_duration = args_tuple
    txt_path = Path(txt_path)
    dataset_root = Path(dataset_root)
    txt_root = dataset_root / "txt"
    flac_root = dataset_root / "flac"
    segments_root = dataset_root / "segs"

    rel_txt = txt_path.relative_to(txt_root)
    flac_path = flac_root / rel_txt.with_suffix(".flac")
    if not flac_path.exists():
        return 0, 1, f"[MISS] no matching flac: {flac_path}"

    ranges = parse_segments(txt_path, min_duration)
    if not ranges:
        return 0, 0, None

    try:
        info = sf.info(str(flac_path))
    except Exception as exc:
        return 0, 1, f"[ERR] cannot read flac info: {flac_path}: {exc}"

    sr = info.samplerate
    total_frames = info.frames
    out_dir = segments_root / rel_txt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = txt_path.stem

    written = 0
    skipped = 0
    for idx, (start_sec, end_sec, text) in enumerate(ranges):
        start_frame = max(0, int(round(start_sec * sr)))
        end_frame = min(total_frames, int(round(end_sec * sr)))
        if end_frame <= start_frame:
            skipped += 1
            continue

        out_flac = out_dir / f"{out_stem}_{idx:04d}.flac"
        out_txt = out_dir / f"{out_stem}_{idx:04d}.txt"
        if out_flac.exists() and out_txt.exists():
            skipped += 1
            continue

        try:
            if not out_flac.exists():
                audio, _ = sf.read(
                    str(flac_path),
                    start=start_frame,
                    stop=end_frame,
                    dtype="float32",
                    always_2d=False,
                )
                sf.write(str(out_flac), audio, sr, format="FLAC")
            if not out_txt.exists():
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(text)
            written += 1
        except Exception as exc:
            delete_status = remove_bad_flac(flac_path)
            return written, skipped + 1, f"[ERR] failed segment: {flac_path} -> {out_flac}: {exc};{delete_status}"

    return written, skipped, None


def build_tasks(root, min_duration):
    dataset_roots = list(find_dataset_roots(root))
    for dataset_root in dataset_roots:
        for txt_path in iter_txt_files(dataset_root):
            yield (txt_path, dataset_root, min_duration)


def segment_worker(num, task_queue, done_queue):
    print(f"segment_worker {num} started")
    for task in iter(task_queue.get, "STOP"):
        done_queue.put(segment_one(task))
    done_queue.put("STOP")
    print(f"segment_worker {num} stopped")


def main():
    parser = argparse.ArgumentParser(description="Cut flac files into subtitle-aligned segments.")
    parser.add_argument(
        "--root",
        default="/usr/local/corpus/4th_biz/zh",
        help="Root directory containing txt/flac dataset directories.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--min-duration", type=float, default=0.05, help="Skip subtitle segments shorter than this many seconds.")
    args = parser.parse_args()

    tasks = list(build_tasks(args.root, args.min_duration))
    if not tasks:
        print(f"No txt/flac datasets found under {args.root}")
        return

    total_written = 0
    total_skipped = 0
    total_errors = 0
    stopped_workers = 0
    task_queue = Queue()
    done_queue = Queue()
    workers = []

    for i in range(args.workers):
        p = Process(target=segment_worker, args=(i, task_queue, done_queue))
        p.start()
        workers.append(p)

    for task in tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put("STOP")

    with tqdm(total=len(tasks), desc="segment") as pbar:
        while stopped_workers < len(workers):
            item = done_queue.get()
            if item == "STOP":
                stopped_workers += 1
                continue

            written, skipped, message = item
            total_written += written
            total_skipped += skipped
            if message:
                total_errors += 1
                print(message)
            pbar.update(1)

    for p in workers:
        p.join()

    print(f"done. files={len(tasks)} written={total_written} skipped={total_skipped} errors={total_errors}")


if __name__ == "__main__":
    main()
