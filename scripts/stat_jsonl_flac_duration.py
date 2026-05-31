#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import soundfile as sf
from torch.multiprocessing import Process, Queue
from tqdm import tqdm


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def flac_duration_seconds(flac_path: Path) -> float:
    info = sf.info(str(flac_path))
    return float(info.frames) / float(info.samplerate)


def iter_flac_paths(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json at {jsonl_path}:{line_no}: {exc}") from exc
            if not isinstance(item, list) or not item:
                raise ValueError(f"bad item at {jsonl_path}:{line_no}: expected non-empty list")
            yield Path(item[0])


def stat_output_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".stat.csv")


def progress_output_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".duration.jsonl")


def load_done(progress_path: Path):
    done = {}
    if not progress_path.exists():
        return done

    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            flac_path = item.get("flac")
            if isinstance(flac_path, str):
                done[flac_path] = item
    return done


def duration_worker(num: int, task_queue: Queue, done_queue: Queue):
    print(f"duration_worker {num} started", flush=True)
    for flac_path_text in iter(task_queue.get, "STOP"):
        flac_path = Path(flac_path_text)
        if not flac_path.exists():
            done_queue.put({"flac": str(flac_path), "status": "missing", "duration": 0.0})
            continue
        try:
            done_queue.put({
                "flac": str(flac_path),
                "status": "ok",
                "duration": flac_duration_seconds(flac_path),
            })
        except Exception as exc:
            done_queue.put({
                "flac": str(flac_path),
                "status": "error",
                "duration": 0.0,
                "error": str(exc),
            })
    done_queue.put({"status": "STOP"})
    print(f"duration_worker {num} stopped", flush=True)


def summarize(items):
    durations = [float(item.get("duration", 0.0)) for item in items if item.get("status") == "ok"]
    missing = sum(1 for item in items if item.get("status") == "missing")
    errors = sum(1 for item in items if item.get("status") == "error")

    count = len(durations)
    total = sum(durations)
    avg = total / count if count else 0.0
    max_duration = max(durations) if durations else 0.0
    min_duration = min(durations) if durations else 0.0
    return count, total, avg, max_duration, min_duration, missing, errors


def write_stat_csv(out_path: Path, jsonl_items: int, summary):
    count, total, avg, max_duration, min_duration, missing, errors = summary
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "jsonl_items",
            "flac_total",
            "duration_total",
            "duration_avg",
            "duration_max",
            "duration_min",
            "missing",
            "errors",
        ])
        writer.writerow([
            jsonl_items,
            count,
            format_duration(total),
            format_duration(avg),
            format_duration(max_duration),
            format_duration(min_duration),
            missing,
            errors,
        ])


def main():
    parser = argparse.ArgumentParser(description="Stat total/count/avg/max/min duration of flac files in a jsonl.")
    parser.add_argument("jsonl", type=Path, help="JSONL file whose first list element is a flac path.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing duration progress and recalculate.")
    args = parser.parse_args()

    jsonl_path = args.jsonl.expanduser().resolve()
    flac_paths = list(iter_flac_paths(jsonl_path))
    progress_path = progress_output_path(jsonl_path)

    if args.overwrite and progress_path.exists():
        progress_path.unlink()

    done = load_done(progress_path)
    task_paths = [flac_path for flac_path in flac_paths if str(flac_path) not in done]

    if task_paths:
        task_queue = Queue()
        done_queue = Queue()
        workers = []
        num_workers = max(1, args.workers)

        for i in range(num_workers):
            p = Process(target=duration_worker, args=(i, task_queue, done_queue))
            p.start()
            workers.append(p)

        for flac_path in task_paths:
            task_queue.put(str(flac_path))
        for _ in workers:
            task_queue.put("STOP")

        stopped_workers = 0
        with progress_path.open("a", encoding="utf-8") as progress_file:
            with tqdm(total=len(task_paths), desc="duration") as pbar:
                while stopped_workers < len(workers):
                    item = done_queue.get()
                    if item.get("status") == "STOP":
                        stopped_workers += 1
                        continue
                    progress_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    progress_file.flush()
                    done[item["flac"]] = item
                    pbar.update(1)

        for p in workers:
            p.join()

    out_path = stat_output_path(jsonl_path)
    summary = summarize(done.values())
    write_stat_csv(out_path, len(flac_paths), summary)
    count, _total, _avg, _max_duration, _min_duration, missing, errors = summary
    skipped = len(flac_paths) - len(task_paths)
    print(
        f"done. jsonl_items={len(flac_paths)} flac_total={count} skipped={skipped} "
        f"missing={missing} errors={errors} output={out_path} progress={progress_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
