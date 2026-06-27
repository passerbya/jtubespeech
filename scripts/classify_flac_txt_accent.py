#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from torch.multiprocessing import Process, Queue
from tqdm import tqdm


STOP = "__STOP__"
SAMPLE_RATE = 16000
PYMODULE_FILE = "custom_interface.py"
CLASSNAME = "CustomEncoderWav2vec2Classifier"
_LABEL_SAFE_RE = re.compile(r"[^0-9a-zA-Z_.-]+")


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def read_jsonl_items(jsonl_path: Path):
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
            yield {"item": item, "flac": str(item[0])}


def unique_flacs(items):
    seen = set()
    for item in items:
        flac = item["flac"]
        if flac in seen:
            continue
        seen.add(flac)
        yield flac

def label_jsonl_path(jsonl_path: Path, label: str) -> Path:
    return jsonl_path.with_suffix(f".{label}.jsonl")

def label_stat_path(jsonl_path: Path, label: str) -> Path:
    return jsonl_path.with_suffix(f".stat.{label}.csv")

def default_progress_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".accent.progress.jsonl")

def default_duration_progress_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".duration.jsonl")


def load_classifier(source: str, device: str):
    try:
        from speechbrain.pretrained.interfaces import foreign_class
    except ImportError:
        from speechbrain.inference.interfaces import foreign_class

    return foreign_class(
        source=source,
        pymodule_file=PYMODULE_FILE,
        classname=CLASSNAME,
        run_opts={"device": device},
    )


def scalar_float(value) -> float:
    while isinstance(value, (list, tuple)):
        value = value[0] if value else 0.0
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def scalar_text(value) -> str:
    while isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def normalize_model_label(label: str) -> str:
    return label.strip().lower()


def safe_label_suffix(label: str) -> str:
    label = _LABEL_SAFE_RE.sub("_", normalize_model_label(label)).strip("._-")
    return label or "o"


def output_label(result: dict, threshold: float) -> str:
    if result.get("status") != "ok":
        return "o"

    pred_label = normalize_model_label(str(result.get("pred_label", "")))
    score = float(result.get("score", 0.0))
    if score > threshold and pred_label:
        return safe_label_suffix(pred_label)
    return "o"


def classify_one(classifier, flac: str, duration: float | None) -> dict:
    flac_path = Path(flac)
    if not flac_path.exists():
        return {
            "flac": flac,
            "status": "missing",
            "pred_label": "",
            "score": 0.0,
            "index": "",
            "error": "",
        }

    try:
        import librosa

        sig, _ = librosa.load(str(flac_path), sr=SAMPLE_RATE, duration=duration)
        sig = torch.tensor(sig)
        with torch.no_grad():
            _out_prob, score, index, text_lab = classifier.classify_batch(sig)
        return {
            "flac": flac,
            "status": "ok",
            "pred_label": normalize_model_label(scalar_text(text_lab)),
            "score": scalar_float(score),
            "index": scalar_text(index),
            "error": "",
        }
    except Exception as exc:
        return {
            "flac": flac,
            "status": "error",
            "pred_label": "",
            "score": 0.0,
            "index": "",
            "error": str(exc),
        }


def classify_worker(
    worker_id: int,
    task_queue,
    done_queue,
    model_source: str,
    duration: float | None,
):
    cuda_count = torch.cuda.device_count()
    device = f"cuda:{worker_id % cuda_count}" if cuda_count > 0 else "cpu"

    print(f"accent_worker {worker_id} started device={device}", flush=True)
    try:
        classifier = load_classifier(
            source=model_source,
            device=device,
        )
    except Exception as exc:
        done_queue.put({
            "event": "worker_error",
            "worker": worker_id,
            "device": device,
            "error": str(exc),
        })
        done_queue.put({"event": "worker_stop", "worker": worker_id, "device": device})
        print(f"accent_worker {worker_id} failed to load model: {exc}", flush=True)
        return

    for flac in iter(task_queue.get, STOP):
        result = classify_one(classifier, flac, duration=duration)
        result["worker"] = worker_id
        result["device"] = device
        done_queue.put({"event": "result", "result": result})

    done_queue.put({"event": "worker_stop", "worker": worker_id, "device": device})
    print(f"accent_worker {worker_id} stopped", flush=True)


def load_jsonl_by_flac(path: Path):
    done = {}
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            flac = item.get("flac")
            if isinstance(flac, str):
                done[flac] = item
    return done


def classify_pending(
    task_flacs,
    results: dict,
    progress_path: Path,
    workers_count: int,
    model_source: str,
    duration: float | None,
):
    if not task_flacs:
        return []

    task_queue = Queue()
    done_queue = Queue()
    workers = []
    worker_errors = []
    num_workers = workers_count if workers_count > 0 else max(1, torch.cuda.device_count())

    for i in range(num_workers):
        proc = Process(
            target=classify_worker,
            args=(
                i,
                task_queue,
                done_queue,
                model_source,
                duration,
            ),
        )
        proc.start()
        workers.append(proc)

    for flac in task_flacs:
        task_queue.put(flac)
    for _ in workers:
        task_queue.put(STOP)

    stopped_workers = 0
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with progress_path.open("a", encoding="utf-8") as progress_file:
            with tqdm(total=len(task_flacs), desc="accent") as pbar:
                while stopped_workers < len(workers):
                    message = done_queue.get()
                    event = message.get("event")

                    if event == "worker_stop":
                        stopped_workers += 1
                        continue

                    if event == "worker_error":
                        worker_errors.append(message)
                        tqdm.write(
                            "worker_error "
                            f"worker={message.get('worker')} device={message.get('device')} "
                            f"error={message.get('error')}"
                        )
                        continue

                    if event != "result":
                        continue

                    result = message["result"]
                    results[result["flac"]] = result
                    progress_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    progress_file.flush()
                    pbar.update(1)
    except KeyboardInterrupt:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join()
        raise

    for proc in workers:
        proc.join()

    return worker_errors


def summarize_duration(label_items, results: dict, durations: dict):
    unique = list(unique_flacs(label_items))
    duration_values = []
    duration_missing_cache = 0
    duration_missing = 0
    duration_errors = 0
    accent_missing = 0
    accent_errors = 0

    for flac in unique:
        result = results.get(flac, {})
        if result.get("status") == "missing":
            accent_missing += 1
        elif result.get("status") == "error":
            accent_errors += 1

        duration_item = durations.get(flac)
        if duration_item is None:
            duration_missing_cache += 1
            continue

        status = duration_item.get("status")
        if status == "ok":
            duration_values.append(float(duration_item.get("duration", 0.0)))
        elif status == "missing":
            duration_missing += 1
        else:
            duration_errors += 1

    count = len(duration_values)
    total = sum(duration_values)
    avg = total / count if count else 0.0
    max_duration = max(duration_values) if duration_values else 0.0
    min_duration = min(duration_values) if duration_values else 0.0
    return {
        "jsonl_items": len(label_items),
        "unique_flac": len(unique),
        "flac_total": count,
        "duration_total": total,
        "duration_avg": avg,
        "duration_max": max_duration,
        "duration_min": min_duration,
        "duration_missing_cache": duration_missing_cache,
        "duration_missing": duration_missing,
        "duration_errors": duration_errors,
        "accent_missing": accent_missing,
        "accent_errors": accent_errors,
    }


def write_stat_csv(out_path: Path, summary: dict):
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "jsonl_items",
            "unique_flac",
            "flac_total",
            "duration_total",
            "duration_avg",
            "duration_max",
            "duration_min",
            "duration_missing_cache",
            "duration_missing",
            "duration_errors",
            "accent_missing",
            "accent_errors",
        ])
        writer.writerow([
            summary["jsonl_items"],
            summary["unique_flac"],
            summary["flac_total"],
            format_duration(summary["duration_total"]),
            format_duration(summary["duration_avg"]),
            format_duration(summary["duration_max"]),
            format_duration(summary["duration_min"]),
            summary["duration_missing_cache"],
            summary["duration_missing"],
            summary["duration_errors"],
            summary["accent_missing"],
            summary["accent_errors"],
        ])
    tmp_path.replace(out_path)


def write_outputs(base_jsonl_path: Path, items, results: dict, durations: dict, threshold: float):
    label_items = {}
    jsonl_files = {}
    counts = {}

    for item in items:
        result = results[item["flac"]]
        label = output_label(result, threshold)
        label_items.setdefault(label, []).append(item)
        counts[label] = counts.get(label, 0) + 1

    try:
        for label in sorted(label_items):
            jsonl_path = label_jsonl_path(base_jsonl_path, label)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
            jsonl_files[label] = (jsonl_path, tmp_path, tmp_path.open("w", encoding="utf-8"))

        for label, items_for_label in label_items.items():
            for item in items_for_label:
                jsonl_files[label][2].write(json.dumps(item["item"], ensure_ascii=False) + "\n")
                jsonl_files[label][2].flush()
    finally:
        for _label, (_path, _tmp_path, handle) in jsonl_files.items():
            handle.close()

    for _label, (path, tmp_path, _handle) in jsonl_files.items():
        tmp_path.replace(path)

    summaries = {}
    for label, items_for_label in sorted(label_items.items()):
        summary = summarize_duration(items_for_label, results, durations)
        write_stat_csv(label_stat_path(base_jsonl_path, label), summary)
        summaries[label] = summary

    return counts, summaries


def main():
    parser = argparse.ArgumentParser(
        description="Classify [flac, txt] jsonl into accent-label jsonl files."
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="Input flac_txt.jsonl whose items are [flac, txt].")
    parser.add_argument("--threshold", type=float, default=0.85, help="Use model label only when score is greater than this.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to load for accent model. Use 0 for full audio.")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes. Default: one per CUDA device, or 1 on CPU.")
    parser.add_argument(
        "--model-source",
        default="/usr/local/corpus/accent/accent-id-commonaccent_xlsr-en-english",
        help="SpeechBrain foreign_class source path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N jsonl items, useful for testing.")
    parser.add_argument("--overwrite-progress", action="store_true", help="Ignore existing accent progress and reclassify.")
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if not jsonl_path.is_file():
        raise SystemExit(f"jsonl is not a file: {jsonl_path}")

    progress_path = default_progress_path(jsonl_path)
    duration_progress_path = default_duration_progress_path(jsonl_path)
    if not duration_progress_path.is_file():
        raise SystemExit(
            f"duration progress is not a file: {duration_progress_path}. "
            "Run stat_jsonl_flac_duration.py first."
        )

    duration = None if args.duration <= 0 else args.duration

    items = list(read_jsonl_items(jsonl_path))
    if args.limit > 0:
        items = items[: args.limit]

    durations = load_jsonl_by_flac(duration_progress_path)

    if args.overwrite_progress and progress_path.exists():
        progress_path.unlink()

    results = load_jsonl_by_flac(progress_path)
    flacs = list(unique_flacs(items))
    task_flacs = [flac for flac in flacs if flac not in results]

    worker_errors = classify_pending(
        task_flacs=task_flacs,
        results=results,
        progress_path=progress_path,
        workers_count=args.workers,
        model_source=args.model_source,
        duration=duration,
    )

    missing_results = [flac for flac in flacs if flac not in results]
    if missing_results:
        raise SystemExit(
            f"missing classification results: {len(missing_results)}. "
            f"worker_errors={len(worker_errors)} progress={progress_path}"
        )

    counts, summaries = write_outputs(jsonl_path, items, results, durations, threshold=args.threshold)
    print(
        "done. "
        f"items={len(items)} unique_flac={len(flacs)} classified={len(task_flacs)} "
        f"labels={','.join(f'{label}:{count}' for label, count in sorted(counts.items()))} "
        f"progress={progress_path} duration_progress={duration_progress_path}",
        flush=True,
    )
    for label in sorted(summaries):
        summary = summaries[label]
        print(
            f"{label}: {label_jsonl_path(jsonl_path, label)} {label_stat_path(jsonl_path, label)} "
            f"duration={format_duration(summary['duration_total'])} flac_total={summary['flac_total']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
