#!/usr/bin/env python3
# coding: utf-8

import argparse
import traceback
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
from torch.multiprocessing import Process, Queue


def iter_flac_files(root):
    seen = set()
    root = Path(root)
    if not root.is_dir():
        print(f"[WARN] skip missing root dir: {root}", flush=True)
        return
    for flac_path in root.rglob("*.flac"):
        if not flac_path.is_file():
            continue
        resolved = flac_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield flac_path


def load_scp(scp_path):
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Path(line)


def output_path_for(flac_path):
    return flac_path.with_suffix(".whisper.txt")


def transcribe_one(model, flac_path, args):
    kwargs = {
        "beam_size": args.beam_size,
        "vad_filter": args.vad_filter,
    }
    if args.language:
        kwargs["language"] = args.language
    if args.initial_prompt:
        kwargs["initial_prompt"] = args.initial_prompt
    if args.vad_filter:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": args.min_silence_duration_ms}

    segments, info = model.transcribe(str(flac_path), **kwargs)
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    content = "".join(text_parts).strip()
    return content, info


def whisper_worker(num, task_queue, done_queue, args):
    cuda_count = torch.cuda.device_count()
    device_index = num % cuda_count if cuda_count > 0 else 0

    print(f"whisper_worker {num} loading model on cuda:{device_index}", flush=True)
    model = WhisperModel(
        args.model,
        device="cuda",
        device_index=device_index,
        compute_type=args.compute_type,
    )
    print(f"whisper_worker {num} started", flush=True)

    for flac_path in iter(task_queue.get, "STOP"):
        flac_path = Path(flac_path)
        out_path = output_path_for(flac_path)
        if out_path.exists() and not args.overwrite:
            done_queue.put(("skip", str(flac_path), str(out_path), "", 0.0, None))
            continue

        try:
            content, info = transcribe_one(model, flac_path, args)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(out_path)
            done_queue.put(
                (
                    "ok",
                    str(flac_path),
                    str(out_path),
                    info.language,
                    float(info.language_probability),
                    None,
                )
            )
        except Exception:
            done_queue.put(("err", str(flac_path), str(out_path), "", 0.0, traceback.format_exc()))

    done_queue.put(("STOP", "", "", "", 0.0, None))
    print(f"whisper_worker {num} stopped", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Transcribe all .flac files under a segs directory with faster-whisper.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/usr/local/corpus/4th_biz/zh/segs"),
        help="The segs directory to scan.",
    )
    parser.add_argument("--scp", type=Path, default=None, help="Optional .scp file. Only listed flac files are transcribed.")
    parser.add_argument("--model", default="/usr/local/data/faster-whisper-large-v3/")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="", help="Optional language code, for example zh/en/ja. Empty means auto-detect.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--initial-prompt", default="")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--min-silence-duration-ms", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true", help="Re-transcribe even when .whisper.txt already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N files, useful for testing.")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes. Default: one per CUDA device, or 1 on CPU.")
    args = parser.parse_args()

    if args.scp:
        flac_files = sorted(load_scp(args.scp))
    else:
        flac_files = sorted(iter_flac_files(args.root))
    if args.limit > 0:
        flac_files = flac_files[: args.limit]

    if not flac_files:
        print(f"No .flac files found. root={args.root}", flush=True)
        return

    task_files = [
        flac_path for flac_path in flac_files
        if args.overwrite or not output_path_for(flac_path).exists()
    ]
    skipped = len(flac_files) - len(task_files)
    if not task_files:
        print(f"done. files={len(flac_files)} processed=0 skipped={skipped} errors=0", flush=True)
        return

    if args.workers > 0:
        num_workers = args.workers
    else:
        num_workers = max(1, torch.cuda.device_count())

    task_queue = Queue()
    done_queue = Queue()
    workers = []

    for i in range(num_workers):
        p = Process(target=whisper_worker, args=(i, task_queue, done_queue, args))
        p.start()
        workers.append(p)

    for flac_path in task_files:
        task_queue.put(str(flac_path))
    for _ in workers:
        task_queue.put("STOP")

    processed = 0
    errors = 0
    stopped_workers = 0
    with tqdm(total=len(task_files), desc="whisper") as pbar:
        while stopped_workers < len(workers):
            status, flac_path, out_path, language, probability, error_text = done_queue.get()
            if status == "STOP":
                stopped_workers += 1
                continue
            if status == "ok":
                processed += 1
                tqdm.write(f"[OK] {flac_path} -> {out_path} lang={language} prob={probability:.3f}")
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                tqdm.write(f"[ERR] failed: {flac_path}")
                if error_text:
                    tqdm.write(error_text)
            pbar.update(1)

    for p in workers:
        p.join()

    print(f"done. files={len(flac_files)} processed={processed} skipped={skipped} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
