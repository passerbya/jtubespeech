#!/usr/bin/python
# coding: utf-8

import torch
import argparse
import hashlib
import traceback
from contextlib import ExitStack
from time import time
from pathlib import Path
from torch.multiprocessing import Process, Queue
from demucs.api import Separator, save_audio


def bucket_for_filename(path: Path, buckets: int) -> int:
    if buckets == 1:
        return 0
    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % buckets


def write_bucket_manifests(src: Path, buckets: int, output_dir: Path) -> None:
    input_dir = src / "wav_org"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = [output_dir / f"bucket_{index}.txt" for index in range(buckets)]
    counts = [0] * buckets

    with ExitStack() as stack:
        manifests = [
            stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
            for path in manifest_paths
        ]
        for directory in input_dir.iterdir():
            if not directory.is_dir():
                continue
            for wav_src in directory.iterdir():
                if wav_src.suffix.lower() not in {".flac", ".wav"}:
                    continue
                bucket = bucket_for_filename(wav_src, buckets)
                relative_path = wav_src.relative_to(input_dir).as_posix()
                manifests[bucket].write(f"{relative_path}\n")
                counts[bucket] += 1

    for index, manifest_path in enumerate(manifest_paths):
        print(
            f"[smoke] bucket={index}/{buckets} files={counts[index]} "
            f"manifest={manifest_path}",
            flush=True,
        )
    print(f"[smoke] done, total_files={sum(counts)}", flush=True)


def separate_worker(_src, cuda_num, task_queue):
    device_id = cuda_num%torch.cuda.device_count()
    print(f"[worker {cuda_num}] started on cuda:{device_id}", flush=True)
    model_path = Path("/usr/local/corpus/penghu/work/voice_song_separation/demucs/outputs/xps/76024946_2stem")
    print(f"[worker {cuda_num}] loading model from {model_path}", flush=True)
    separator = Separator(
        model='best_singing_in_vocal',
        repo=model_path,
        device=f"cuda:{device_id}",
        shifts=1,
        split=True,
        overlap=0.25,
        progress=True,
        jobs=2,
        segment=23.76562358276644
    )
    print(
        f"[worker {cuda_num}] model loaded, samplerate={separator.samplerate}",
        flush=True,
    )
    kwargs = {
        "samplerate": separator.samplerate,
        "bitrate": 320,
        "preset": 2,
        "clip": 'rescale',
        "as_float": False,
        "bits_per_sample": 16,
    }
    processed = 0
    skipped = 0
    for wav_dest, wav_src in iter(task_queue.get, "STOP"):
        if wav_dest.exists():
            skipped += 1
            print(f"[worker {cuda_num}] skip existing: {wav_dest}", flush=True)
            continue
        start = time()
        print(f"[worker {cuda_num}] start: {wav_src} -> {wav_dest}", flush=True)
        wav_dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            origin, res = separator.separate_audio_file(str(wav_src))
            source = res.pop('vocals')
            save_audio(source, str(wav_dest), **kwargs)
        except Exception:
            print(f"[worker {cuda_num}] failed: {wav_src}", flush=True)
            traceback.print_exc()
            raise
        processed += 1
        elapsed = time() - start
        print(
            f"[worker {cuda_num}] done: {wav_dest} "
            f"elapsed={elapsed:.2f}s processed={processed}",
            flush=True,
        )

    print(
        f"[worker {cuda_num}] stopped, processed={processed}, skipped={skipped}",
        flush=True,
    )

def main(config):
    src = config.path
    print(f"[main] src={src}", flush=True)
    print(f"[main] bucket={config.bucket}/{config.buckets}", flush=True)
    print(f"[main] cuda_count={torch.cuda.device_count()}", flush=True)
    print(f"[main] number_of_processes={NUMBER_OF_PROCESSES}", flush=True)
    print(f"[main] input_dir={src / 'wav_org'}", flush=True)
    print(f"[main] output_dir={src / 'flac'}", flush=True)
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = Process(
            target=separate_worker,
            args=(src, i, task_queue),
        )
        p.start()
        processes.append(p)
        print(f"[main] started worker {i}, pid={p.pid}", flush=True)

    scanned_dirs = 0
    scanned_files = 0
    queued = 0
    skipped_existing = 0
    skipped_suffix = 0
    skipped_bucket = 0
    for _dir in (src / 'wav_org').iterdir():
        if not _dir.is_dir():
            continue
        scanned_dirs += 1
        print(f"[main] scanning dir: {_dir}", flush=True)
        for wav_src in _dir.iterdir():
            scanned_files += 1
            if wav_src.suffix.lower() not in {".flac", ".wav"}:
                skipped_suffix += 1
                continue
            if bucket_for_filename(wav_src, config.buckets) != config.bucket:
                skipped_bucket += 1
                continue
            wav_dest = src / "flac" / wav_src.relative_to(src / "wav_org")
            wav_dest = wav_dest.with_suffix('.flac')
            if wav_dest.exists():
                skipped_existing += 1
                continue
            if not wav_dest.parent.exists():
                wav_dest.parent.mkdir(parents=True)
            print(f"[main] queue: {wav_src} -> {wav_dest}", flush=True)
            task_queue.put((wav_dest, wav_src))
            queued += 1

    print(
        f"[main] scan done, dirs={scanned_dirs}, files={scanned_files}, "
        f"queued={queued}, skipped_existing={skipped_existing}, "
        f"skipped_suffix={skipped_suffix}, skipped_bucket={skipped_bucket}",
        flush=True,
    )

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        print(f"[main] send STOP {i + 1}/{NUMBER_OF_PROCESSES}", flush=True)
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()
        print(f"[main] worker pid={p.pid} exitcode={p.exitcode}", flush=True)

    print("[main] separate done.", flush=True)

NUMBER_OF_PROCESSES = torch.cuda.device_count()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/corpus/4th_biz/zh"))
    parser.add_argument("--buckets", type=int, default=1, help="Total number of buckets. 1 disables bucketing.")
    parser.add_argument("--bucket", type=int, default=0, help="Zero-based bucket index for this server.")
    parser.add_argument(
        "--bucket-smoke-test",
        type=Path,
        default=None,
        metavar="OUTPUT_DIR",
        help="Write one input manifest per bucket and exit without starting workers.",
    )
    args = parser.parse_args()
    if args.buckets < 1:
        parser.error("--buckets must be at least 1")
    if not 0 <= args.bucket < args.buckets:
        parser.error("--bucket must be in the range [0, --buckets)")
    if args.bucket_smoke_test is not None:
        write_bucket_manifests(args.path, args.buckets, args.bucket_smoke_test)
        raise SystemExit(0)
    main(args)
