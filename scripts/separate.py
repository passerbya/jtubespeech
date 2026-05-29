#!/usr/bin/python
# coding: utf-8

import torch
import argparse
import traceback
from time import time
from pathlib import Path
from torch.multiprocessing import Process, Queue
from demucs.api import Separator, save_audio

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

def main():
    print(f"[main] src={src}", flush=True)
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
        f"skipped_suffix={skipped_suffix}",
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
    args = parser.parse_args()
    src = args.path
    main()

