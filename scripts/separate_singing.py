#!/usr/bin/python
# coding: utf-8

import re
import os
import sys
import json
import shutil
import torch
import argparse
from pathlib import Path
from torch.multiprocessing import Process, Queue

def extract_tags(text):
    """从文本中提取形如<|tag|>的标签"""
    return re.findall(r'<\|(.*?)\|>', text)

def scandir_generator(path):
    """仅列出目录中的文件"""
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(entry.path)

def separate_worker(_src, cuda_num, task_queue):
    device_id = cuda_num%torch.cuda.device_count()
    print(f"separate_worker {cuda_num} started", device_id)
    torch.cuda.set_device(device_id)
    from demucs.api import Separator, save_audio
    from demucs.audio import convert_audio_channels
    separator = Separator(model="htdemucs_ft",
                          device=f"cuda:{device_id}",
                          shifts=4,
                          split=True,
                          overlap=0.25,
                          progress=True,
                          jobs=2,
                          segment=7.8)

    for flac_dest, flac_src in iter(task_queue.get, "STOP"):
        print(flac_dest, flac_src)
        origin, res = separator.separate_audio_file(str(flac_src))
        source = res.pop('vocals')
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": 320,
            "preset": 2,
            "clip": 'rescale',
            "as_float": False,
            "bits_per_sample": 16,
        }
        source = convert_audio_channels(source, 1)
        save_audio(source, str(flac_dest), **kwargs)
    print(cuda_num, 'done')

def main():
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

    singing_list = set()
    with open(str(src/'singing.list'), 'r', encoding='utf-8') as f:
        for line in f:
            singing_list.add(line.strip())

    for sub_dir in (src/'audio').iterdir():
        for mp3_path in scandir_generator(sub_dir):
            if mp3_path.suffix != '.mp3' or str(mp3_path) not in singing_list:
                continue
            new_path = src / 'audio_sep' / mp3_path.relative_to(src / 'audio')
            new_path = new_path.parent / f'{new_path.stem}.flac'
            if new_path.exists():
                continue

            new_path.parent.mkdir(parents=True, exist_ok=True)
            task_queue.put((new_path, mp3_path))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()

    print("separate done.")


NUMBER_OF_PROCESSES = 2*torch.cuda.device_count()
if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/data1/audio/music-singing/mtg-jamendo-dataset"))
    args = parser.parse_args()
    src = args.path
    main()

