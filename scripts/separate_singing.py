#!/usr/bin/python
# coding: utf-8

import os
import sys
import torch
import argparse
import soundfile as sf
from pathlib import Path
from demucs.api import Separator, save_audio
from torch.multiprocessing import Process, Queue

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
    model_path = Path("/usr/local/corpus/penghu/work/voice_song_separation/demucs/outputs/xps/76024946_2stem")
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
    kwargs = {
        "samplerate": separator.samplerate,
        "bitrate": 320,
        "preset": 2,
        "clip": 'rescale',
        "as_float": False,
        "bits_per_sample": 16,
    }

    for flac_dest, flac_src in iter(task_queue.get, "STOP"):
        print(flac_dest, flac_src)
        origin, res = separator.separate_audio_file(flac_src.as_posix())
        source = res.pop('vocals')
        save_audio(source, str(flac_dest), **kwargs)
    print(cuda_num, 'done')

def main():
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = Process(
            target=separate_worker,
            args=(input_path, i, task_queue),
        )
        p.start()
        processes.append(p)

    '''
    singing_list = set()
    with open(str(input_path/'singing.list'), 'r', encoding='utf-8') as f:
        for line in f:
            singing_list.add(line.strip())
    '''

    for sub_dir in input_path.iterdir():
        for audio_path in scandir_generator(sub_dir):
            #if audio_path.suffix != '.flac' or str(audio_path) not in singing_list:
            if audio_path.suffix != '.flac':
                continue
            new_path = output_path / audio_path.relative_to(input_path)
            new_path = new_path.parent / f'{new_path.stem}.flac'
            if new_path.exists():
                continue

            new_path.parent.mkdir(parents=True, exist_ok=True)
            task_queue.put((new_path, audio_path))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()

    print("separate done.")


NUMBER_OF_PROCESSES = torch.cuda.device_count()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/data/VocalExtractor/data2/audio/music-singing/ArabicClips/"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    input_path = args.path
    output_path = args.out
    main()

