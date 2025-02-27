#!/usr/bin/python
# coding: utf-8

import re
import os
import sys
import json
import shutil
import torch
import argparse
import librosa, vocal
import soundfile as sf
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
    device = f"cuda:{device_id}"
    script_dir = Path(__file__).parent
    voc_ft_model_path = str(script_dir / 'UVR-MDX-NET-Voc_FT.cuda.pt')
    vocal_separator = torch.jit.load(voc_ft_model_path).to(device)

    for flac_src in iter(task_queue.get, "STOP"):
        if flac_src.is_dir() or flac_src.suffix != '.flac':
            continue
        flac_dest = src / 'speech_sep' / flac_src.relative_to(src / 'speech')
        if flac_dest.exists():
            continue
        json_path = Path(str(flac_src.parent).replace('/speech/','/speech_json/')) / f'{flac_src.stem}.json'
        if not json_path.exists():
            continue
        with open(str(json_path), 'r', encoding='utf-8') as f:
            data = json.load(f)
        tags = set()
        if isinstance(data, list):  # 处理JSON是列表的情况
            for item in data:
                text = item.get("text", "")
                tags = set(extract_tags(text))
        print(flac_src, tags)
        #if 'Speech' not in tags:
        #    continue
        if 'Breath' in tags or 'Laughter' in tags or 'Applause' in tags or 'Sneeze' in tags or 'Cough' in tags or 'Cry' in tags:
            continue
        if 'ANGRY' not in tags and 'HAPPY' not in tags and 'SAD' not in tags and 'SURPRISED' not in tags and 'FEARFUL' not in tags and 'DISGUSTED' not in tags:
            continue

        flac_dest.parent.mkdir(parents=True, exist_ok=True)
        print(flac_dest, flac_src)
        flac, sr = librosa.load(str(flac_src), mono=True, sr=44100)
        wav_vocal = vocal.separate_vocal(vocal_separator, flac, device, silent=False)[0]
        sf.write(str(flac_dest), format="flac", data=wav_vocal.T, samplerate=44100)
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

    for lang_dir in (src/'speech').iterdir():
        for flac_path in scandir_generator(lang_dir/'wav_org'):
            task_queue.put(flac_path)

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
    parser.add_argument("--path", type=Path, default=Path("/usr/local/corpus/4th_biz"))
    args = parser.parse_args()
    src = args.path
    main()

