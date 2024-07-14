#!/usr/bin/python
# coding: utf-8

import sys
import shutil
import demucs.separate
import torch
from pathlib import Path
from torch.multiprocessing import Process, Queue

def delete_folder(pth) :
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
            sub.rmdir()
        else:
            sub.unlink()
    pth.rmdir()

def separate_worker(_src, cuda_num, task_queue):
    outdir = _src / 'temp'
    for wav_dest, wav_src in iter(task_queue.get, "STOP"):
        demucs.separate.main(["-d", f"cuda:{cuda_num}", "-n", "htdemucs_ft", "--shifts", "4", "--two-stems", "vocals", "-o", str(outdir), wav_src])
        wav_src = Path(wav_src)
        temp_path = outdir / 'htdemucs_ft' / wav_src.stem / 'vocals.wav'
        temp_path.rename(wav_dest)
        delete_folder(outdir / 'htdemucs_ft' / wav_src.stem)
    print(cuda_num, 'done')

def main():
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(
            target=separate_worker,
            args=(src, i, task_queue),
        ).start()

    for _dir in (src / 'txt').iterdir():
        if not _dir.is_dir():
            continue
        for txt in _dir.iterdir():
            if txt.name == 'segments.txt' or txt.suffix != '.txt':
                continue
            wav_dest = Path(str(txt).replace('/txt/', '/wav/').replace('.txt', '.wav'))
            print(wav_dest)
            if wav_dest.exists():
                continue
            if not wav_dest.parent.exists():
                wav_dest.parent.mkdir(parents=True)
            wav_src = str(txt).replace('/txt/', '/wav_org/').replace('.txt', '.wav')
            task_queue.put((wav_dest, wav_src))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    print("separate done.")


NUMBER_OF_PROCESSES = 2
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    src = Path('/usr/local/ocr/5th_biz/zh/')
    main()

