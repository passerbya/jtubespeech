#!/usr/bin/python
# coding: utf-8

import sys
import shutil
#import demucs.separate
import torch, torchaudio
import argparse
from pathlib import Path
from torch.multiprocessing import Process, Queue
from torchaudio.functional import resample
from vocalextractor import VocalExtractor


def delete_folder(pth) :
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
            sub.rmdir()
        else:
            sub.unlink()
    pth.rmdir()

def separate_worker(_src, _checkpoint_dir, cuda_num, task_queue):
    #outdir = _src / 'temp'
    model = VocalExtractor(str(_checkpoint_dir/'checkpoint.pkl'), str(_checkpoint_dir/'config.yml'), device=f"cuda:{cuda_num}")
    for wav_dest, wav_src in iter(task_queue.get, "STOP"):
        if wav_dest.exists():
            continue
        '''
        demucs.separate.main(["-d", f"cuda:{cuda_num}", "-n", "htdemucs_ft", "--shifts", "4", "--two-stems", "vocals", "-o", str(outdir), wav_src])
        wav_src = Path(wav_src)
        temp_path = outdir / 'htdemucs_ft' / wav_src.stem / 'vocals.wav'
        temp_path.rename(wav_dest)
        delete_folder(outdir / 'htdemucs_ft' / wav_src.stem)
        '''

        y, sr = torchaudio.load(str(wav_src))
        if sr != model.sampling_rate: y = resample(y, sr, model.sampling_rate)
        vocal, _ = model(y)
        torchaudio.save(str(wav_dest), vocal, model.sampling_rate, bits_per_sample=16, format='flac')
    print(cuda_num, 'done')

def main():
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = Process(
            target=separate_worker,
            args=(src, checkpoint_dir, i % torch.cuda.device_count(), task_queue),
        )
        p.start()
        processes.append(p)

    for _dir in (src / 'txt').iterdir():
        if not _dir.is_dir():
            continue
        for txt in _dir.iterdir():
            if txt.name == 'segments.txt' or txt.suffix != '.txt':
                continue
            wav_dest = Path(str(txt).replace('/txt/', '/flac/').replace('.txt', '.flac'))
            print(wav_dest)
            if wav_dest.exists():
                continue
            if not wav_dest.parent.exists():
                wav_dest.parent.mkdir(parents=True)
            wav_src = str(txt).replace('/txt/', '/wav_org/').replace('.txt', '.flac')
            task_queue.put((wav_dest, wav_src))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()

    print("separate done.")

NUMBER_OF_PROCESSES = 3 * torch.cuda.device_count()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/corpus/4th_biz/zh"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("/usr/local/data2/workspace/egs_vocal_extractor/exp/transformernet2"))
    args = parser.parse_args()
    src = args.path
    checkpoint_dir = args.checkpoint_dir
    main()

