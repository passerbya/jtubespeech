#!/usr/bin/python
# coding: utf-8

import torch
import argparse
import librosa
import soundfile as sf
from pathlib import Path
from torch.multiprocessing import Process, Queue

def find_files(flacdir, txtdir):
    files_dict = {}
    txt_dict = {}
    for item in txtdir.glob("**/*.txt"):
        txt_dict[item.stem] = item
    for flac in flacdir.glob("**/*.flac"):
        if flac.stem.endswith('_16k'):
            continue
        files_dict[flac.stem] = (flac, txt_dict[flac.stem])
    return files_dict

def slice_worker(_src, num, task_queue):
    print(f"slice_worker {num} started")

    for flac, txt in iter(task_queue.get, "STOP"):
        stem = flac.stem
        flac_dest = src / 'speech' / flac.relative_to(src / 'wav_org')
        audio, sample_rate = sf.read(flac)
        if sample_rate != 44100:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=44100)
            sample_rate = 44100
        with open(str(txt)) as f:
            lines = f.readlines()
        flac_dest.parent.mkdir(parents=True, exist_ok=True)
        for i, line in enumerate(lines):
            utt1 = line.strip()
            rec_id = f"{stem}_{i:04}"
            segment_path = flac_dest.parent / (rec_id + '.flac')
            utt_start1, utt_end1, _ = utt1.split("\t", 2)
            utt_start1 = float(utt_start1)
            utt_end1 = float(utt_end1)
            start_idx = int(utt_start1 * sample_rate)
            end_idx = int(utt_end1 * sample_rate)
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(audio):
                end_idx = len(audio)
            audio_slice = audio[start_idx:end_idx]
            sf.write(str(segment_path), format="flac", data=audio_slice, samplerate=44100)
        print('slice done', flac)
    print(num, 'done')

def main():
    flacdir = Path(src) / 'wav_org'
    txtdir = Path(src) / 'txt'
    # find files
    files_dict = find_files(flacdir, txtdir)
    num_files = len(files_dict)
    print(f"Found {num_files} files.")

    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = Process(
            target=slice_worker,
            args=(src, i, task_queue),
        )
        p.start()
        processes.append(p)

    for stem in files_dict.keys():
        (flac, txt) = files_dict[stem]
        task_queue.put((flac, txt))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()

    print("slice done.")


NUMBER_OF_PROCESSES = 4
if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/corpus/4th_biz/demo"))
    args = parser.parse_args()
    src = args.path
    main()

