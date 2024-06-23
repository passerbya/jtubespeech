#!/usr/bin/python
# coding: utf-8

import subprocess
from pathlib import Path
from torch.multiprocessing import Process, Queue

def separate_worker(num, task_queue):
    outdir = src / 'temp'
    outdir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = '/usr/local/ffmpeg/bin/ffmpeg'
    for wav_dest, wav_src in iter(task_queue.get, "STOP"):
        temp_path = outdir / wav_dest.name
        cmd = f'{ffmpeg_exe} -i "{wav_src}" -y "{temp_path}"'
        print(cmd)
        subprocess.run(cmd, shell=True)
        temp_path.rename(wav_dest)
        if wav_dest.exists():
            wav_src.unlink()
    print(num, 'done')

def main():
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(
            target=separate_worker,
            args=(i, task_queue),
        ).start()

    for wav_src in src.rglob('*.wav'):
        wav_dest = wav_src.parent / (wav_src.stem + '.flac')
        print(wav_dest)
        if wav_dest.exists():
            continue
        task_queue.put((wav_dest, wav_src))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    print("convert done.")


NUMBER_OF_PROCESSES = 3
if __name__ == "__main__":
    src = Path('/usr/local/corpus/4th_wav')
    main()

