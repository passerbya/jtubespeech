#!/usr/bin/python
# coding: utf-8

import shlex
import subprocess
from pathlib import Path
from torch.multiprocessing import Process, Queue

def convert_worker(num, task_queue):
    outdir = src / 'temp'
    outdir.mkdir(parents=True, exist_ok=True)
    #ffmpeg_exe = '/usr/local/ffmpeg/bin/ffmpeg'
    ffmpeg_exe = '/usr/bin/ffmpeg'
    for wav_dest, wav_src in iter(task_queue.get, "STOP"):
        temp_path = outdir / wav_dest.name
        in_file = shlex.quote(str(wav_src))
        out_file = shlex.quote(str(temp_path))
        cmd = f'{ffmpeg_exe} -i "{in_file}" -ac 1 -y "{out_file}"'
        print(cmd)
        subprocess.run(cmd, shell=True)
        temp_path.rename(wav_dest)
        if wav_dest.exists():
            wav_src.unlink()
    print(num, 'done')

def main():
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)

    # Start worker processes
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = Process(
            target=convert_worker,
            args=(i, task_queue),
        )
        p.start()
        processes.append(p)

    for wav_src in src.rglob('*.wav'):
        wav_dest = wav_src.parent / (wav_src.stem + '.flac')
        print(wav_dest)
        if wav_dest.exists():
            continue
        task_queue.put((wav_dest, wav_src))

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()

    print("convert done.")


NUMBER_OF_PROCESSES = 6
if __name__ == "__main__":
    src = Path('/usr/local/ocr/5th_biz/zh/wav_org')
    main()

