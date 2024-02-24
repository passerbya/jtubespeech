#!/usr/bin/python
# coding: utf-8

import sys
import shutil
import demucs.separate
import subprocess
import threading
from pathlib import Path
from queue import Queue, LifoQueue

def delete_folder(pth) :
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
            sub.rmdir()
        else:
            sub.unlink()
    pth.rmdir()

class TaskThread(threading.Thread):

    def __init__(self, cuda_num):
        threading.Thread.__init__(self)
        self.queue_obj = LifoQueue()
        self.cuda_num = cuda_num

    def add_data(self, cmd, outdir, wav_dest, wav_src):
        self.queue_obj.put((cmd, outdir, wav_dest, wav_src))
    def run(self):
        while not self.queue_obj.empty():
            (cmd, outdir, wav_dest, wav_src) = self.queue_obj.get()
            print(cmd)

            demucs.separate.main(["-d", f"cuda:{self.cuda_num}", "-n", "htdemucs_ft", "--shifts", "4", "--two-stems", "vocals", "-o", str(outdir), wav_src])
            ffmpeg_exe = '/usr/local/ffmpeg/bin/ffmpeg'
            temp_path = outdir / 'htdemucs_ft' / wav_dest.stem / 'vocals.wav'
            cmd = f'{ffmpeg_exe} -i "{temp_path}" -vn -ar 24000 -ac 1 -sample_fmt s16 -y "{wav_dest}"'
            subprocess.check_output(cmd, shell=True)
            try:
                delete_folder(outdir / 'htdemucs_ft' / wav_dest.stem)
            except:
                pass

def main():
    thread_count = 1
    ts = []
    for i in range(thread_count):
        ts.append(TaskThread(i))
    i = 0
    for lang_dir in src.glob("*"):
        if not lang_dir.is_dir():
            continue
        for srt in lang_dir.glob("**/*.txt"):
            if srt.name == 'segments.txt':
                continue
            wav_dest = Path(str(srt).replace('/txt/', '/wav/').replace('.txt', '.wav'))
            print(wav_dest)
            if wav_dest.exists():
                continue
            if not wav_dest.parent.exists():
                wav_dest.parent.mkdir(parents=True)
            outdir = src / 'temp'
            wav_src = str(srt).replace('/txt/', '/wav_org/').replace('.txt', '.wav')
            cmd = f"source /etc/profile && /root/miniconda3/bin/demucs -d cuda:{i % thread_count} -n htdemucs_ft --shifts=4 -o {outdir} {wav_src}"
            ts[i % thread_count].add_data(cmd, outdir, wav_dest, wav_src)
            i += 1

    print('-'*20, i)
    for i in range(thread_count):
        ts[i].start()

if __name__ == "__main__":
    src = Path('/usr/local/corpus/4th_biz/zh')
    main()

