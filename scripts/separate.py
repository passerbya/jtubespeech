#!/usr/bin/python
# coding: utf-8

import sys
import shutil
#import demucs.separate
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue, LifoQueue
from datetime import datetime

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
        #self.queue_obj = Queue()
        self.cuda_num = cuda_num

    def add_data(self, cmd, outdir, wav_dest, wav_src):
        self.queue_obj.put((cmd, outdir, wav_dest, wav_src))

    def run(self):
        while not self.queue_obj.empty():
            '''
            now = datetime.now()
            current_hour = now.hour
            if (self.cuda_num == 0 and not (3 <= current_hour < 7)) or (self.cuda_num == 1 and not (0 <= current_hour < 25)):
                time.sleep(60)
                continue
            '''

            (cmd, outdir, wav_dest, wav_src) = self.queue_obj.get()
            if wav_dest.exists():
                continue
            print(cmd)

            #demucs.separate.main(["-d", f"cuda:{self.cuda_num}", "-n", "htdemucs_ft", "--shifts", "4", "--two-stems", "vocals", "-o", str(outdir), wav_src])
            subprocess.check_output(cmd, shell=True, executable='/bin/bash')
            '''
            ffmpeg_exe = '/usr/local/ffmpeg/bin/ffmpeg'
            temp_path = outdir / 'htdemucs_ft' / wav_dest.stem / 'vocals.wav'
            cmd = f'{ffmpeg_exe} -i "{temp_path}" -vn -ar 24000 -ac 1 -sample_fmt s16 -y "{wav_dest}"'
            subprocess.check_output(cmd, shell=True)
            '''
            temp_path = outdir / 'htdemucs_ft' / wav_src.stem / 'vocals.wav'
            temp_path.rename(wav_dest)

            try:
                delete_folder(outdir / 'htdemucs_ft' / wav_src.stem)
            except:
                pass

def main():
    thread_count = 2
    ts = []
    for i in range(thread_count):
        ts.append(TaskThread(i))
    i = 0
    for lang_dir in (src / 'txt').iterdir():
        if not lang_dir.is_dir():
            continue
        for srt in lang_dir.iterdir():
            if srt.name == 'segments.txt' or srt.suffix != '.txt':
                continue
            wav_dest = Path(str(srt).replace('/txt/', '/wav/').replace('.txt', '.wav'))
            print(wav_dest)
            if wav_dest.exists():
                continue
            if not wav_dest.parent.exists():
                wav_dest.parent.mkdir(parents=True)
            outdir = src / 'temp'
            wav_src = str(srt).replace('/txt/', '/wav_org/').replace('.txt', '.wav')
            cmd = f"source /etc/profile && /root/miniconda3/bin/demucs -d cuda:{i % thread_count} -n htdemucs_ft -j 4 --shifts=4 -o {outdir} {wav_src}"
            ts[i % thread_count].add_data(cmd, outdir, wav_dest, wav_src)
            i += 1

    print('-'*20, i)
    for i in range(thread_count):
        ts[i].start()

if __name__ == "__main__":
    src = Path('/usr/local/corpus/4th_biz/en')
    main()

