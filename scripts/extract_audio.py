#!/usr/bin/python
# coding: utf-8

import sys
import subprocess
import shutil
import os
import threading
import time
from pathlib import Path
from queue import Queue

class TaskThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue_obj = Queue()

    def add_data(self, _cmd, _srt, _spath):
        self.queue_obj.put((_cmd, _srt, _spath))
    def run(self):
        while not self.queue_obj.empty():
            (_cmd, _srt, _spath) = self.queue_obj.get()
            subprocess.check_output(_cmd, shell=True).decode('utf-8', 'ignore')
            print(_cmd)
            print('run', self.queue_obj.qsize())
            shutil.copy(_srt, _spath)

if __name__ == "__main__":
    src = Path('D:/语料/第四批语料')
    dest = Path('E:/语料/第四批语料')
    ffmpeg_exe = 'E:/win/ffmpeg.exe'
    threadCount = 20
    ts = []
    for i in range(threadCount):
        ts.append(TaskThread())
    i = 0
    for srt in src.glob("**/*.srt"):
        d = dest / Path(Path(srt.parent).parent).name / Path(srt.parent).name
        if not d.exists():
            os.makedirs(d)
        vpath = None
        for v in Path(srt.parent).glob(srt.stem + '.*'):
            if v.suffix == '.srt':
                continue
            vpath = v
            break
        if vpath is None:
            print(srt, srt.stem)
        apath = d / (srt.stem + ".wav")
        spath = d / (srt.stem + ".srt")
        if apath.exists() and spath.exists():
            continue
        cmd = f'{ffmpeg_exe} -analyzeduration 2147483647 -probesize 2147483647 -i "{vpath}" -max_muxing_queue_size 9999 -map_metadata -1 -map_chapters -1 -vn -ar 16000 -ac 1 -sample_fmt s16 -y "{apath}"'
        ts[i % threadCount].add_data(cmd, srt, spath)
        i += 1
    print('-'*20, i)
    for i in range(threadCount):
        ts[i].start()
