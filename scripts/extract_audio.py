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
            subprocess.check_output(_cmd, shell=True)
            print(_cmd)
            print('run', self.queue_obj.qsize())
            shutil.copy(_srt, _spath)

if __name__ == "__main__":
    src = Path('/usr/local/corpus/4th_orig')
    dest = Path('/usr/local/corpus/4th_wav')
    ffmpeg_exe = '/usr/local/ffmpeg/bin/ffmpeg'
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
        cmd = f'{ffmpeg_exe} -i "{vpath}" -vn -af aresample=async=1 -ac 1 -sample_fmt s16 -y "{apath}"'
        ts[i % threadCount].add_data(cmd, srt, spath)
        i += 1
    print('-'*20, i)
    for i in range(threadCount):
        ts[i].start()
