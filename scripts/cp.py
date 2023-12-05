#!/usr/bin/python
# coding: utf-8

import os
import shutil
import threading
from pathlib import Path
from queue import Queue

class TaskThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue_obj = Queue()

    def add_data(self, _s, _d):
        self.queue_obj.put((_s, _d))
    def run(self):
        while not self.queue_obj.empty():
            (_s, _d) = self.queue_obj.get()
            shutil.copy(_s, _d)
            stinfo = os.stat(_s)
            os.utime(_d, (stinfo.st_atime, stinfo.st_mtime))
            print(f'{_d} ok')

if __name__ == "__main__":
    src = Path('/usr/local/data/jtubespeech')
    dest = Path('/usr/local/share/ooo/38语料/语料盘/jtubespeech')
    threadCount = 20
    ts = []
    for i in range(threadCount):
        ts.append(TaskThread())
    i = 0
    for s in src.glob("**/*"):
        d = dest / s.relative_to(src)
        if s.is_dir():
            continue
        if i % threadCount == 0:
            print(d)
        if d.exists() and d.stat().st_mtime == s.stat().st_mtime and d.stat().st_size == s.stat().st_size:
            continue
        if not d.parent.exists():
            d.parent.mkdir(parents=True)
        ts[i % threadCount].add_data(s, d)
        i += 1
    print('-'*20, i)
    for i in range(threadCount):
        ts[i].start()
