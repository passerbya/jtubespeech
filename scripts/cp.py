#!/usr/bin/python
# coding: utf-8

import os
import shutil
import threading
import argparse
from pathlib import Path
from queue import Queue

class TaskThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue_obj = Queue()

    def add_data(self, _s, _d):
        self.queue_obj.put((_s, _d))

    def run(self):
        while True:
            (_s, _d) = self.queue_obj.get()
            if _s is None and _d is None:
                break
            if _d.exists() and int(_d.stat().st_mtime) == int(_s.stat().st_mtime) and _d.stat().st_size == _s.stat().st_size:
                print(d, 'exists')
                continue
            if not _d.parent.exists():
                _d.parent.mkdir(parents=True)
            stinfo = os.stat(_s)
            if os.name == 'nt':
                atime = int(stinfo.st_atime)
                mtime = int(stinfo.st_mtime)
            try:
                shutil.copy(_s, _d)
                os.utime(_d, (atime, mtime))
            except OSError:
                _d = _d.parent / (_d.stem[:_d.stem.rfind(' ')] + _d.suffix)
                print(_d)
                try:
                    shutil.copy(_s, _d)
                    os.utime(_d, (atime, mtime))
                except OSError:
                    _d = _d.parent / (_d.stem[:int(len(_d.stem)/2)] + _d.suffix)
                    print(_d)
                    shutil.copy(_s, _d)
                    os.utime(_d, (atime, mtime))
            print(f'{_d} ok')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="D:/38语料/语料盘/语料")
    parser.add_argument("--dest", type=str, default="W:/")
    parser.add_argument("--threads", type=int, default=50)
    args = parser.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    threadCount = args.threads
    ts = []
    for i in range(threadCount):
        ts.append(TaskThread())
        ts[i].start()
    i = 0
    for s in src.glob("**/*"):
        d = dest / s.relative_to(src)
        if s.is_dir():
            continue
        idx = i % threadCount
        ts[idx].add_data(s, d)
        i += 1
    print('-'*20, i)
    for i in range(threadCount):
        ts[i].add_data(None, None)