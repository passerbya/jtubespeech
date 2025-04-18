#!/usr/bin/python
# coding: utf-8


import argparse

import re
import os
import torch
from pathlib import Path
from torch.multiprocessing import Process, Queue

def scandir_generator(path):
    """仅列出目录中的文件"""
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(entry.path)

def main():
    if args.scp_path is not None:
        scp_file = Path(args.scp_path)
        with open(scp_file) as f:
            scp_list = f.readlines()
        scp_list = set([item.strip() for item in scp_list])
    else:
        scp_list = set()
    print(len(scp_list))

    fragment_pattern = re.compile(r"^(.*)_speech_\d+\.flac$")
    fragment_files = []
    for clip in scandir_generator(args.testset_dir):
        clip = clip.resolve()
        if clip.suffix != '.flac':
            continue
        if len(scp_list) > 0:
            raw_path = Path(str(clip).replace('/data1/', '/data2/').replace('_fragment/', '/'))
            filename = raw_path.name
            match = fragment_pattern.match(filename)
            if match:
                new_name = match.group(1) + ".flac"
                raw_path = raw_path.parent / new_name
            if str(raw_path) not in scp_list:
                continue

        fragment_files.append(str(clip).replace('/data1/', '/data2/'))

    with open(args.fragment_scp,'w',encoding='utf-8') as f:
        f.writelines([f'{item}\n' for item in fragment_files])
        f.flush()
    print(len(fragment_files))
    print("scan done.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.',
                        help='Path to the dir containing audio clips in .flac to be evaluated')
    parser.add_argument('-o', "--fragment_scp", default=None, help='')
    parser.add_argument('-s', "--scp_path", default=None, help='scp')

    args = parser.parse_args()

    main()
