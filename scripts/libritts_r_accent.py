#!/usr/bin/python
# coding: utf-8

import os
import shutil
import torch
import librosa
import argparse
from pathlib import Path
from speechbrain.pretrained.interfaces import foreign_class

def scandir_generator(path):
    """仅列出目录中的文件"""
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(entry.path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/corpus/SpeechData_flac/en_vctk"))
    parser.add_argument("--suffix", type=str, default=".flac")
    args = parser.parse_args()
    src = Path(args.path)
    suffix = args.suffix

    classifier = foreign_class(source="/usr/local/corpus/accent/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", run_opts={'device':'cuda:0'})

    label_file_dict = {'us':[], 'uk':[], 'o':[]}
    file_handles = {}
    file_already_classified = set()
    for label in label_file_dict:
        list_file = src/f'{label}.list'
        if not list_file.exists():
            list_file.touch()
            continue
        with open(list_file, 'r') as f:
            for line in f:
                file_already_classified.add(line.strip())
                label_file_dict[label].append(line.strip())
    stat = {}
    for label, files in label_file_dict.items():
        list_file = src/f'{label}.list'
        stat[label] = len(files)
        file_handles[label] = open(list_file, 'a', encoding='utf-8')

    for wav in scandir_generator(str(src)):
        fpath = str(wav)
        if not wav.name.endswith(args.suffix) or fpath in file_already_classified:
            continue
        print(fpath)
        sig, _ = librosa.load(fpath, sr=16000, duration=60)
        sig = torch.tensor(sig)
        out_prob, score, index, text_lab = classifier.classify_batch(sig)
        #out_prob, score, index, text_lab = classifier.classify_file(fpath)
        if score.item() > 0.85:
            label = text_lab[0]
        else:
            label = 'o'
        print(score.item(), label, stat)
        label_file_dict[label].append(fpath)
        stat[label] += 1
        file_already_classified.add(fpath)
        f = file_handles[label]
        f.write(f'{fpath}\n')
        if len(label_file_dict[label]) % 100:
            f.flush()

    for label, file_handle in file_handles.items():
        file_handle.close()
    print(stat)

