#!/usr/bin/python
# coding: utf-8

# Usage:
# python sigmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.jsonl
#

import argparse

import os
import librosa
from soundfile import LibsndfileError
import time
import json
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from sigmos import SigMOS
from torch.multiprocessing import Process, Queue

def split_audio(audio, sr, segment_length=10):
    """ 切片音频，避免产生空片段 """
    samples_per_segment = sr * segment_length
    num_segments = len(audio) // samples_per_segment
    segments = [audio[i * samples_per_segment:(i + 1) * samples_per_segment]
                for i in range(num_segments) if len(audio[i * samples_per_segment:(i + 1) * samples_per_segment]) > 0]

    if len(audio) % samples_per_segment != 0:
        last_segment = audio[num_segments * samples_per_segment:]
        if len(last_segment) > 0:  # 只在有内容时添加
            segments.append(last_segment)

    return segments

def listen_worker(in_queue, jsonl_file):
    print("listen_worker started.")

    for clip_dict in iter(in_queue.get, "STOP"):
        print('listen_worker', clip_dict['filename'], clip_dict['MOS_OVRL'], clip_dict['MOS_SIG'], clip_dict['MOS_DISC'])
        with open(jsonl_file,'a',encoding='utf-8') as f:
            line = json.dumps(clip_dict)
            f.write(f'{line}\n')
            f.flush()

    print("listen_worker ended.")


def compute_segments(flac, sigmos_estimator):
    audio_data, sr = librosa.load(str(flac), sr=None)
    if sr != sigmos_estimator.sampling_rate:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sigmos_estimator.sampling_rate, res_type=sigmos_estimator.resample_type)
        sr = sigmos_estimator.sampling_rate  # 更新 sr
    #print(sr, flac)
    segments = split_audio(audio_data, sr, segment_length=10)
    scores = []
    for i, seg in enumerate(segments):
        duration = len(seg) / sr
        features = sigmos_estimator.stft(seg)
        #print(f"Segment {i}: shape={seg.shape}, STFT shape={features.shape}")
        if duration < 0.5 or features.shape[0] == 0 or features.shape[1] == 0:
            print(f"Skipping segment {i}, shape: {features.shape}, STFT produced empty feature map.")
            continue
        scores.append(sigmos_estimator.run(seg, sr=sr))
    avg_scores = {key: np.mean([s[key] for s in scores]) for key in scores[0]}
    avg_scores['filename'] = str(flac)
    return avg_scores

def compute_worker(in_queue, out_queue, script_dir, num):
    print(f"compute_worker {num} started")
    print('loading ...')
    sigmos_estimator = SigMOS(model_dir=script_dir, device_id=num % torch.cuda.device_count())
    print('load done')
    for flac in iter(in_queue.get, "STOP"):
        try:
            audio_data, sr = librosa.load(str(flac), sr=None)
            if sr != sigmos_estimator.sampling_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sigmos_estimator.sampling_rate, res_type=sigmos_estimator.resample_type)
                sr = sigmos_estimator.sampling_rate  # 更新 sr
            features = sigmos_estimator.stft(audio_data)
            duration = len(audio_data) / sr  # 计算音频时长
            #print(f"Loaded: {flac}, shape: {features.shape}, sample rate: {sr}, duration: {duration:.2f} seconds")
            if duration < 0.5 or features.shape[0] == 0 or features.shape[1] == 0:
                print(f"Skipping flac {flac}, shape: {features.shape}, duration: {duration:.2f} seconds, STFT produced empty feature map.")
                continue
            clip_dict = sigmos_estimator.run(audio_data, sr=sr)
            clip_dict['filename'] = str(flac)
            out_queue.put(clip_dict)
        except ValueError as e:
            print('ValueError', flac, e)
        except LibsndfileError:
            print('LibsndfileError', flac)
        except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
            if "Failed to allocate memory" in str(e):
                #print('oom1', flac, num % torch.cuda.device_count(), e)
                del sigmos_estimator
                torch.cuda.empty_cache()  # 清理 PyTorch 显存
                torch.cuda.ipc_collect()  # 释放 CUDA 共享显存
                time.sleep(5)  # 稍作等待，确保显存释放生效
                sigmos_estimator = SigMOS(model_dir=script_dir, device_id=num % torch.cuda.device_count())
                clip_dict = compute_segments(flac, sigmos_estimator)
                out_queue.put(clip_dict)
            else:
                raise

    print(f"compute_worker {num} stopped")

def scandir_generator(path):
    """仅列出目录中的文件"""
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_symlink() and Path(entry.path).resolve().is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(entry.path)

def start_worker(in_queue, out_queue, script_dir, num):
    """ 启动一个新的 compute_worker 进程 """
    p = Process(target=compute_worker, args=(in_queue, out_queue, script_dir, num))
    p.start()
    return p

def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent

    jsonl_file = Path(args.jsonl_path)
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            mos_list = f.readlines()
        mos_list = set([json.loads(item.strip())['filename'] for item in mos_list])
    else:
        mos_list = set()
        jsonl_file.touch()
    print(len(mos_list))

    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)
    done_queue = Queue()

    # Start worker processes
    Process(
        target=listen_worker,
        args=(
            done_queue,
            jsonl_file
        ),
    ).start()

    workers = []
    for i in range(NUMBER_OF_PROCESSES):
        p = start_worker(task_queue, done_queue, script_dir, i)
        workers.append(p)

    for clip in scandir_generator(args.testset_dir):
        # 监控进程状态，自动重启崩溃的进程
        for i, p in enumerate(workers):
            if not p.is_alive():  # 如果进程退出（OOM 导致）
                print(f"⚠️ Worker {i} crashed. Restarting...")
                new_p = start_worker(task_queue, done_queue, script_dir, i)
                workers[i] = new_p  # 更新 worker 列表
        clip = clip.resolve()
        if clip.suffix != '.flac':
            continue
        if str(clip) in mos_list:
            continue
        task_queue.put(clip)

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")
    while not task_queue.empty() or not done_queue.empty():
        time.sleep(20)
    done_queue.put("STOP")
    print("compute done.")

NUMBER_OF_PROCESSES = torch.cuda.device_count()
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.',
                        help='Path to the dir containing audio clips in .flac to be evaluated')
    parser.add_argument('-o', "--jsonl_path", default=None, help='Dir to the jsonl that saves the results')

    args = parser.parse_args()

    main()
