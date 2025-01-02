#!/usr/bin/python
# coding: utf-8

# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import argparse

import os
import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import time
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.multiprocessing import Process, Queue

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, num=0) -> None:
        providers = [('CUDAExecutionProvider', {'device_id': num % torch.cuda.device_count()}), 'CPUExecutionProvider']
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=providers)
        print("Current providers:", self.onnx_sess.get_providers())
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=providers)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg) #P835 MOS, Emilia>3.2 4.0播音员 3.5以上OK
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos) #WenetSpeech4TTS Premium>4.0  Standard>3.8  Basic>3.6
        return clip_dict

def listen_worker(in_queue, jsonl_file):
    print("listen_worker started.")

    for clip_dict in iter(in_queue.get, "STOP"):
        print('listen_worker', clip_dict['filename'], clip_dict['OVRL'], clip_dict['P808_MOS'])
        with open(jsonl_file,'a',encoding='utf-8') as f:
            clip_dict = {key: float(value) if isinstance(value, np.float32) else value for key, value in clip_dict.items()}
            line = json.dumps(clip_dict)
            f.write(f'{line}\n')
            f.flush()

    print("listen_worker ended.")

def compute_worker(in_queue, out_queue, primary_model_path, p808_model_path, num):
    print(f"compute_worker {num} started")
    print('loading ...')
    compute_score = ComputeScore(primary_model_path, p808_model_path, num)
    print('load done')
    is_personalized_eval = args.personalized_MOS
    desired_fs = SAMPLING_RATE
    for flac in iter(in_queue.get, "STOP"):
        clip_dict = compute_score(str(flac), desired_fs, is_personalized_eval)
        out_queue.put(clip_dict)
    print(f"compute_worker {num} stopped")

def scandir_generator(path):
    """仅列出目录中的文件"""
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                yield Path(entry.path)
            elif entry.is_dir():
                yield from scandir_generator(entry.path)

def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    # 定义模型路径
    p808_model_path = str(script_dir / 'model_v8.onnx')
    primary_model_path = str(script_dir / ('pDNSMOS' if args.personalized_MOS else 'DNSMOS') / 'sig_bak_ovr.onnx')

    jsonl_file = Path(args.csv_path)
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

    for i in range(NUMBER_OF_PROCESSES):
        Process(target=compute_worker, args=(task_queue, done_queue, primary_model_path, p808_model_path, i)).start()

    for clip in scandir_generator(args.testset_dir):
        if clip.suffix != '.flac':
            continue
        if str(clip) in mos_list:
            continue
        task_queue.put(clip.resolve())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")
    while not task_queue.empty() or not done_queue.empty():
        time.sleep(20)
    done_queue.put("STOP")
    print("compute done.")

NUMBER_OF_PROCESSES = 1
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .flac to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true', 
                        help='Flag to indicate if personalized MOS score is needed or regular')
    
    args = parser.parse_args()

    main()
