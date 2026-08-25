#!/usr/bin/python
# coding: utf-8

# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.jsonl -p
#

import argparse
import hashlib
import os

# Prevent every worker/session from creating a large CPU thread pool. Use the
# task-specific DNSMOS_CPU_THREADS variable to override the default of one.
CPU_THREAD_COUNT = int(os.environ.get("DNSMOS_CPU_THREADS", "1"))
if CPU_THREAD_COUNT < 1:
    raise ValueError("DNSMOS_CPU_THREADS must be at least 1")
for thread_env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
):
    os.environ[thread_env_var] = str(CPU_THREAD_COUNT)

from queue import Empty

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from soundfile import LibsndfileError
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.multiprocessing import Process, Queue

torch.set_num_threads(CPU_THREAD_COUNT)

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def bucket_for_filename(path: Path, buckets: int) -> int:
    if buckets == 1:
        return 0
    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % buckets


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, device_id=0) -> None:
        providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = CPU_THREAD_COUNT
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

        self.onnx_sess = ort.InferenceSession(
            primary_model_path,
            sess_options=session_options,
            providers=providers,
        )
        print("Current providers:", self.onnx_sess.get_providers())
        self.p808_onnx_sess = ort.InferenceSession(
            p808_model_path,
            sess_options=session_options,
            providers=providers,
        )

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def prepare_clip(self, fpath, sampling_rate):
        aud, input_fs = sf.read(fpath)
        if aud.ndim > 1:
            aud = np.mean(aud, axis=1)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        if actual_audio_len == 0:
            raise ValueError("empty audio")
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        input_features = []
        p808_input_features = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features.append(np.asarray(audio_seg, dtype=np.float32))
            p808_input_features.append(
                np.asarray(self.audio_melspec(audio=audio_seg[:-160]), dtype=np.float32)
            )

        if not input_features:
            raise ValueError("audio produced no valid DNSMOS windows")

        return {
            'filename': fpath,
            'len_in_sec': actual_audio_len / fs,
            'sr': fs,
            'num_hops': num_hops,
            'input_features': input_features,
            'p808_input_features': p808_input_features,
        }

    def score_files(self, fpaths, sampling_rate, is_personalized_MOS, batch_size):
        clips = []
        failures = []
        input_features = []
        p808_input_features = []
        owners = []

        for fpath in fpaths:
            try:
                clip = self.prepare_clip(fpath, sampling_rate)
            except (LibsndfileError, ValueError) as error:
                failures.append((fpath, error))
                continue

            owner = len(clips)
            clip_input_features = clip.pop('input_features')
            clip_p808_input_features = clip.pop('p808_input_features')
            clips.append(clip)
            input_features.extend(clip_input_features)
            p808_input_features.extend(clip_p808_input_features)
            owners.extend([owner] * len(clip_input_features))

        predictions = [
            {
                'SIG_raw': [],
                'BAK_raw': [],
                'OVRL_raw': [],
                'SIG': [],
                'BAK': [],
                'OVRL': [],
                'P808_MOS': [],
            }
            for _ in clips
        ]

        for start in range(0, len(input_features), batch_size):
            end = start + batch_size
            input_batch = np.stack(input_features[start:end])
            p808_input_batch = np.stack(p808_input_features[start:end])

            p808_batch = self.p808_onnx_sess.run(
                None,
                {'input_1': p808_input_batch},
            )[0].reshape(-1)
            primary_batch = self.onnx_sess.run(
                None,
                {'input_1': input_batch},
            )[0]
            mos_sig_raw = primary_batch[:, 0]
            mos_bak_raw = primary_batch[:, 1]
            mos_ovr_raw = primary_batch[:, 2]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw,
                mos_bak_raw,
                mos_ovr_raw,
                is_personalized_MOS,
            )

            for index, owner in enumerate(owners[start:end]):
                prediction = predictions[owner]
                prediction['SIG_raw'].append(mos_sig_raw[index])
                prediction['BAK_raw'].append(mos_bak_raw[index])
                prediction['OVRL_raw'].append(mos_ovr_raw[index])
                prediction['SIG'].append(mos_sig[index])
                prediction['BAK'].append(mos_bak[index])
                prediction['OVRL'].append(mos_ovr[index])
                prediction['P808_MOS'].append(p808_batch[index])

        clip_dicts = []
        for clip, prediction in zip(clips, predictions):
            clip_dict = dict(clip)
            clip_dict['OVRL_raw'] = np.mean(prediction['OVRL_raw'])
            clip_dict['SIG_raw'] = np.mean(prediction['SIG_raw'])
            clip_dict['BAK_raw'] = np.mean(prediction['BAK_raw'])
            clip_dict['OVRL'] = np.mean(prediction['OVRL'])  # P835 MOS, Emilia>3.2 4.0播音员 3.5以上OK
            clip_dict['SIG'] = np.mean(prediction['SIG'])
            clip_dict['BAK'] = np.mean(prediction['BAK'])
            clip_dict['P808_MOS'] = np.mean(prediction['P808_MOS'])  # WenetSpeech4TTS Premium>4.0  Standard>3.8  Basic>3.6
            clip_dicts.append(clip_dict)

        return clip_dicts, failures

    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        clip_dicts, failures = self.score_files(
            [fpath],
            sampling_rate,
            is_personalized_MOS,
            batch_size=1,
        )
        if failures:
            raise failures[0][1]
        return clip_dicts[0]


def listen_worker(in_queue, jsonl_file):
    print("listen_worker started.")

    i = 0
    for clip_dict in iter(in_queue.get, "STOP"):
        print('listen_worker', clip_dict['filename'], clip_dict['OVRL'], clip_dict['P808_MOS'])
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            clip_dict = {key: float(value) if isinstance(value, np.float32) else value for key, value in clip_dict.items()}
            line = json.dumps(clip_dict)
            f.write(f'{line}\n')
            i += 1
            if i % 500 == 0:
                f.flush()

    print("listen_worker ended.")


def compute_worker(
        in_queue,
        out_queue,
        primary_model_path,
        p808_model_path,
        worker_num,
        device_id,
        batch_size,
        is_personalized_eval,
):
    print(f"compute_worker {worker_num} started on GPU {device_id}")
    print('loading ...')
    compute_score = ComputeScore(primary_model_path, p808_model_path, device_id)
    print('load done')
    desired_fs = SAMPLING_RATE

    stop_requested = False
    while not stop_requested:
        flac = in_queue.get()
        if flac == "STOP":
            break

        flacs = [flac]
        while len(flacs) < batch_size:
            try:
                flac = in_queue.get(timeout=0.05)
            except Empty:
                break
            if flac == "STOP":
                stop_requested = True
                break
            flacs.append(flac)

        clip_dicts, failures = compute_score.score_files(
            [str(item) for item in flacs],
            desired_fs,
            is_personalized_eval,
            batch_size,
        )
        for clip_dict in clip_dicts:
            out_queue.put(clip_dict)

        for failed_flac, error in failures:
            print(type(error).__name__, failed_flac, error)
            try:
                Path(failed_flac).unlink(missing_ok=True)
                print('deleted bad audio file', failed_flac)
            except OSError as delete_error:
                print('delete failed', failed_flac, delete_error)
    print(f"compute_worker {worker_num} stopped")


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

    jsonl_file = Path(args.jsonl_path)
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            mos_list = f.readlines()
        mos_list = set([json.loads(item.strip())['filename'] for item in mos_list])
    else:
        mos_list = set()
        jsonl_file.touch()
    print(len(mos_list))
    print(f"bucket={args.bucket}/{args.buckets}")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No CUDA GPU detected")
    number_of_processes = args.workers_per_gpu * gpu_count
    print(
        f"gpus={gpu_count}, workers={number_of_processes}, "
        f"workers_per_gpu={args.workers_per_gpu}, batch_size={args.batch_size}"
    )

    task_queue = Queue(maxsize=2048)
    done_queue = Queue()

    # Start worker processes
    listener = Process(
        target=listen_worker,
        args=(
            done_queue,
            jsonl_file
        ),
    )
    listener.start()

    workers = []
    for i in range(number_of_processes):
        worker = Process(
            target=compute_worker,
            args=(
                task_queue,
                done_queue,
                primary_model_path,
                p808_model_path,
                i,
                i % gpu_count,
                args.batch_size,
                args.personalized_MOS,
            ),
        )
        worker.start()
        workers.append(worker)

    skipped_bucket = 0
    testset_dir = Path(args.testset_dir).absolute()
    for clip in scandir_generator(testset_dir):
        if clip.suffix != '.flac':
            continue
        if bucket_for_filename(clip, args.buckets) != args.bucket:
            skipped_bucket += 1
            continue
        if str(clip) in mos_list:
            continue
        task_queue.put(clip)

    print(f"bucket scan done, skipped_bucket={skipped_bucket}")

    # Tell child processes to stop
    for i in range(number_of_processes):
        task_queue.put("STOP")

    for worker in workers:
        worker.join()
    done_queue.put("STOP")
    listener.join()

    failed_workers = [worker.pid for worker in workers if worker.exitcode != 0]
    if failed_workers:
        raise RuntimeError(f"Compute workers failed: {failed_workers}")
    print("compute done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.',
                        help='Path to the dir containing audio clips in .flac to be evaluated')
    parser.add_argument('-o', "--jsonl_path", default=None, help='Dir to the jsonl that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true',
                        help='Flag to indicate if personalized MOS score is needed or regular')
    parser.add_argument("--buckets", type=int, default=1, help="Total number of buckets. 1 disables bucketing.")
    parser.add_argument("--bucket", type=int, default=0, help="Zero-based bucket index for this server.")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Compute worker processes per GPU.")
    parser.add_argument("--batch_size", type=int, default=32, help="Maximum ONNX inference batch size.")

    args = parser.parse_args()
    if args.buckets < 1:
        parser.error("--buckets must be at least 1")
    if not 0 <= args.bucket < args.buckets:
        parser.error("--bucket must be in the range [0, --buckets)")
    if args.workers_per_gpu < 1:
        parser.error("--workers_per_gpu must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    main()
