#!/usr/bin/python
# coding: utf-8

# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.jsonl -p
#

import argparse
import hashlib
import io
import json
import os
import time

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
import soundfile as sf
from soundfile import LibsndfileError
import torch
import torchaudio
from pathlib import Path
from torch.multiprocessing import Process, Queue

torch.set_num_threads(CPU_THREAD_COUNT)

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
READER_DONE_EVENT = "READER_DONE"
COMPUTE_FAILURE_EVENT = "COMPUTE_FAILURE"


def bucket_for_filename(path: Path, buckets: int) -> int:
    if buckets == 1:
        return 0
    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % buckets


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path=None, device_id=0) -> None:
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.device)
        else:
            # Keeping CPU construction possible makes the scorer importable and
            # testable on a workstation.  The CLI still requires CUDA below.
            self.device = torch.device("cpu")
        self.resamplers = {}

        available_providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available_providers:
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            # Used only by direct scorer tests; main() still enforces CUDA.
            providers = ["CPUExecutionProvider"]
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
        self.p808_onnx_sess = None
        self.p808_spectrogram = None
        self.p808_mel_basis = None
        if p808_model_path is not None:
            self.p808_onnx_sess = ort.InferenceSession(
                p808_model_path,
                sess_options=session_options,
                providers=providers,
            )
            self.p808_spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=321,
                win_length=321,
                hop_length=160,
                power=2.0,
                center=True,
                pad_mode="reflect",
            ).to(self.device)
            # torchaudio's low-resolution 120-band filter bank contains an
            # empty band for this model.  Build the model-compatible Slaney
            # matrix once, then perform every per-audio operation on the GPU.
            mel_basis = librosa.filters.mel(
                sr=SAMPLING_RATE,
                n_fft=321,
                n_mels=120,
            )
            self.p808_mel_basis = torch.from_numpy(mel_basis).to(self.device)
        self.cuda_ort_enabled = (
            self.device.type == "cuda"
            and "CUDAExecutionProvider" in self.onnx_sess.get_providers()
        )
        print(f"DNSMOS ONNX providers (GPU {device_id}): {self.onnx_sess.get_providers()}")

    def audio_melspec(self, audio):
        """Calculate librosa-compatible P808 Mel features on the worker GPU."""
        power_spec = self.p808_spectrogram(audio)
        mel_spec = torch.matmul(self.p808_mel_basis, power_spec)
        amin = 1e-10
        mel_db = 10.0 * torch.log10(torch.clamp(mel_spec, min=amin))
        ref_db = 10.0 * torch.log10(
            torch.clamp(mel_spec.amax(dim=(-2, -1), keepdim=True), min=amin)
        )
        mel_db = mel_db - ref_db
        mel_db = torch.maximum(
            mel_db,
            mel_db.amax(dim=(-2, -1), keepdim=True) - 80.0,
        )
        return (mel_db + 40.0) / 40.0

    def run_onnx(self, session, input_tensor):
        """Run ONNX directly from a CUDA tensor when CUDA EP is active."""
        input_tensor = input_tensor.contiguous()
        input_name = session.get_inputs()[0].name
        if not self.cuda_ort_enabled:
            input_array = np.ascontiguousarray(input_tensor.detach().cpu().numpy())
            return session.run(None, {input_name: input_array})[0]

        io_binding = session.io_binding()
        device_id = self.device.index
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        for output in session.get_outputs():
            io_binding.bind_output(
                name=output.name,
                device_type="cuda",
                device_id=device_id,
            )
        io_binding.synchronize_inputs()
        session.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0]

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

    def prepare_clip(self, audio_item, sampling_rate):
        # Decode on the host; resampling, windowing, and P808 Mel features run
        # as batched Torch operations on the worker's GPU.
        if isinstance(audio_item, tuple):
            fpath, audio_bytes = audio_item
            audio_source = io.BytesIO(audio_bytes)
        else:
            fpath = str(audio_item)
            audio_source = fpath
        aud, input_fs = sf.read(audio_source, dtype="float32", always_2d=True)
        audio_tensor = torch.from_numpy(aud).to(self.device).transpose(0, 1)
        if input_fs != sampling_rate:
            resampler_key = (input_fs, sampling_rate)
            resampler = self.resamplers.get(resampler_key)
            if resampler is None:
                # Resample constructs and caches its sinc kernel.  Reusing one
                # GPU module per source rate avoids rebuilding that kernel for
                # every clip.
                resampler = torchaudio.transforms.Resample(
                    orig_freq=input_fs,
                    new_freq=sampling_rate,
                ).to(self.device)
                self.resamplers[resampler_key] = resampler
            audio_tensor = resampler(audio_tensor)
        # DNSMOS is mono.  Average channels after resampling to preserve the
        # behaviour of the previous implementation.
        audio = audio_tensor.mean(dim=0)
        fs = sampling_rate
        actual_audio_len = audio.numel()
        len_samples = int(INPUT_LENGTH * fs)
        if actual_audio_len == 0:
            raise ValueError("empty audio")
        while audio.numel() < len_samples:
            audio = torch.cat((audio, audio))

        hop_len_samples = fs
        input_tensor = audio.unfold(0, len_samples, hop_len_samples).contiguous()
        num_hops = input_tensor.shape[0]
        if num_hops == 0:
            raise ValueError("audio produced no valid DNSMOS windows")

        p808_input_features = None
        if self.p808_onnx_sess is not None:
            with torch.inference_mode():
                p808_input_features = self.audio_melspec(
                    input_tensor[:, :-160]
                ).transpose(1, 2).contiguous()
        input_features = input_tensor

        clip = {
            'filename': fpath,
            'len_in_sec': actual_audio_len / fs,
            'sr': fs,
            'num_hops': int(num_hops),
            'input_features': input_features,
        }
        if p808_input_features is not None:
            clip['p808_input_features'] = p808_input_features
        return clip

    def score_files(self, fpaths, sampling_rate, is_personalized_MOS, batch_size):
        clips = []
        failures = []
        input_features = []
        p808_input_features = []
        owners = []
        p808_enabled = self.p808_onnx_sess is not None

        for audio_item in fpaths:
            fpath = audio_item[0] if isinstance(audio_item, tuple) else str(audio_item)
            try:
                clip = self.prepare_clip(audio_item, sampling_rate)
            except (LibsndfileError, ValueError) as error:
                failures.append((fpath, error))
                continue

            owner = len(clips)
            clip_input_features = clip.pop('input_features')
            clip_p808_input_features = clip.pop('p808_input_features', None)
            clips.append(clip)
            input_features.append(clip_input_features)
            if clip_p808_input_features is not None:
                p808_input_features.append(clip_p808_input_features)
            owners.extend([owner] * clip_input_features.shape[0])

        if input_features:
            input_features = torch.cat(input_features, dim=0)
        else:
            input_features = torch.empty((0, int(INPUT_LENGTH * sampling_rate)), device=self.device)
        if p808_enabled and p808_input_features:
            p808_input_features = torch.cat(p808_input_features, dim=0)
        elif p808_enabled:
            p808_input_features = torch.empty((0, 900, 120), device=self.device)

        predictions = [
            {
                'SIG_raw': [],
                'BAK_raw': [],
                'OVRL_raw': [],
                'SIG': [],
                'BAK': [],
                'OVRL': [],
                **({'P808_MOS': []} if p808_enabled else {}),
            }
            for _ in clips
        ]

        for start in range(0, input_features.shape[0], batch_size):
            end = start + batch_size
            input_batch = input_features[start:end]
            p808_batch = None
            if p808_enabled:
                p808_batch = self.run_onnx(
                    self.p808_onnx_sess,
                    p808_input_features[start:end],
                ).reshape(-1)
            primary_batch = self.run_onnx(self.onnx_sess, input_batch)

            if primary_batch.ndim == 1:
                primary_batch = primary_batch.reshape(-1, 3)
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
                prediction['SIG_raw'].append(float(mos_sig_raw[index]))
                prediction['BAK_raw'].append(float(mos_bak_raw[index]))
                prediction['OVRL_raw'].append(float(mos_ovr_raw[index]))
                prediction['SIG'].append(float(mos_sig[index]))
                prediction['BAK'].append(float(mos_bak[index]))
                prediction['OVRL'].append(float(mos_ovr[index]))
                if p808_batch is not None:
                    prediction['P808_MOS'].append(float(p808_batch[index]))

        clip_dicts = []
        for clip, prediction in zip(clips, predictions):
            clip_dict = dict(clip)
            clip_dict['OVRL_raw'] = float(np.mean(prediction['OVRL_raw']))
            clip_dict['SIG_raw'] = float(np.mean(prediction['SIG_raw']))
            clip_dict['BAK_raw'] = float(np.mean(prediction['BAK_raw']))
            clip_dict['OVRL'] = float(np.mean(prediction['OVRL']))  # P835 MOS, Emilia>3.2 4.0播音员 3.5以上OK
            clip_dict['SIG'] = float(np.mean(prediction['SIG']))
            clip_dict['BAK'] = float(np.mean(prediction['BAK']))
            if p808_enabled:
                clip_dict['P808_MOS'] = float(np.mean(prediction['P808_MOS']))  # WenetSpeech4TTS Premium>4.0  Standard>3.8  Basic>3.6
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


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def listen_worker(in_queue, jsonl_file, report_interval):
    print("progress reporter started", flush=True)

    succeeded = 0
    failed = 0
    total = None
    total_audio_seconds = 0.0
    first_completion_time = None
    last_report_time = time.monotonic()
    last_report_completed = 0
    last_report_audio_seconds = 0.0

    def report_progress(now, final=False):
        nonlocal last_report_time, last_report_completed, last_report_audio_seconds
        completed = succeeded + failed
        if first_completion_time is None:
            file_rate = 0.0
            realtime_rate = 0.0
        elif final:
            elapsed = max(now - first_completion_time, 1e-9)
            file_rate = completed / elapsed
            realtime_rate = total_audio_seconds / elapsed
        else:
            interval_start = last_report_time or first_completion_time
            elapsed = max(now - interval_start, 1e-9)
            file_rate = (completed - last_report_completed) / elapsed
            realtime_rate = (
                total_audio_seconds - last_report_audio_seconds
            ) / elapsed

        if total is None:
            progress = f"{completed}/?"
            eta = "?"
        elif total == 0:
            progress = "0/0 (100.00%)"
            eta = "00:00:00"
        else:
            percentage = min(completed / total * 100.0, 100.0)
            progress = f"{completed}/{total} ({percentage:.2f}%)"
            remaining = max(total - completed, 0)
            eta = format_duration(remaining / file_rate) if file_rate > 0 else "?"

        label = "final" if final else "progress"
        print(
            f"[{label}] {progress} | {file_rate:.2f} files/s | "
            f"{realtime_rate:.1f}x realtime | audio={total_audio_seconds / 3600:.2f}h | "
            f"failed={failed} | ETA={eta}",
            flush=True,
        )
        last_report_time = now
        last_report_completed = completed
        last_report_audio_seconds = total_audio_seconds

    with open(jsonl_file, 'a', encoding='utf-8', buffering=1024 * 1024) as f:
        while True:
            try:
                item = in_queue.get(timeout=report_interval)
            except Empty:
                report_progress(time.monotonic())
                f.flush()
                continue
            if item == "STOP":
                break

            if isinstance(item, tuple) and item:
                if item[0] == READER_DONE_EVENT:
                    total = item[1]
                    continue
                if item[0] == COMPUTE_FAILURE_EVENT:
                    failed += 1
                else:
                    continue
            else:
                clip_dict = item
                total_audio_seconds += float(clip_dict.get('len_in_sec', 0.0))
                succeeded += 1
                clip_dict = {
                    key: value.item() if isinstance(value, np.generic) else value
                    for key, value in clip_dict.items()
                }
                f.write(f'{json.dumps(clip_dict)}\n')

            now = time.monotonic()
            if first_completion_time is None:
                first_completion_time = now
            if now - last_report_time >= report_interval:
                report_progress(now)
                f.flush()

        f.flush()

    completed = succeeded + failed
    if completed > last_report_completed or total is not None:
        report_progress(time.monotonic(), final=True)
    print("progress reporter stopped", flush=True)


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
        audio_item = in_queue.get()
        if audio_item == "STOP":
            break

        audio_items = [audio_item]
        while len(audio_items) < batch_size:
            try:
                audio_item = in_queue.get(timeout=0.05)
            except Empty:
                break
            if audio_item == "STOP":
                stop_requested = True
                break
            audio_items.append(audio_item)

        clip_dicts, failures = compute_score.score_files(
            audio_items,
            desired_fs,
            is_personalized_eval,
            batch_size,
        )
        for clip_dict in clip_dicts:
            out_queue.put(clip_dict)

        for failed_flac, error in failures:
            print(type(error).__name__, failed_flac, error)
            out_queue.put((COMPUTE_FAILURE_EVENT, str(failed_flac)))
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


def read_worker(
        testset_dir,
        completed_filenames,
        out_queue,
        progress_queue,
        buckets,
        bucket,
):
    """Prefetch FLAC files into RAM so compute workers do not wait on storage."""
    print(f"read_worker started, prefetching {testset_dir}")
    queued = 0
    skipped_bucket = 0
    skipped_completed = 0
    read_failures = 0

    for clip in scandir_generator(testset_dir):
        if clip.suffix != '.flac':
            continue
        if bucket_for_filename(clip, buckets) != bucket:
            skipped_bucket += 1
            continue

        filename = str(clip)
        if filename in completed_filenames:
            skipped_completed += 1
            continue

        try:
            audio_bytes = clip.read_bytes()
        except OSError as error:
            read_failures += 1
            print(type(error).__name__, filename, error)
            continue

        # Match dnsmos_mp.py: the reader performs storage I/O and places
        # encoded audio bytes in the bounded RAM queue.  GPU workers decode
        # from BytesIO without reopening the source file.
        out_queue.put((filename, audio_bytes))
        queued += 1

    print(
        f"read_worker stopped, queued={queued}, skipped_bucket={skipped_bucket}, "
        f"skipped_completed={skipped_completed}, read_failures={read_failures}"
    )
    progress_queue.put((READER_DONE_EVENT, queued))


def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    # 定义模型路径
    p808_model_path = str(script_dir / 'model_v8.onnx')
    primary_model_path = str(script_dir / ('pDNSMOS' if args.personalized_MOS else 'DNSMOS') / 'sig_bak_ovr.onnx')

    jsonl_file = Path(args.jsonl_path)
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            _mos_list = f.readlines()
        mos_list = set()
        correct_list = []
        error = 0
        for item in _mos_list:
            try:
                mos_list.add(json.loads(item.strip())['filename'])
                correct_list.append(item)
            except:
                print(item)
                error += 1
                pass
        if error > 0:
            print(f"Error: {error}")
            i = 0
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                f.writelines(correct_list)
                i += 1
                if i % 100 == 0:
                    f.flush()
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
        f"workers_per_gpu={args.workers_per_gpu}, batch_size={args.batch_size}, "
        f"prefetch_size={args.prefetch_size}"
    )

    audio_queue = Queue(maxsize=args.prefetch_size)
    done_queue = Queue()

    # Start worker processes
    listener = Process(
        target=listen_worker,
        args=(
            done_queue,
            jsonl_file,
            args.progress_interval,
        ),
    )
    listener.start()

    workers = []
    for i in range(number_of_processes):
        worker = Process(
            target=compute_worker,
            args=(
                audio_queue,
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

    reader = Process(
        target=read_worker,
        args=(
            Path(args.testset_dir).absolute(),
            mos_list,
            audio_queue,
            done_queue,
            args.buckets,
            args.bucket,
        ),
    )
    reader.start()
    reader.join()

    # Sentinels are appended only after every prefetched file, so no worker can
    # stop while unread audio remains in the queue.
    for _ in range(number_of_processes):
        audio_queue.put("STOP")

    for worker in workers:
        worker.join()
    done_queue.put("STOP")
    listener.join()

    failed_workers = [worker.pid for worker in workers if worker.exitcode != 0]
    if reader.exitcode != 0:
        raise RuntimeError(f"Read worker failed with exit code {reader.exitcode}")
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
    parser.add_argument(
        "--prefetch_size",
        type=int,
        default=10000,
        help="Maximum number of encoded FLAC files prefetched into the RAM queue.",
    )
    parser.add_argument(
        "--progress_interval",
        type=float,
        default=1.0,
        help="Seconds between progress and throughput reports.",
    )

    args = parser.parse_args()
    if args.buckets < 1:
        parser.error("--buckets must be at least 1")
    if not 0 <= args.bucket < args.buckets:
        parser.error("--bucket must be in the range [0, --buckets)")
    if args.workers_per_gpu < 1:
        parser.error("--workers_per_gpu must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.prefetch_size < 1:
        parser.error("--prefetch_size must be at least 1")
    if args.progress_interval <= 0:
        parser.error("--progress_interval must be greater than 0")

    main()
