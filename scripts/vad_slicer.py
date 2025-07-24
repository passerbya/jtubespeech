
#!/usr/bin/env python3
# coding: utf-8

import torch
import argparse
import soundfile as sf
from pathlib import Path
from torch.multiprocessing import Process, Queue

def detect_and_slice(device, flac_path, output_path, model, get_speech_timestamps, read_audio):
    wav = read_audio(str(flac_path), sampling_rate=16000).to(device)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    if not speech_timestamps:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, segment in enumerate(speech_timestamps):
        start = segment['start']
        end = segment['end']
        audio_chunk = wav[start:end]
        chunk_path = output_path.with_suffix(f'.seg{idx:03d}.flac')
        sf.write(str(chunk_path), audio_chunk.cpu(), samplerate=16000)
    return True

def worker(root_dir, output_root, queue, worker_id):
    device_count = torch.cuda.device_count()
    device_id = worker_id % device_count if device_count > 0 else 0
    device = f"cuda:{device_id}" if device_count > 0 else "cpu"

    print(f"Worker {worker_id} using device {device}")

    # Load silero-vad model
    device = torch.device(device)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    get_speech_timestamps, _, read_audio, _, _ = utils
    model.to(device)

    for flac_path in iter(queue.get, "STOP"):
        relative = flac_path.relative_to(root_dir)
        output_path = output_root / relative
        if output_path.with_suffix('.seg000.flac').exists():
            continue
        ok = detect_and_slice(device, flac_path, output_path, model, get_speech_timestamps, read_audio)
        print(f"[{'OK' if ok else 'SKIP'}] {flac_path}")

    print(f"Worker {worker_id} done.")

def main():
    args = parser.parse_args()
    root_dir = Path(args.dir).resolve()
    output_root = root_dir / "speech_sep"

    with open(args.scp, 'r', encoding='utf-8') as f:
        allow_list = set(Path(line.strip()) for line in f if line.strip())

    task_queue = Queue(maxsize=NUM_WORKERS)
    processes = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker, args=(root_dir, output_root, task_queue, i))
        p.start()
        processes.append(p)

    for flac_path in allow_list:
        task_queue.put(flac_path)

    for _ in range(NUM_WORKERS):
        task_queue.put("STOP")

    for p in processes:
        p.join()

    print("🎉 All done.")

NUM_WORKERS = 3*torch.cuda.device_count()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Silero VAD Speech Slicer with Torch Hub")
    parser.add_argument("--dir", type=str, required=True, help="Root directory containing .flac files")
    parser.add_argument("--scp", type=str, required=True, help="Path to .scp file (must list all allowed files)")
    main()
