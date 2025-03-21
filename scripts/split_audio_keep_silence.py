import librosa
import numpy as np
import os
import soundfile as sf
import argparse
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

def split_audio_remove_long_silence(input_audio, scale_factor=0.5, min_rms_threshold=0.0005, max_rms_threshold=0.01, min_silence_duration=1.2):
    """
    通过动态计算的 RMS 阈值检测静音部分，并返回语音片段。

    逻辑：
    - **去除长静音 (默认 > 1 秒)**，但保留短暂停顿，使语音自然。
    - **合并短片段，确保 `≥ 1 秒`**，避免过短的 `speech` 片段。
    - **合并后的片段如果包含 `speech`，则整个片段归 `speech`**，否则归 `silence`。
    - **最终返回所有 `speech` 片段，供外部代码批量处理**。

    参数：
    - input_audio: 输入音频文件路径
    - scale_factor: 计算动态 RMS 阈值的缩放系数
    - min_rms_threshold: 最小 RMS 阈值
    - max_rms_threshold: 最大 RMS 阈值
    - min_silence_duration: 认为是“长静音”的最小时长 (秒)，长于此时长的静音会被去掉
    - sr: 采样率，默认为 None，使用原始采样率

    返回：
    - speech_segments: 仅包含 `speech` 片段的列表，每个片段为 NumPy 数组
    - sr: 采样率，保持与原始音频一致
    """
    # 加载音频
    y, sr = librosa.load(input_audio, sr=None)
    frame_length = int(sr * 0.05)  # 50ms 窗口
    hop_length = max(1, int(sr * 0.01 // 2 * 2))  # 确保 `hop_length` 为偶数，防止 librosa 计算异常

    # 计算短时 RMS 能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 计算动态 RMS 阈值
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    dynamic_threshold = mean_rms - (std_rms * scale_factor)
    rms_threshold = max(min_rms_threshold, min(dynamic_threshold, max_rms_threshold))

    print(f"[{input_audio}] 动态计算的 RMS 阈值: {rms_threshold:.6f}")

    # 计算时间轴
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # 检测静音区域
    silent_mask = rms < rms_threshold

    # 生成分割点
    split_points = [0]
    for i in range(1, len(silent_mask)):
        if silent_mask[i] != silent_mask[i - 1]:  # 发生静音/语音切换
            split_points.append(times[i])
    split_points.append(len(y) / sr)  # 添加最后的终点

    # **去除“长静音”**
    filtered_split_points = [split_points[0]]
    for i in range(1, len(split_points) - 1, 2):  # 只检查静音部分（奇数索引）
        silence_duration = split_points[i + 1] - split_points[i]
        if silence_duration < min_silence_duration:
            filtered_split_points.extend([split_points[i], split_points[i + 1]])
        else:
            print(f"移除静音: {silence_duration:.2f}s")

    # 遍历切割点，合并短片段
    speech_segments = []
    current_segment = []
    accumulated_duration = 0
    contains_speech = False

    for i in range(len(filtered_split_points) - 1):
        start_sample = int(filtered_split_points[i] * sr)
        end_sample = int(filtered_split_points[i + 1] * sr)
        segment = y[start_sample:end_sample]
        segment_duration = (end_sample - start_sample) / sr  # 当前片段时长

        is_silence = np.all(silent_mask[int(start_sample / hop_length): int(end_sample / hop_length)])

        # **所有片段都先合并，直到总长度 ≥ 1 秒**
        current_segment.append(segment)
        accumulated_duration += segment_duration
        contains_speech = contains_speech or not is_silence  # 只要有 `speech`，整个片段就是 `speech`

        # **如果合并后片段达到最小长度，存储**
        if accumulated_duration >= min_silence_duration:
            full_segment = np.concatenate(current_segment)
            if contains_speech:
                speech_segments.append(full_segment)
            current_segment = []
            accumulated_duration = 0
            contains_speech = False  # 重置标记

    # 处理最后的片段
    if current_segment:
        full_segment = np.concatenate(current_segment)
        if contains_speech:
            speech_segments.append(full_segment)

    return speech_segments, sr  # 仅返回人声片段，供外部批量处理

def split_worker(num, task_queue, input_dir, output_dir, scale_factor, min_rms_threshold, max_rms_threshold, min_silence_duration):
    print(f"split_worker {num} started")

    for file in iter(task_queue.get, "STOP"):
        speech_segments, sr = split_audio_remove_long_silence(str(file), scale_factor, min_rms_threshold, max_rms_threshold, min_silence_duration)

        # 保存提取的 speech 片段，格式化索引为 4 位数字（0001, 0002, ...）
        for idx, segment in enumerate(speech_segments, start=1):
            output_path = Path(output_dir) / file.parent.relative_to(Path(input_dir)) / f"{file.stem}_speech_{idx:04d}.flac"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), segment, sr, format="FLAC")
            print(f"保存: {output_path}")

    print(num, 'done')

# ------------- **外部批量处理多个音频文件** -------------
def process_audio_directory(input_dir, output_dir, scale_factor, min_rms_threshold, max_rms_threshold, min_silence_duration):
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)
    # Start worker processes
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = Process(
            target=split_worker,
            args=(i, task_queue, input_dir, output_dir, scale_factor, min_rms_threshold, max_rms_threshold, min_silence_duration),
        )
        p.start()
        processes.append(p)

    for file in scandir_generator(input_dir):
        if file.suffix not in ('.flac', '.wav', '.mp3'):
            continue
        task_queue.put(file)

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # Ensure all processes finish execution
    for p in processes:
        if p.is_alive():
            p.join()

    print("split done.")


NUMBER_OF_PROCESSES = 30
# ------------- **argparse 解析参数** -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理音频文件，移除长静音并提取 speech 片段")
    parser.add_argument("--input_dir", type=str, required=True, help="输入音频文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出文件夹")
    parser.add_argument("--scale_factor", type=float, default=0.5, help="RMS 阈值缩放系数")
    parser.add_argument("--min_rms_threshold", type=float, default=0.0005, help="最小 RMS 阈值")
    parser.add_argument("--max_rms_threshold", type=float, default=0.01, help="最大 RMS 阈值")
    parser.add_argument("--min_silence_duration", type=float, default=1.2, help="认为是长静音的最小时长 (秒)")

    args = parser.parse_args()

    # 执行批量处理
    process_audio_directory(args.input_dir, args.output_dir, args.scale_factor, args.min_rms_threshold, args.max_rms_threshold, args.min_silence_duration)
