#!/usr/bin/env python3

# Copyright 2021, Ludwig Kürzinger, Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Perform CTC Re-Segmentation on japanese dataset.
Either start this program as a script or from the interactive python REPL.

# Recommended model:
# Japanese Transformer Model by Shinji (note: this model has FRAMES_PER_INDEX=768 )
asr_model_name = "Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave"
d = ModelDownloader()
model = d.download_and_unpack(asr_model_name)
# Start the program, e.g.:
align(wavdir=dir_wav, txtdir=dir_txt, output=output, ngpu=ngpu, longest_audio_segments=longest_audio_segments, **model)
"""

import regex
import argparse
import logging
import sys
import time
from typing import Union
import torch
import numpy as np

from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.utils import config_argparse
from espnet2.utils.types import str_or_none

from pathlib import Path
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_align import CTCSegmentation
from torch.multiprocessing import Process, Queue
from espnet2.utils.types import str2bool

# Language specific imports - japanese
from tts_norm.normalizer import Normalizer
import re

# NUMBER_OF_PROCESSES determines how many CTC segmentation workers
# are started. Set this higher or lower, depending how fast your
# network can do the inference and how much RAM you have
NUMBER_OF_PROCESSES = 1
normalizer_map = {}

def text_processing(utt_txt, _lang):
    """Normalize text.
    Use for Japanese text.
    Args:
        utt_txt: String of Japanese text.
        _lang: 语言代码
    Returns:
        utt_txt: Normalized
    """
    # replace some special characters
    utt_txt = utt_txt.replace('"', "")
    if _lang in normalizer_map:
        normalizer = normalizer_map[_lang]
    else:
        normalizer = Normalizer(_lang)
        normalizer_map[_lang] = normalizer
    txt, unnormalize = normalizer.normalize(utt_txt)
    # replace all the numbers
    '''
    numbers = re.findall(r"\d+\.?\d*", utt_txt)
    transcribed_numbers = [num2words(item, lang=_lang) for item in numbers]
    for nr in range(len(numbers)):
        print(utt_txt)
        old_nr = numbers[nr]
        new_nr = transcribed_numbers[nr]
        utt_txt = utt_txt.replace(old_nr, new_nr, 1)
    return utt_txt
    '''
    return txt, unnormalize

def get_partitions(
    t: int = 100000,
    max_len_s: float = 1280.0,
    fs: int = 24000,
    samples_to_frames_ratio=512,
    overlap: int = 0,
):
    """Obtain partitions

    Note that this is implemented for frontends that discard trailing data.

    Note that the partitioning strongly depends on your architecture.

    A note on audio indices:
        Based on the ratio of audio sample points to lpz indices (here called
        frame), the start index of block N is:
        0 + N * samples_to_frames_ratio
        Due to the discarded trailing data, the end is then in the range of:
        [N * samples_to_frames_ratio - 1 .. (1+N) * samples_to_frames_ratio] ???
    """
    # max length should be ~ cut length + 25%
    cut_time_s = max_len_s / 1.25
    max_length = int(max_len_s * fs)
    cut_length = int(cut_time_s * fs)
    # make sure its a multiple of frame size
    max_length -= max_length % samples_to_frames_ratio
    cut_length -= cut_length % samples_to_frames_ratio
    overlap = int(max(0, overlap))
    if (max_length - cut_length) <= samples_to_frames_ratio * (2 + overlap):
        raise ValueError(
            f"Pick a larger time value for partitions. "
            f"time value: {max_len_s}, "
            f"overlap: {overlap}, "
            f"ratio: {samples_to_frames_ratio}."
        )
    partitions = []
    duplicate_frames = []
    cumulative_lpz_length = 0
    cut_length_lpz_frames = int(cut_length // samples_to_frames_ratio)
    partition_start = 0
    while t > max_length:
        start = int(max(0, partition_start - samples_to_frames_ratio * overlap))
        end = int(
            partition_start + cut_length + samples_to_frames_ratio * (1 + overlap) - 1
        )
        partitions += [(start, end)]
        # overlap - duplicate frames shall be deleted.
        cumulative_lpz_length += cut_length_lpz_frames
        for i in range(overlap):
            duplicate_frames += [
                cumulative_lpz_length - i,
                cumulative_lpz_length + (1 + i),
            ]
        # next partition
        t -= cut_length
        partition_start += cut_length
    else:
        start = int(max(0, partition_start - samples_to_frames_ratio * overlap))
        partitions += [(start, None)]
    partition_dict = {
        "partitions": partitions,
        "overlap": overlap,
        "delete_overlap_list": duplicate_frames,
        "samples_to_frames_ratio": samples_to_frames_ratio,
        "max_length": max_length,
        "cut_length": cut_length,
        "cut_time_s": cut_time_s,
    }
    return partition_dict


def align_worker(in_queue, out_queue, num=0):
    print(f"align_worker {num} started")
    for task in iter(in_queue.get, "STOP"):
        try:
            start_time = task.segments[0][0]
            result = CTCSegmentation.get_segments(task)
            task.set(**result)
            for i in range(len(task.segments)):
                segment = task.segments[i]
                task.segments[i] = (segment[0] + start_time, segment[1] + start_time, segment[2])
            segments_str = str(task)
            out_queue.put(segments_str)
            # calculate average score
            scores = [boundary[2] for boundary in task.segments]
            if len(scores) == 0:
                logging.info(f"{task.name} has no score {segments_str}")
                print(segments_str, task.segments)
            else:
                avg = sum(scores) / len(scores)
                logging.info(f"Aligned {task.name} with avg score {avg:3.4f}")
        except (AssertionError, IndexError) as e:
            # AssertionError: Audio is shorter than ground truth
            # IndexError: backtracking not successful (e.g. audio-text mismatch)
            logging.error(
                f"Failed to align {task.utt_ids[0]} in {task.name} because of: {e}"
            )
            #import traceback
            #traceback.print_exc()
        del task
        torch.cuda.empty_cache()
    out_queue.put("STOP")
    print(f"align_worker {num} stopped")


def listen_worker(in_queue, segments="./segments.txt"):
    if segments.exists():
        model = 'a'
        print("listen_worker append started.")
    else:
        model = 'w'
        print("listen_worker write started.")
    with open(segments, model) as f:
        for item in iter(in_queue.get, "STOP"):
            if segments is None:
                print(item)
            else:
                f.write(item)
                f.flush()
    print("listen_worker ended.")


def find_files(wavdir, txtdir):
    """Search for files in given directories."""
    files_dict = {}
    dir_txt_list = list(txtdir.glob("**/*.txt"))
    for wav in wavdir.glob("**/*.wav"):
        stem = wav.stem
        txt = None
        for item in dir_txt_list:
            if item.stem == stem:
                if txt is not None:
                    raise ValueError(f"Duplicate found: {stem}")
                txt = item
        if txt is None:
            logging.error(f"No text found for {stem}.wav")
        else:
            files_dict[stem] = (wav, txt)
    return files_dict


def format_times(ts):
    ts = ts / 1000
    s = ts % 60
    ts = ts / 60
    m = ts % 60
    ts = ts / 60
    h = ts

    return "%d:%02d:%02d" % (h, m, s)

def align(
    wavdir: Path,
    txtdir: Path,
    output: Path,
    asr_train_config: Union[Path, str],
    asr_model_file: Union[Path, str] = None,
    longest_audio_segments: float = 320,
    partitions_overlap_frames: int = 30,
    log_level: Union[int, str] = "INFO",
    **kwargs,
):
    """Provide the scripting interface to score text to audio.

    longest_audio_segments:
        Size of maximum length for partitions. If an audio file
        is longer, it gets split into parts that are 75% of this value.
        The 75% was chosen to prevent empty partitions.
        This value is chosen based on the observation that Transformer-based
        models crash on audio parts longer than ~400-500 s on a computer
        with 64GB RAM

    partitions_overlap_frames:
        Additional overlap between audio segments. This number is measured
        in lpz indices. The time is calculated as:
        overlap_time [s] = frontend_frame_size / fs * OVERLAP
        Should be > 600 ms.
    """
    assert check_argument_types()
    # make sure that output is a path!
    logfile = output / "segments.log"
    segments = output / "segments.txt"
    logging.basicConfig(
        level=log_level,
        filename=str(logfile),
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if segments.exists():
        with open(segments) as f:
            seg_list = f.readlines()
        seg_list = set([item.split(' ', 1)[0] for item in seg_list])
    else:
        seg_list = set()
    print(len(seg_list))

    # Ignore configuration values that are set to None (from parser).
    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
    lang = kwargs['lang'] if 'lang' in kwargs else ''

    # Prepare CTC segmentation module
    model = {
        "asr_train_config": asr_train_config,
        "asr_model_file": asr_model_file,
    }
    print(kwargs)
    logging.info(f"Loading ASR model from {asr_model_file}")
    aligner = CTCSegmentation(
        **model, **kwargs, kaldi_style_text=True
    )
    fs = 24000
    logging.info(
        f"Zero cost transitions (gratis_blank) set to"
        f" {aligner.config.blank_transition_cost_zero}."
    )

    # Set fixed ratio for time stamps.
    # Note: This assumes that the Frontend discards trailing data.
    aligner.set_config(
        time_stamps="fixed",
    )
    # estimated index to frames ratio, usually 512, but sometimes 768
    # - depends on architecture
    samples_to_frames_ratio = int(aligner.estimate_samples_to_frames_ratio())
    # Forced fix for some issues where the ratio is not correctly determined...
    if 500 <= samples_to_frames_ratio <= 520:
        samples_to_frames_ratio = 512
    elif 750 <= samples_to_frames_ratio <= 785:
        samples_to_frames_ratio = 768
    aligner.set_config(
        samples_to_frames_ratio=samples_to_frames_ratio,
    )
    logging.info(
        f"Timing ratio (sample points per CTC index) set to"
        f" {samples_to_frames_ratio} ({aligner.time_stamps})."
    )
    logging.info(
        f"Partitioning over {longest_audio_segments}s."
        f" Overlap time: "
        f"{samples_to_frames_ratio/fs*(2*partitions_overlap_frames)}s"
        f" (overlap={partitions_overlap_frames})"
    )

    if lang == 'ja':
        ## application-specific settings
        # japanese text cleaning
        aligner.preprocess_fn.text_cleaner.cleaner_types += ["jaconv"]
    elif lang == 'vi':
        aligner.preprocess_fn.text_cleaner.cleaner_types += ["vietnamese"]
    elif lang == 'ko':
        aligner.preprocess_fn.text_cleaner.cleaner_types += ["korean_cleaner"]

    # Create queues
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)
    done_queue = Queue()

    # find files
    files_dict = find_files(wavdir, txtdir)
    num_files = len(files_dict)
    logging.info(f"Found {num_files} files.")

    # Start worker processes
    Process(
        target=listen_worker,
        args=(
            done_queue,
            segments,
        ),
    ).start()
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=align_worker, args=(task_queue, done_queue, i)).start()

    # Align
    count_files = 0
    skip_duration = 0
    for stem in files_dict.keys():
        count_files += 1
        (wav, txt) = files_dict[stem]

        # generate kaldi-style `text`
        with open(txt) as f:
            utterance_list = f.readlines()
        utterance_list = [
            item.replace("\n", "") for item in utterance_list
        ]
        overlap_keys = set()
        for i1, utt1 in enumerate(utterance_list):
            utt_start1, utt_end1, _ = utt1.split("\t", 2)
            key = f'{utt_start1}_{utt_end1}'
            utt_start1 = float(utt_start1)
            utt_end1 = float(utt_end1)
            for i2, utt2 in enumerate(utterance_list):
                if i1 == i2:
                    continue
                utt_start2, utt_end2, _ = utt2.split("\t", 2)
                utt_start2 = float(utt_start2)
                utt_end2 = float(utt_end2)
                if max(utt_start1, utt_start2) < min(utt_end1, utt_end2):
                    skip_duration += utt_end1 - utt_start1
                    #print(stem, utt1, utt2, format_times(int(skip_duration*1000)))
                    overlap_keys.add(key)
                    break
        text = []
        timestamps = []
        for i, utt in enumerate(utterance_list):
            if f"{stem}_{i:04}" in seg_list:
                continue
            utt_start, utt_end, utt_txt = utt.split("\t", 2)
            key = f'{utt_start}_{utt_end}'
            if float(utt_end) - float(utt_start) <= 0 or key in overlap_keys:
                continue
            # text processing
            utt_txt, unnormalize = text_processing(utt_txt, lang)
            if unnormalize:
                print(txt, utt)
                continue
            cleaned = aligner.preprocess_fn.text_cleaner(utt_txt)
            text.append(f"{stem}_{i:04} {cleaned}")
            timestamps.append((utt_start, utt_end))

        if len(timestamps) == 0:
            continue
        # audio
        _speech, sample_rate = soundfile.read(wav)
        speech_len = _speech.shape[0]
        speech = torch.tensor(_speech)
        partitions = get_partitions(
            speech_len,
            max_len_s=longest_audio_segments,
            samples_to_frames_ratio=samples_to_frames_ratio,
            fs=fs,
            overlap=partitions_overlap_frames,
        )
        duration = speech_len / sample_rate
        # CAVEAT Assumption: Frontend discards trailing data:
        expected_lpz_length = (speech_len // samples_to_frames_ratio) - 1

        logging.info(
            f"Inference on file {stem} {count_files}/{num_files}: {len(utterance_list)}"
            f" utterances:  ({duration}s ~{len(partitions['partitions'])}p)"
        )
        try:
            # infer
            lpzs = [
                torch.tensor(aligner.get_lpz(speech[start:end]))
                for start, end in partitions["partitions"]
            ]
            lpz = torch.cat(lpzs).numpy()
            lpz = np.delete(lpz, partitions["delete_overlap_list"], axis=0)
            if lpz.shape[0] != expected_lpz_length and lpz.shape[0] != (
                expected_lpz_length + 1
            ):
                # The one-off error fix is a little bit dirty,
                # but it helps to deal with different frontend configurations
                logging.error(
                    f"LPZ size mismatch on {stem}: "
                    f"got {lpz.shape[0]}-{expected_lpz_length} expected."
                )
            task = aligner.prepare_segmentation_task(
                text, lpz, name=stem, speech_len=speech_len
            )
            task.segments = [(0, duration, 0)]
            # align (done by worker)
            task_queue.put(task)
        except KeyboardInterrupt:
            print(" -- Received keyboard interrupt. Stopping.")
            break
        except Exception as e:
            # RuntimeError: unknown CUDA value error (at inference)
            # TooShortUttError: Audio too short (at inference)
            # IndexError: ground truth is empty (thrown at preparation)
            #logging.error(f"LPZ failed for file {stem}; {e.__class__}: {e}")
            start = 0
            step = 32
            while True:
                end = start+step if len(timestamps)>start+step else len(timestamps)
                print(wav, start, end)
                timestamp_slice = timestamps[start:end]
                text_slice = text[start:end]
                start_time = float(timestamp_slice[0][0]) - 10
                end_time = float(timestamp_slice[-1][-1]) + 10
                if start_time < 0:
                    start_time = 0
                if end_time > float(timestamps[-1][-1]):
                    end_time = float(timestamps[-1][-1])
                _speech_slice = _speech[int(start_time * sample_rate):int(end_time * sample_rate)]
                speech_len = _speech_slice.shape[0]
                speech_slice = torch.tensor(_speech_slice)
                partitions = get_partitions(
                    speech_len,
                    max_len_s=longest_audio_segments,
                    samples_to_frames_ratio=samples_to_frames_ratio,
                    fs=fs,
                    overlap=partitions_overlap_frames,
                )
                duration = speech_len / sample_rate
                # CAVEAT Assumption: Frontend discards trailing data:
                expected_lpz_length = (speech_len // samples_to_frames_ratio) - 1

                logging.info(
                    f"Inference on file {stem} {count_files}/{num_files}: {len(utterance_list)}"
                    f" utterances:  ({duration}s ~{len(partitions['partitions'])}p)"
                )
                try:
                    # infer
                    lpzs = [
                        torch.tensor(aligner.get_lpz(speech_slice[start:end]))
                        for start, end in partitions["partitions"]
                    ]
                    lpz = torch.cat(lpzs).numpy()
                    lpz = np.delete(lpz, partitions["delete_overlap_list"], axis=0)
                    if lpz.shape[0] != expected_lpz_length and lpz.shape[0] != (
                            expected_lpz_length + 1
                    ):
                        # The one-off error fix is a little bit dirty,
                        # but it helps to deal with different frontend configurations
                        logging.error(
                            f"LPZ size mismatch on {stem}: "
                            f"got {lpz.shape[0]}-{expected_lpz_length} expected."
                        )
                    task = aligner.prepare_segmentation_task(
                        text_slice, lpz, name=stem, speech_len=speech_len
                    )
                    task.segments = [(start_time, end_time, 0)]
                    # align (done by worker)
                    task_queue.put(task)
                except KeyboardInterrupt:
                    print(" -- Received keyboard interrupt. Stopping.")
                    break
                except Exception as e:
                    # RuntimeError: unknown CUDA value error (at inference)
                    # TooShortUttError: Audio too short (at inference)
                    # IndexError: ground truth is empty (thrown at preparation)
                    logging.error(f"LPZ failed for file {stem}; {e.__class__}: {e}")
                    step = int(step // 2)
                    if step == 0:
                        break
                    continue
                start = end
                if end >= len(timestamps):
                    break

    logging.info("Shutting down workers.")
    # wait for workers to finish
    time.sleep(20)
    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")


def get_parser():
    """Obtain an argument-parser for the script interface."""
    parser = config_argparse.ArgumentParser(
        description="CTC segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--use-dict-blank",
        type=int,
        default=None,
        help="DEPRECATED.",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    group = parser.add_argument_group("CTC segmentation related")
    group.add_argument(
        "--fs",
        type=int,
        default=24000,
        help="Sampling Frequency."
        " The sampling frequency (in Hz) is needed to correctly determine the"
        " starting and ending time of aligned segments.",
    )
    group.add_argument(
        "--gratis_blank",
        type=str2bool,
        default=True,
        help="Set the transition cost of the blank token to zero. Audio sections"
        " labeled with blank tokens can then be skipped without penalty. Useful"
        " if there are unrelated audio segments between utterances.",
    )

    group.add_argument(
        "--longest_audio_segments",
        type=int,
        default=320,
        help="Inference on very long audio files requires much memory."
        " To avoid out-of-memory errors, long audio files can be partitioned."
        " Set this value to the maximum unpartitioned audio length.",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)

    group = parser.add_argument_group("Input/output arguments")
    group.add_argument(
        "--wavdir",
        type=Path,
        required=True,
        help="WAV folder.",
    )
    group.add_argument(
        "--txtdir",
        type=Path,
        required=True,
        help="Text files folder.",
    )
    group.add_argument(
        "--output",
        type=Path,
        help="Output segments directory.",
    )
    group.add_argument(
        "--lang",
        type=str_or_none,
    )
    return parser


def main(cmd=None):
    """Parse arguments and start."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    align(**kwargs)


if __name__ == "__main__":
    main()
