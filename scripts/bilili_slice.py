#!/usr/bin/env python3

import argparse
import time
import re
import regex

import subprocess
from pathlib import Path
from tts_norm.normalizer import Normalizer
from torch.multiprocessing import Process, Queue

normalizer_map = {}
#ffmpegExe = "/usr/local/ffmpeg/bin/ffmpeg"
#ffprobeExe = "/usr/local/ffmpeg/bin/ffprobe"
ffmpegExe = "/usr/bin/ffmpeg"
ffprobeExe = "/usr/bin/ffprobe"


def is_chinese(content):
    mobj1 = re.search('[\u4E00-\u9FA5]+', content)
    mobj2 = re.fullmatch('[\u4E00-\u9FA5a-zA-Z\s,.?!]+', content)
    return mobj1 is not None and mobj2 is not None

def text_processing(utt_txt):
    normalizer = Normalizer('zh')
    txt, unnormalize, _ = normalizer.normalize(utt_txt)
    return txt, unnormalize

def find_files(flacdir, txtdir):
    files_dict = {}
    txt_dict = {}
    for item in txtdir.glob("**/*.txt"):
        txt_dict[item.stem] = item
    for flac in flacdir.glob("**/*.flac"):
        files_dict[flac.stem] = (flac, txt_dict[flac.stem])
    return files_dict

def format_times(ts):
    ts = ts / 1000
    s = ts % 60
    ts = ts / 60
    m = ts % 60
    ts = ts / 60
    h = ts

    return "%d:%02d:%02d" % (h, m, s)

def listen_worker(in_queue, segment_file, flac_out):
    print("listen_worker started.")

    for flac, subs in iter(in_queue.get, "STOP"):
        print('listen_worker', segment_file, flac_out, flac, len(subs))
        for sub in subs:
            rec_id = sub[0]
            opath = flac_out / rec_id[0:2] / (rec_id + '.flac')
            if not opath.parent.exists():
                opath.parent.mkdir()
            cut_cmd = f'{ffmpegExe} -ss {sub[1]} -to {sub[2]} -i "{flac}" -y "{opath}"'
            subprocess.run(cut_cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=True)

        with open(segment_file,'a',encoding='utf-8') as f:
            for sub in subs:
                rec_id = sub[0]
                opath = flac_out / rec_id[0:2] / (rec_id + '.flac')
                line = f'{opath}\t{sub[3]}\t{sub[4]}\t0\n'
                f.write(line)
                f.flush()

    print("listen_worker ended.")

def align_worker(in_queue, out_queue, seg_list, num=0):
    print(f"align_worker {num} started")
    global skip_duration
    for flac, txt in iter(in_queue.get, "STOP"):
        stem = flac.stem
        with open(txt) as f:
            lines = f.readlines()
        utterance_list = []
        for line in lines:
            utterance_list.append(line.strip())
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
                    overlap_keys.add(key)
                    break
        print(f"{stem}, skip {skip_duration}s, {len(overlap_keys)} records.")
        subs = []
        for i, utt in enumerate(utterance_list):
            rec_id = f"{stem}_{i:04}"
            if rec_id in seg_list:
                continue
            utt_start, utt_end, utt_txt = utt.split("\t", 2)
            key = f'{utt_start}_{utt_end}'
            utt_start = float(utt_start)
            utt_end = float(utt_end)
            if key in overlap_keys:
                continue
            if utt_end - utt_start <= 1.0:
                #去掉1秒以下的
                continue
            # text processing
            utt_txt = utt_txt.replace('"', "")
            utt_txt = pattern_space.sub(" ", utt_txt)
            cleaned, unnormalize = text_processing(utt_txt)
            cleaned = pattern_punctuation.sub("", cleaned)
            if len(cleaned) <= 2:
                #去掉2个字或词以下的内容
                continue
            if unnormalize:
                skip_duration += utt_end - utt_start
                print(f"{stem}, skip {skip_duration}s, unnormalize {utt_txt} {cleaned}.")
                continue
            if not is_chinese(cleaned):
                skip_duration += utt_end - utt_start
                print(f"{stem}, skip {skip_duration}s, illegal character {utt_txt} {cleaned}.")
                continue

            subs.append((rec_id, utt_start, utt_end, utt_txt, cleaned))
        if len(subs) > 0:
            out_queue.put((flac,subs))

    print(f"align_worker {num} stopped")

def align(
    flacdir: Path,
    txtdir: Path,
    output: Path,
    **kwargs,
):
    if not output:
        output.mkdir()
    segment_file = output / "segments.trans.tsv"
    if segment_file.exists():
        with open(segment_file) as f:
            seg_list = f.readlines()
        seg_list = set([item.split('\t', 1)[0] for item in seg_list])
    else:
        seg_list = set()
        segment_file.touch()
    print(len(seg_list))

    # find files
    files_dict = find_files(flacdir, txtdir)
    num_files = len(files_dict)
    print(f"Found {num_files} files.")

    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)
    done_queue = Queue()

    # Start worker processes
    Process(
        target=listen_worker,
        args=(
            done_queue,
            segment_file,
            output
        ),
    ).start()

    for i in range(NUMBER_OF_PROCESSES):
        Process(target=align_worker, args=(task_queue, done_queue, seg_list, i)).start()

    # Align
    for stem in files_dict.keys():
        (flac, txt) = files_dict[stem]
        task_queue.put((flac, txt))
    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")
    while not task_queue.empty() or not done_queue.empty():
        time.sleep(20)
    done_queue.put("STOP")
    print("align done.")

def get_parser():
    parser = argparse.ArgumentParser(description="CTC segmentation")
    parser.add_argument(
        "--flacdir",
        type=Path,
        required=True,
        help="FLAC folder.",
    )
    parser.add_argument(
        "--txtdir",
        type=Path,
        required=True,
        help="Text files folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output segments directory.",
    )
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    align(**kwargs)

NUMBER_OF_PROCESSES = 1
skip_duration = 0
pattern_space = regex.compile(r'\s')
pattern_punctuation = regex.compile(r'[\p{P}\p{C}\p{S}\s]')
if __name__ == "__main__":
    main()
