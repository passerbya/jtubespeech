#!/usr/bin/env python3

import argparse
import time
import torch
import numpy as np
import shutil
import re
import regex

import tempfile
import subprocess
import soundfile
import ctc_segmentation
import pykakasi
from pathlib import Path
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
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

def is_japanese(content):
    mobj = re.fullmatch('[\u3040-\u309F\u30A0-\u30FF\s,.?!]+', content)
    return mobj is not None

def is_korean(content):
    mobj = re.fullmatch('[\uAC00-\uD7A3\s,.?!]+', content)
    return mobj is not None

def is_english(content):
    mobj = re.fullmatch('[a-zA-Z0-9\-_\'\s,.?!]+', content)
    return mobj is not None

def text_processing(utt_txt, _lang):
    if _lang in normalizer_map:
        normalizer = normalizer_map[_lang]
    else:
        normalizer = Normalizer(_lang)
        normalizer_map[_lang] = normalizer
    txt, unnormalize, _ = normalizer.normalize(utt_txt)
    return txt, unnormalize

def find_files(flacdir, txtdir):
    files_dict = {}
    txt_dict = {}
    for item in txtdir.glob("**/*.txt"):
        txt_dict[item.stem] = item
    for flac in flacdir.glob("**/*.flac"):
        if flac.stem.endswith('_16k'):
            continue
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

def align_worker(in_queue, out_queue, lang, seg_list, num=0):
    print(f"align_worker {num} started")
    global skip_duration
    batch_size = 1
    device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")
    print(device)
    print('loading ...')
    if lang == 'zh':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
    elif lang == 'en':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-english"
    elif lang == 'ja':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-japanese"
    elif lang == 'ko':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-large-mms-1b-korean-colab_v0"
    elif lang == 'vi':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-53-vietnamese"
    elif lang == 'th':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-th-cv11_0"
    elif lang == 'ru':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xlsr-1b-ru"
    elif lang == 'es':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-es"
    elif lang == 'fr':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-french"
    elif lang == 'de':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-de-with-lm"
    elif lang == 'pt':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-pt-cv7-from-bp400h"
    elif lang == 'ar':
        model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-ar"
    else:
        model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-english"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    if "<unk>" in vocab:
        unk_id = vocab["<unk>"]
    elif "<UNK>" in vocab:
        unk_id = vocab["<UNK>"]
    elif "[unk]" in vocab:
        unk_id = vocab["[unk]"]
    elif "[UNK]" in vocab:
        unk_id = vocab["[UNK]"]
    print('load done')
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
        unm_transcripts = []
        transcripts = []
        cleaned_texts = []
        timestamps = []
        rec_ids = []
        kks = pykakasi.kakasi()
        for i, utt in enumerate(utterance_list):
            rec_id = f"{stem}_{i:04}"
            if rec_id in seg_list:
                continue
            utt_start, utt_end, utt_txt = utt.split("\t", 2)
            key = f'{utt_start}_{utt_end}'
            if float(utt_end) <= float(utt_start) or key in overlap_keys:
                continue
            # text processing
            utt_txt = utt_txt.replace('"', "")
            utt_txt = pattern_space.sub(" ", utt_txt)
            cleaned, unnormalize = text_processing(utt_txt, lang)
            if unnormalize:
                skip_duration += float(utt_end) - float(utt_start)
                print(f"{stem}, skip {skip_duration}s, unnormalize {utt_txt} {cleaned}.")
                continue
            if lang == 'zh' and not is_chinese(cleaned):
                skip_duration += float(utt_end) - float(utt_start)
                print(f"{stem}, skip {skip_duration}s, illegal character {utt_txt} {cleaned}.")
                continue
            if lang == 'ja' and not (is_japanese(cleaned) or is_chinese(cleaned)):
                skip_duration += float(utt_end) - float(utt_start)
                print(f"{stem}, skip {skip_duration}s, illegal character {utt_txt} {cleaned}.")
                continue
            if lang == 'ko' and not is_korean(cleaned):
                skip_duration += float(utt_end) - float(utt_start)
                print(f"{stem}, skip {skip_duration}s, illegal character {utt_txt} {cleaned}.")
                continue
            if lang == 'en' and not is_english(cleaned):
                skip_duration += float(utt_end) - float(utt_start)
                print(f"{stem}, skip {skip_duration}s, illegal character {utt_txt} {cleaned}.")
                continue
            if lang == 'ja':
                result = kks.convert(cleaned)
                if result is not None and len(result) >0:
                    spell = ''
                    sep = ''
                    for item in result:
                        hira = item['hira']
                        #hira = item['kana']
                        spell += sep + hira
                        sep  = ' '
                else:
                    spell = text
                transcripts.append(spell)
            else:
                transcripts.append(cleaned)
            cleaned_texts.append(cleaned)
            unm_transcripts.append(utt_txt)
            rec_ids.append(rec_id)
            timestamps.append((float(utt_start), float(utt_end)))

        if len(timestamps) == 0:
            continue
        flac16k = flac.parent / (flac.stem+'_16k.flac')
        if not flac16k.exists():
            with tempfile.TemporaryDirectory() as temp_dir_path:
                temp_path = Path(temp_dir_path) / flac16k.name
                cmd = f'{ffmpegExe} -i "{flac}" -vn -ar 16000 -ac 1 -sample_fmt s16 -y "{temp_path}"'
                subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, shell=True)
                shutil.move(temp_path, flac16k)
        audio, sample_rate = soundfile.read(flac16k)

        # Run prediction, get logits and probabilities
        start = end = 0
        total = 0
        accuracy = 0
        subs = []
        offset_second = 0.1
        while True:
            try:
                start_time = timestamps[start][0]
                end_time = timestamps[start][0]
                end = start
                while end < len(timestamps):
                    end = end + 1
                    end_time = timestamps[end-1][0]
                    if end_time - start_time > batch_size*60:
                        break
                if start > 0:
                    if end_time - timestamps[start-1][0] <= (batch_size+1)*60:
                        offset = 1
                        start -= offset
                        start_time = timestamps[start][0]
                    else:
                        offset = 0
                else:
                    offset = 0
                if end_time - start_time > (batch_size+1)*60:
                    if end-1 > start:
                        end -= 1
                        end_time = timestamps[end-1][0]
                    else:
                        if start >= len(timestamps)-1:
                            #忽略并中止
                            print('-'*10, 'ignore and stop')
                            break
                        #忽略本条字幕
                        print('='*10, 'ignore', start, end, transcripts[start:end], end_time - start_time)
                        start = end
                        continue

                timestamp_slice = timestamps[start:end]
                transcripts_slice = transcripts[start:end]
                cleaned_texts_slice = cleaned_texts[start:end]
                unm_transcripts_slice = unm_transcripts[start:end]
                rec_ids_slice = rec_ids[start:end]
                start_time -= offset_second
                end_time += offset_second
                #print(start, end, end_time - start_time)
                start_idx = int(start_time * sample_rate)
                end_idx = int(end_time * sample_rate)
                if start_idx < 0:
                    start_idx = 0
                    start_time = 0
                if end_idx > len(audio):
                    end_idx = len(audio)
                audio_slice = audio[start_idx:end_idx]
                inputs = processor(audio_slice, sampling_rate=sample_rate, return_tensors="pt", padding="longest")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    #logits = model(inputs['input_values'].to(device)).logits.cpu()[0]
                    logits = model(**inputs).logits.cpu()[0]
                    probs = torch.nn.functional.softmax(logits,dim=-1)

                # Tokenize transcripts
                tokens = []
                texts = []
                indexes = []
                for i, transcript in enumerate(transcripts_slice):
                    assert len(transcript) > 0
                    tok_ids = tokenizer(transcript.replace("\n"," ").lower())['input_ids']
                    tok_ids = np.array(tok_ids,dtype=np.int64)
                    know_ids = tok_ids[tok_ids != unk_id]
                    if len(know_ids) != 0:
                        tokens.append(know_ids)
                        texts.append(transcript)
                        indexes.append(i)

                # Align
                char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
                config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
                config.index_duration = audio_slice.shape[0] / probs.size()[0] / sample_rate

                ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
                timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
                #print(len(tokens), len(texts), len(ground_truth_mat), len(utt_begin_indices), utt_begin_indices, len(timings))
                segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, texts)
                for i, t, p in zip(indexes, texts, segments):
                    if i < offset:
                        continue
                    utt_start, utt_end = timestamp_slice[i]
                    stime = p[0] + start_time
                    etime = p[1] + start_time
                    if p[2] > 1.5:
                        total += 1
                        skip_duration += utt_end - utt_start
                        print("low score", f'{stem}, skip {skip_duration}s', {"start": stime, "end": etime, "conf": p[2], "text": t})
                        continue
                    total += 1
                    diff1 = abs(utt_start - stime)
                    diff2 = abs(utt_end - etime)
                    if diff1 > 1 or diff2 > 1:
                        skip_duration += utt_end - utt_start
                        print("large deviation", f'{stem}, skip {skip_duration}s', diff1, diff2, t)
                    else:
                        if utt_end - utt_start > 1.0:
                            #去掉1秒以下，2个字或词以下的内容
                            if lang in ('zh', 'ja', 'th'):
                                utt_txt = pattern_punctuation.sub("", cleaned_texts_slice[i])
                                if len(utt_txt) > 2:
                                    accuracy += 1
                                    subs.append((rec_ids_slice[i], utt_start, utt_end, unm_transcripts_slice[i], cleaned_texts_slice[i]))
                            else:
                                if cleaned_texts_slice[i].count(' ') > 1:
                                    accuracy += 1
                                    subs.append((rec_ids_slice[i], utt_start, utt_end, unm_transcripts_slice[i], cleaned_texts_slice[i]))
                offset_second = 0.1
            except AssertionError:
                offset_second = 0.5
            except Exception:
                print(txt, unm_transcripts_slice, timestamp_slice)
                import traceback
                traceback.print_exc()
            start = end
            if end >= len(timestamps):
                break
        if total == 0:
            print('skip:', stem, accuracy, total)
        elif accuracy/total < 0.7:
            print('skip:', stem, accuracy/total)
        else:
            print('pass:', stem, accuracy/total)
            out_queue.put((flac,subs))
    print(f"align_worker {num} stopped")

def align(
    flacdir: Path,
    txtdir: Path,
    output: Path,
    lang: str = 'en',
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
        Process(target=align_worker, args=(task_queue, done_queue, lang, seg_list, i)).start()

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
    parser.add_argument(
        "--lang",
        default='en',
        type=str,
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
