#!/usr/bin/python
# coding: utf-8
import tempfile
import subprocess
import random
import shutil
from pathlib import Path

if __name__ == "__main__":
    ffmpegExe = "/usr/local/ffmpeg/bin/ffmpeg"
    src = Path('/usr/local/corpus/4th_biz/en')
    sample = src / 'sample'
    if sample.exists():
        shutil.rmtree(sample)
    sample.mkdir()
    segment_file = src / 'segments' / 'segments.trans.half-width.tsv'
    with open(segment_file) as f:
        seg_list = f.readlines()
        for seg in seg_list:
            uid, _, _, _ = seg.split("\t", 3)
            wav_path = src / 'segments' / uid[0:2] / f'{uid}.wav'
            if wav_path.exists():
                continue
            print(wav_path)
            md5 = uid[0:uid.rfind('_')]
            wav = src / 'wav' / md5[0:2] / (md5+'.wav')
            wav24k = src / 'wav' / md5[0:2] / (md5+'_24k.wav')
            if not wav24k.exists():
                with tempfile.TemporaryDirectory() as temp_dir_path:
                    temp_path = Path(temp_dir_path) / wav24k.name
                    cmd = f'{ffmpegExe} -i "{wav}" -vn -ar 24000 -ac 1 -sample_fmt s16 -y "{temp_path}"'
                    subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, shell=True)
                    shutil.move(temp_path, wav24k)
            txt_path = src / 'txt' / md5[0:2] / f'{md5}.txt'
            with open(txt_path) as ft:
                lines = ft.readlines()
            for i, utt in enumerate(lines):
                rec_id = f"{md5}_{i:04}"
                if uid != rec_id:
                    continue
                utt_start, utt_end, _ = utt.split("\t", 2)
                if not wav_path.parent.exists():
                    wav_path.parent.mkdir()
                cut_cmd = f'{ffmpegExe} -ss {utt_start} -to {utt_end} -i "{wav24k}" -y "{wav_path}"'
                print(cut_cmd)
                subprocess.run(cut_cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=True)

    random_samples = random.sample(seg_list, 10)
    for random_sample in random_samples:
        uid, _, txt, _ = random_sample.split("\t", 3)
        wav_path = src / 'segments' / uid[0:2] / f'{uid}.wav'
        if (sample/wav_path.name).exists():
            (sample/wav_path.name).unlink()
        shutil.copy(wav_path, sample/wav_path.name)
        with open(sample/f'{uid}.txt', 'w', encoding='utf-8') as f:
            f.write(txt)
            f.flush()
print('done')