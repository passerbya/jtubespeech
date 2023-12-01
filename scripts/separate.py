#!/usr/bin/python
# coding: utf-8

import sys
import shutil
import demucs.separate
import subprocess
from pathlib import Path

def delete_folder(pth) :
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
            sub.rmdir()
        else:
            sub.unlink()
    pth.rmdir()

def main():
    for lang_dir in src.glob("*"):
        if not lang_dir.is_dir():
            continue
        for srt in lang_dir.glob("**/*.txt"):
            wav_dest = Path(str(srt).replace('/txt/', '/wav/').replace('.txt', '.wav'))
            print(wav_dest)
            if wav_dest.exists():
                continue
            if not wav_dest.parent.exists():
                wav_dest.parent.mkdir(parents=True)
            outdir = src / 'temp'
            wav_src = str(srt).replace('/txt/', '/wav16k/').replace('.txt', '.wav')
            cmd = f"source /etc/profile && /root/miniconda3/bin/demucs -n htdemucs_ft --two-stems=vocals -o {outdir} {wav_src}"
            print(cmd)
            demucs.separate.main(["-n", "htdemucs_ft", "--two-stems", "vocals", "-o", str(outdir), wav_src])
            ffmpeg_exe = '/usr/local/ffmpeg/bin/ffmpeg'
            temp_path = outdir / 'htdemucs_ft' / wav_dest.stem / 'vocals.wav'
            cmd = f'{ffmpeg_exe} -i "{temp_path}" -vn -ar 16000 -ac 1 -sample_fmt s16 -y "{wav_dest}"'
            subprocess.check_output(cmd, shell=True).decode('utf-8', 'ignore')
            try:
                delete_folder(outdir / 'htdemucs_ft' / wav_dest.stem)
            except:
                pass

if __name__ == "__main__":
    src = Path('/usr/local/share/38语料/语料盘/语料/4th_biz')
    main()

