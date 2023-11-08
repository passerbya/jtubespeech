#!/usr/bin/python
# coding: utf-8

import argparse
import sys
import subprocess
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate background music from vocals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wavdir", type=str, help='WAV folder.')
    parser.add_argument("--outdir", type=str, help='dirname to save vocals.')
    args = parser.parse_args(sys.argv[1:])
    wavdir = Path(args.wavdir)
    for wav in wavdir.glob("**/*.wav"):
        destination = Path(args.outdir) / Path(wav.parent).name / wav.name
        outdir = Path(Path(args.outdir).parent) / 'htdemucs_ft' / wav.stem
        cmd = f"source /etc/profile && /root/miniconda3/bin/demucs -n htdemucs_ft --two-stems=vocals -o {Path(args.outdir).parent} {wav}"
        print(cmd)
        subprocess.check_output(cmd, shell=True).decode('utf-8', 'ignore')
        shutil.move(outdir / 'vocals.wav', destination)
        shutil.rmtree(outdir)
        print(destination)
        print(outdir / 'vocals.wav')
        print(outdir)

