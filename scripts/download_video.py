import os
import time
import argparse
import sys
import subprocess
import shutil
#import pydub
import glob
import random
from pathlib import Path
from util import make_video_url, make_basename, vtt2txt, autovtt2txt
import pandas as pd
from tqdm import tqdm
from torch.multiprocessing import Process, Queue

def parse_args():
  parser = argparse.ArgumentParser(
    description="Downloading videos with subtitle.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("lang",         type=str, help="language code (ja, en, ...)")
  parser.add_argument("sublist",      type=str, help="filename of list of video IDs with subtitles")  
  parser.add_argument("--outdir",     type=str, default="video", help="dirname to save videos")
  parser.add_argument("--proxies",    type=str, nargs='+', default="192.168.8.23:7890 192.168.8.123:7890 192.168.8.25:7890")
  parser.add_argument("--keeporg",    action='store_true', default=False, help="keep original audio file.")
  return parser.parse_args(sys.argv[1:])

def download_worker(proxy, lang, task_queue, error_queue, empty_queue, wait_sec, keep_org):
  r = str(round(time.time()*1000)) + '_' + str(random.randint(10000000, 999999999))
  cookie_file = f'cookies_{r}.txt'
  shutil.copy('cookies.txt', cookie_file)
  for videoid, fn in iter(task_queue.get, "STOP"):
    url = make_video_url(videoid)
    base = fn["wav"].parent.joinpath(fn["wav"].stem)
    cmd = f"export http_proxy=http://{proxy} && export https_proxy=http://{proxy} && yt-dlp -v --match-filter \"duration < 7200\" --cookies {cookie_file} --sub-langs \"{lang}.*\" --skip-download --write-sub {url} -o {base}.\%\(ext\)s"
    print(cmd)
    cp = subprocess.run(cmd, shell=True, universal_newlines=True, capture_output=True, text=True)
    if cp.returncode != 0:
      print(f"Failed to download the video: url = {url}")
      for f in glob.glob(f"{base}.{lang}*.vtt"):
        os.remove(f)
      if ('ERROR: [youtube]' in cp.stdout and ('Video unavailable' in cp.stdout or 'This video is unavailable' in cp.stdout or 'Private video' in cp.stdout)) \
              or ('ERROR: [youtube]' in cp.stderr and ('Video unavailable' in cp.stderr or 'This video is unavailable' in cp.stderr or 'Private video' in cp.stderr)):
        error_queue.put(videoid)
      continue
    try:
      f = glob.glob(f"{base}.{lang}*.vtt")[0]
      shutil.move(f, fn["vtt"])
    except Exception as e:
      print(f"Failed to rename subtitle file. The download may have failed: url = {url}, filename = {base}.{lang}.vtt, error = {e}")
      continue

    # vtt -> txt (reformatting)
    try:
      with open(fn["vtt"], "r") as f:
        txt = vtt2txt(f.readlines())
      if len(txt) == 0:
        empty_queue.put(videoid)
        fn["vtt"].unlink()
        continue
      with open(fn["txt"], "w") as f:
        f.writelines([f"{t[0]:1.3f}\t{t[1]:1.3f}\t\"{t[2]}\"\n" for t in txt])
    except Exception as e:
      print(f"Falied to convert subtitle file to txt file: url = {url}, filename = {fn['vtt']}, error = {e}")
      continue

    cmd = f"export http_proxy=http://{proxy} && export https_proxy=http://{proxy} && yt-dlp -v --match-filter \"duration < 7200\" --cookies {cookie_file} --sub-langs \"{lang}.*\" --extract-audio --audio-format wav {url} -o {base}.\%\(ext\)s"
    print(cmd)
    cp = subprocess.run(cmd, shell=True, universal_newlines=True, capture_output=True, text=True)
    if cp.returncode != 0 or not fn["wav"].exists():
      print(f"Failed to download the video: url = {url}")
      continue

    # wav -> wav16k (resampling to 16kHz, 1ch)
    try:
      subprocess.run("ffmpeg -i {} -ac 1 -y {}".format(fn["wav"], fn["wav_org"]), shell=True, universal_newlines=True)
      #shutil.move(fn["wav"], fn["wav_org"])
      '''
      wav = pydub.AudioSegment.from_file(fn["wav"], format = "wav")
      wav = pydub.effects.normalize(wav, 5.0).set_frame_rate(16000).set_channels(1)
      wav.export(fn["wav16k"], format="wav", bitrate="16k")
      '''
    except Exception as e:
      print(f"Failed to normalize or resample downloaded audio: url = {url}, filename = {fn['wav']}, error = {e}")
      continue

    # remove original wav
    if not keep_org:
      fn["wav"].unlink()

    # wait
    if wait_sec > 0.01:
      time.sleep(wait_sec)

  os.remove(cookie_file)
  print(proxy, 'done')
  error_queue.put('STOP')
  empty_queue.put('STOP')

def save_error_worker(error_fn, in_queue):
  with open(str(error_fn), "w") as f:
    for videoid in iter(in_queue.get, "STOP"):
      f.write(videoid+'\n')
      f.flush()
  print(f'save {error_fn} done')

def download_video(lang, fn_sub, proxies, outdir="video", wait_sec=10, keep_org=False):
  """
  Tips:
    If you want to download automatic subtitles instead of manual subtitles, please change as follows.
      1. replace "sub[sub["sub"]==True]" of for-loop with "sub[sub["auto"]==True]"
      2. replace "--write-sub" option of yt-dlp with "--write-auto-sub"
      3. replace vtt2txt() with autovtt2txt()
      4 (optional). change fn["vtt"] (path to save subtitle) to another. 
  """

  sub = pd.read_csv(fn_sub)
  task_queue = Queue(maxsize=len(proxies))

  error_queue = Queue()
  error_fn = Path(f'videoid/error/{lang}wiki-latest-pages-articles-multistream-index.txt')
  if not error_fn.exists():
    error_fn.parent.mkdir(parents=True, exist_ok=True)
    error_fn.touch()
  error_vids = set()
  with open(str(error_fn), "r") as f:
    for line in f.readlines():
      vid = line.strip()
      error_queue.put(vid)
      error_vids.add(vid)

  empty_queue = Queue()
  empty_fn = Path(f'videoid/empty/{lang}wiki-latest-pages-articles-multistream-index.txt')
  if not empty_fn.exists():
    empty_fn.parent.mkdir(parents=True, exist_ok=True)
    empty_fn.touch()
  empty_vids = set()
  with open(str(empty_fn), "r") as f:
    for line in f.readlines():
      vid = line.strip()
      empty_queue.put(vid)
      empty_vids.add(vid)

  # Start worker processes
  for proxy in proxies:
    Process(
      target=download_worker,
      args=(
        proxy, lang, task_queue, error_queue, empty_queue, wait_sec, keep_org
      ),
    ).start()
  Process(
    target=save_error_worker,
    args=(
      error_fn, error_queue
    ),
  ).start()
  Process(
    target=save_error_worker,
    args=(
      empty_fn, empty_queue
    ),
  ).start()
  for videoid in tqdm(sub[sub["sub"]==True]["videoid"]): # manual subtitle only
    if videoid in empty_vids or videoid in error_vids:
      continue
    fn = {}
    for k in ["wav", "wav_org", "vtt", "txt"]:
      if k == 'wav_org':
        fn[k] = Path(outdir) / lang / k / (make_basename(videoid) + ".flac")
      else:
        fn[k] = Path(outdir) / lang / k / (make_basename(videoid) + "." + k[:3])
      fn[k].parent.mkdir(parents=True, exist_ok=True)
    if fn["wav_org"].exists() and fn["txt"].exists():
      continue

    # download
    task_queue.put((videoid, fn))

  for _ in proxies:
    task_queue.put('STOP')

  return Path(outdir) / lang

if __name__ == "__main__":
  args = parse_args()

  dirname = download_video(args.lang, args.sublist, args.proxies, args.outdir, keep_org=args.keeporg)
  print(f"save {args.lang.upper()} videos to {dirname}.")

