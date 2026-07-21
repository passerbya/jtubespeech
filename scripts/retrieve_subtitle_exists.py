import os
import json
import time
import requests
import argparse
import re
import sys
import traceback
import subprocess
import shutil
import random
from pathlib import Path
from util import make_video_url
import pandas as pd
from tqdm import tqdm
from torch.multiprocessing import Process, Queue


def language_matches(candidate_lang, requested_lang):
  """Match exact and regional language tags, such as en and en-US."""
  if not candidate_lang:
    return False
  candidate = str(candidate_lang).lower().replace('_', '-')
  requested = requested_lang.lower().replace('_', '-')
  return (
    candidate == requested
    or candidate.startswith(requested + '-')
    or requested.startswith(candidate + '-')
  )


def has_language(lang_codes, requested_lang):
  return any(language_matches(code, requested_lang) for code in lang_codes)


def has_audio_only_format(formats, requested_lang):
  return any(
    fmt.get('vcodec') == 'none'
    and fmt.get('acodec') not in (None, 'none')
    and language_matches(fmt.get('language'), requested_lang)
    for fmt in (formats or [])
  )

def parse_args():
  parser = argparse.ArgumentParser(
    description="Retrieving whether subtitles exists or not.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("lang",         type=str, help="language code (ja, en, ...)")
  parser.add_argument("videoidlist",  type=str, help="filename of video ID list")  
  parser.add_argument("--outdir",     type=str, default="sub", help="dirname to save results")
  parser.add_argument("--checkpoint", type=str, default=None, help="filename of list checkpoint (for restart retrieving)")
  parser.add_argument("--proxies",    type=str, nargs='+', default="192.168.8.23:7890 192.168.8.123:7890 192.168.8.25:7890")
  return parser.parse_args(sys.argv[1:])

def retrieve_worker(proxy, lang, in_queue, out_queue, error_queue, empty_queue, wait_sec):
  for videoid in iter(in_queue.get, "STOP"):
    # sleep
    if wait_sec > 0.01:
      time.sleep(wait_sec)
    url = make_video_url(videoid)
    try:
      r = str(round(time.time()*1000)) + '_' + str(random.randint(10000000, 999999999))
      cookie_file = f'cookies_{r}.txt'
      shutil.copy('cookies.txt', cookie_file)
      po_token = 'MlPA_YR3HhR4wsDBBnSs4Kb5qjFJHmEIvJ_--oUBgYqmHeBtnnqr22Iz6EzvvK49vIwWPeXyqr_dvFl-ZQ1h9J-Pj65pDyjsiU-NqsL95oE5s5Cllg=='
      cmd = [
        'yt-dlp', '-J', '--skip-download',
        '--proxy', f'http://{proxy}',
        '--cookies', cookie_file,
        '--js-runtimes', 'node',
        '--extractor-args',
        f'youtube:player-client=default,mweb;po_token=mweb.gvs+{po_token}',
        url,
      ]
      cp = subprocess.run(cmd, universal_newlines=True, capture_output=True, text=True)
      os.unlink(cookie_file)
      if cp.returncode != 0:
        if ('ERROR: [youtube]' in cp.stdout and ('video is no longer available' in cp.stdout or 'Video unavailable' in cp.stdout or 'This video is unavailable' or 'This video is not available' in cp.stdout or 'Private video' in cp.stdout or 'This video has been removed' in cp.stdout or 'Join this channel to get access' in cp.stdout or 'This video requires payment to watch' in cp.stdout)) \
                or ('ERROR: [youtube]' in cp.stderr and ('video is no longer available' in cp.stderr or 'Video unavailable' in cp.stderr or 'This video is unavailable' or 'This video is not available' in cp.stderr or 'Private video' in cp.stderr or 'This video has been removed' in cp.stderr or 'Join this channel to get access' in cp.stderr or 'This video requires payment to watch' in cp.stderr)):
          error_queue.put(videoid)
        elif ('ERROR: [youtube]' in cp.stdout and 'Sign in to confirm' in cp.stdout and 'not a bot' in cp.stdout) \
                or ('ERROR: [youtube]' in cp.stderr and 'Sign in to confirm' in cp.stderr and 'not a bot' in cp.stderr)\
                or ('ERROR: [youtube]' in cp.stdout and 'The uploader has not made this video available in your country' in cp.stdout) \
                or ('ERROR: [youtube]' in cp.stderr and 'The uploader has not made this video available in your country' in cp.stderr):
          print(f'!!! Change {proxy} !!!', cp.stderr)
        continue
      info = json.loads(cp.stdout)
      auto_lang = set((info.get('automatic_captions') or {}).keys())
      manu_lang = set((info.get('subtitles') or {}).keys())
      audio_exists = has_audio_only_format(info.get('formats'), lang)
      if not audio_exists:
        print(f'[NO AUDIO] videoid={videoid}, lang={lang}', flush=True)
        empty_queue.put(videoid)
      out_queue.put((videoid, audio_exists, auto_lang, manu_lang))
    except:
      traceback.print_exc()

  #os.remove(cookie_file)
  print(proxy, 'done')

def write_worker(lang, fn_sub, in_queue):
  with open(str(fn_sub), 'a', encoding='utf-8') as f:
    for videoid, audio_exists, auto_lang, manu_lang in iter(in_queue.get, "STOP"):
      auto_exists = audio_exists and has_language(auto_lang, lang)
      manu_exists = audio_exists and has_language(manu_lang, lang)
      line = f'{videoid},{auto_exists},{manu_exists}\n'
      f.write(line)
      f.flush()

  print('write done')

def save_error_worker(error_fn, in_queue):
  with open(str(error_fn), "w") as f:
    for videoid in iter(in_queue.get, "STOP"):
      f.write(videoid+'\n')
      f.flush()
  print('save error done')

def retrieve_subtitle_exists(lang, fn_videoid, proxies, outdir="sub", wait_sec=1, fn_checkpoint=None):
  fn_sub = Path(outdir) / lang / f"{Path(fn_videoid).stem}.csv"
  fn_sub.parent.mkdir(parents=True, exist_ok=True)
  columns = ["videoid", "auto", "sub"]

  # if file exists, load it and restart retrieving.
  if fn_checkpoint is None:
    subtitle_exists = pd.DataFrame(columns=columns, dtype=str)
  else:
    subtitle_exists = pd.read_csv(fn_checkpoint)
    if not set(columns).issubset(subtitle_exists.columns):
      print('Checkpoint columns are incompatible; all video IDs will be checked again.', flush=True)
      subtitle_exists = pd.DataFrame(columns=columns, dtype=str)
    else:
      subtitle_exists = subtitle_exists[columns]
  subtitle_exists.to_csv(fn_sub, index=None)

  vids = set(subtitle_exists["videoid"])
  task_queue = Queue(maxsize=len(proxies))
  done_queue = Queue()
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
  Process(
    target=write_worker,
    args=(
      lang, fn_sub, done_queue
    ),
  ).start()
  processes = []
  for proxy in proxies:
    p = Process(
      target=retrieve_worker,
      args=(
        proxy, lang, task_queue, done_queue, error_queue, empty_queue, wait_sec
      ),
    )
    p.start()
    processes.append(p)
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
  with open(fn_videoid) as f:
    nvids = f.readlines()
  print(len(vids), len(nvids))
  for videoid in tqdm(nvids):
    videoid = videoid.strip(" ").strip("\n")
    if videoid in vids or videoid in error_vids or videoid in empty_vids:
      continue
    task_queue.put(videoid)
    vids.add(videoid)

  for _ in proxies:
    task_queue.put("STOP")

  # Ensure all processes finish execution
  for p in processes:
    if p.is_alive():
      p.join()

  done_queue.put("STOP")
  error_queue.put('STOP')
  empty_queue.put('STOP')
  return fn_sub

if __name__ == "__main__":
  args = parse_args()

  filename = retrieve_subtitle_exists(args.lang, args.videoidlist, args.proxies, \
    args.outdir, fn_checkpoint=args.checkpoint)
  print(f"save {args.lang.upper()} subtitle info to {filename}.")
