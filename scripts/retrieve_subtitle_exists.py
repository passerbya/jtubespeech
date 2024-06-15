import time
import requests
import argparse
import re
import sys
import traceback
import subprocess
from pathlib import Path
from util import make_video_url, get_subtitle_language
import pandas as pd
from tqdm import tqdm
from torch.multiprocessing import Process, Queue

def parse_args():
  parser = argparse.ArgumentParser(
    description="Retrieving whether subtitles exists or not.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("lang",         type=str, help="language code (ja, en, ...)")
  parser.add_argument("videoidlist",  type=str, help="filename of video ID list")  
  parser.add_argument("--outdir",     type=str, default="sub", help="dirname to save results")
  parser.add_argument("--checkpoint", type=str, default=None, help="filename of list checkpoint (for restart retrieving)")
  return parser.parse_args(sys.argv[1:])

def retrieve_worker(proxy, lang, in_queue, out_queue, wait_sec):
  for videoid in iter(in_queue.get, "STOP"):
    url = make_video_url(videoid)
    try:
      cmd = f"export http_proxy=http://{proxy} && export https_proxy=http://{proxy} && yt-dlp --list-subs --sub-lang {lang} --skip-download {url}"
      print(cmd)
      result = subprocess.check_output(cmd, shell=True, universal_newlines=True)
      auto_lang, manu_lang = get_subtitle_language(result)
      out_queue.put((videoid, auto_lang, manu_lang))
    except:
      traceback.print_exc()
    # sleep
    if wait_sec > 0.01:
      time.sleep(wait_sec)

def write_worker(lang, fn_sub, subtitle_exists, in_queue):
  for videoid, auto_lang, manu_lang in iter(in_queue.get, "STOP"):
    subtitle_exists = pd.concat([subtitle_exists, pd.DataFrame([{"videoid": videoid, "auto": lang in auto_lang, "sub": lang in manu_lang}])], ignore_index=True)
    # write current result
    if len(subtitle_exists) % 100 == 0:
      subtitle_exists.to_csv(fn_sub, index=None)

def retrieve_subtitle_exists(lang, fn_videoid, outdir="sub", wait_sec=0.2, fn_checkpoint=None):
  fn_sub = Path(outdir) / lang / f"{Path(fn_videoid).stem}.csv"
  fn_sub.parent.mkdir(parents=True, exist_ok=True)

  # if file exists, load it and restart retrieving.
  if fn_checkpoint is None:
    subtitle_exists = pd.DataFrame({"videoid": [], "auto": [], "sub": []}, dtype=str)
  else:
    subtitle_exists = pd.read_csv(fn_checkpoint)

  vids = set(subtitle_exists["videoid"])
  proxies = ['169.254.9.60:7890', '169.254.9.70:7890', '192.168.8.25:7890']
  task_queue = Queue(maxsize=len(proxies))
  done_queue = Queue()
  # Start worker processes
  Process(
    target=write_worker,
    args=(
      lang, fn_sub, subtitle_exists, done_queue
    ),
  ).start()
  for proxy in proxies:
    Process(
      target=retrieve_worker,
      args=(
        proxy, lang, task_queue, done_queue, wait_sec
      ),
    ).start()
  for videoid in tqdm(open(fn_videoid).readlines()):
    videoid = videoid.strip(" ").strip("\n")
    if videoid in vids:
      continue
    task_queue.put(videoid)

  for _ in proxies:
    task_queue.put("STOP")
  while not task_queue.empty() or not done_queue.empty():
    time.sleep(20)
  done_queue.put("STOP")
  # write
  subtitle_exists.to_csv(fn_sub, index=None)
  return fn_sub

if __name__ == "__main__":
  args = parse_args()

  filename = retrieve_subtitle_exists(args.lang, args.videoidlist, \
    args.outdir, fn_checkpoint=args.checkpoint)
  print(f"save {args.lang.upper()} subtitle info to {filename}.")
