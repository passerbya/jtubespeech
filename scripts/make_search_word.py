import requests
import bz2
import argparse
import sys
from util import make_dump_url
from pathlib import Path
import langid
import re
import regex
import html

def is_chinese(content):
  mobj = re.search('[\u4E00-\u9FA5]+', content)
  return mobj is not None

def is_japanese(content):
  mobj = re.search('[\u3040-\u309F\u30A0-\u30FF]+', content)
  return mobj is not None

def is_korean(content):
  mobj = re.search('[\uAC00-\uD7A3]+', content)
  return mobj is not None

def parse_args():
  parser = argparse.ArgumentParser(
    description="Making search words from Wikipedia",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("lang",     type=str, help="language code (ja, en, ...)")
  parser.add_argument("--outdir", type=str, default="word", help="dirname to save words")
  return parser.parse_args(sys.argv[1:])


def make_search_word(lang, outdir="word"):
  # download wikipedia index
  url = make_dump_url(lang)
  print(url)
  fn_index = Path(outdir) / "dump" / lang / Path(url).name # xxx.txt.bz2
  fn_index.parent.mkdir(parents=True, exist_ok=True)

  if not fn_index.exists():
    with open(fn_index, "wb") as f:
      f.write(requests.get(url).content)

  # obtain words
  fn_word = Path(outdir) / "word" / lang / fn_index.stem
  print(fn_word)
  fn_word.parent.mkdir(parents=True, exist_ok=True)

  with bz2.open(fn_index, "rt", encoding="utf-8") as f:
    words = list(map(lambda x: x.strip("\n").split(":")[-1], f.readlines()))
  words = [w.strip(" ") for w in set(words) if len(w) > 0]
  words.sort()

  with open(fn_word, "w", encoding="utf-8") as f:
    for w in words:
      try:
        w = html.unescape(w)
        if len(w) > 20 or len(w) < 2:
          continue
        pattern = regex.compile(r'[\p{P}$\d]')
        if 3 * len(pattern.findall(w)) >= len(w):
          print(w)
          continue
        if lang == 'ja':
          if not is_japanese(w) and not is_chinese(w):
            print(w)
            continue
        elif lang == 'zh':
          if not is_chinese(w) or is_japanese(w) or is_korean(w):
            print(w)
            continue
        elif lang == 'ko':
          if not is_korean(w):
            print(w)
            continue
        else:
          if is_chinese(w) or is_japanese(w) or is_korean(w):
            print(w)
            continue
          '''
          _lang = langid.classify(w)[0]
          if lang != _lang:
            print(lang, _lang, w)
            continue
          '''
        f.write(w + "\n")
      except:
        continue

  return fn_word

if __name__ == "__main__":
  args = parse_args()

  filename = make_search_word(args.lang, args.outdir)
  print(f"save {args.lang.upper()} words to {filename}.")
