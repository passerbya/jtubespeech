#!/usr/bin/python
# coding: utf-8

import sys
import json
import requests
import argparse
from pathlib import Path
from util import make_dump_url

def parse_args():
    parser = argparse.ArgumentParser(
        description="Making search words from Wikipedia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wikidict",     type=str, help="url from https://kaikki.org/dictionary/")
    parser.add_argument("lang",     type=str, help="language code (ja, en, ...)")
    parser.add_argument("--outdir", type=str, default="word", help="dirname to save words")
    return parser.parse_args(sys.argv[1:])

def make_search_word(wikidict, lang, outdir="word"):
    words = []
    word_set = set()
    i = 0
    print(wikidict)
    for line in requests.get(wikidict).content.splitlines():
        data = json.loads(line.strip())
        i += 1
        if data['word'] not in word_set:
            words.append(data['word'])
            word_set.add(data['word'])

    print(len(words))

    url = make_dump_url(lang)
    print(url)
    fn_index = Path(outdir) / "dump" / lang / Path(url).name # xxx.txt.bz2
    # obtain words
    fn_word = Path(outdir) / "word" / lang / fn_index.stem
    print(fn_word)
    fn_word.parent.mkdir(parents=True, exist_ok=True)
    with open(fn_word, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + '\n')

    return fn_word

if __name__ == "__main__":
    args = parse_args()

    filename = make_search_word(args.wikidict, args.lang, args.outdir)
    print(f"save {args.lang.upper()} words to {filename}.")