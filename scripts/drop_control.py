#!/usr/bin/python
# coding: utf-8
import shutil
import regex
from pathlib import Path

def format_times(ts):
    ts = ts / 1000
    s = ts % 60
    ts = ts / 60
    m = ts % 60
    ts = ts / 60
    h = ts

    return "%d:%02d:%02d" % (h, m, s)

def main():
    for lang_dir in src.iterdir():
        if not lang_dir.is_dir():
            continue
        for txt_file in lang_dir.glob("txt/**/*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            pattern_punctuation_cjk = regex.compile(r"(?![\s])[\p{C}]")
            _content = pattern_punctuation_cjk.sub('', content)
            if len(content) != len(_content):
                print(lang_dir.name, txt_file, len(content), len(_content))
                shutil.copy(str(txt_file), str(lang_dir))
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(_content)

if __name__ == "__main__":
    src = Path('/usr/local/corpus/4th_biz')
    main()