#!/usr/bin/python
# coding: utf-8
import argparse
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
    lang_stat = {}
    for lang_dir in src.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        segments_file = lang_dir / 'segments' / 'segments.trans.tsv'
        if not segments_file.exists():
            continue
        with open(segments_file) as f:
            seg_list = f.readlines()
        txts = {}
        for item in seg_list:
            rec_id, _ = item.split('\t', 1)
            md5 = rec_id[:rec_id.rfind('_')]
            txt_file = lang_dir / 'txt' / md5[0:2] / (md5 + '.txt')
            if txt_file not in txts:
                with open(txt_file) as f:
                    utterance_list = f.readlines()
                txt = {}
                for i, utt in enumerate(utterance_list):
                    utt_start, utt_end, _ = utt.split("\t", 2)
                    txt[f"{md5}_{i:04}"] = (float(utt_start), float(utt_end))
                txts[txt_file] = txt
            else:
                txt = txts[txt_file]
            start, end = txt[rec_id]
            if lang not in lang_stat:
                lang_stat[lang] = 0
            lang_stat[lang] += int((end - start)*1000)
    csv_lines = [(float('inf'), 'lang,duration')]
    for lang in lang_stat:
        csv_lines.append((lang_stat[lang], f'{lang},{format_times(lang_stat[lang])}'))
    csv_lines.sort(key=lambda x:-x[0])
    with open(src/'stat.csv', "wb") as f:
        f.write('\r\n'.join([x[1] for x in csv_lines]).encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/ocr/5th_biz"))
    args = parser.parse_args()
    src = args.path
    main()