#!/usr/bin/python
# coding: utf-8
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
        segments_file = lang_dir / 'segments' / 'segments.txt'
        if not segments_file.exists():
            continue
        with open(segments_file) as f:
            seg_list = f.readlines()
        nos = set()
        for item in seg_list:
            no, md5, start, end, score, txt = item.replace('\n', '').split(' ', 5)
            if no in nos:
                print(item)
                continue
            nos.add(no)
            start = float(start)
            end = float(end)
            score = float(score)
            if score > -0.3:
                #print(item)
                continue
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
    src = Path('/usr/local/corpus/4th_biz')
    main()