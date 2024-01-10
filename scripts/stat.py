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
    skip_durations = {}
    for lang_dir in src.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        segments_file = lang_dir / 'segments' / 'segments.txt'
        if not segments_file.exists():
            continue
        with open(segments_file) as f:
            seg_list = f.readlines()
        txts = {}
        for item in seg_list:
            no, md5, start, end, score, _ = item.replace('\n', '').split(' ', 5)
            txt_file = lang_dir / 'txt' / md5[0:2] / (md5 + '.txt')
            if txt_file not in txts:
                with open(txt_file) as f:
                    utterance_list = f.readlines()
                utterance_list = [
                    item.replace("\t", " ").replace("\n", "") for item in utterance_list
                ]
                txt = {}
                for i, utt in enumerate(utterance_list):
                    utt_start, utt_end, _ = utt.split(" ", 2)
                    txt[f"{md5}_{i:04}"] = (float(utt_start), float(utt_end), utt)
                txts[txt_file] = txt
            else:
                txt = txts[txt_file]
            start = float(start)
            end = float(end)
            score = float(score)
            if score > -0.3:
                #print(md5)
                continue
            utt_start, utt_end, utt = txt[no]
            if abs(utt_start - start) > 2:
                if lang not in skip_durations:
                    skip_durations[lang] = 0
                skip_durations[lang] += int((utt_end - utt_start)*1000)
                print(lang, md5, format_times(skip_durations[lang]))
                continue
            if lang not in lang_stat:
                lang_stat[lang] = 0
            lang_stat[lang] += int((utt_end - utt_start)*1000)
    csv_lines = [(float('inf'), 'lang,duration')]
    for lang in lang_stat:
        csv_lines.append((lang_stat[lang], f'{lang},{format_times(lang_stat[lang])}'))
    csv_lines.sort(key=lambda x:-x[0])
    with open(src/'stat.csv', "wb") as f:
        f.write('\r\n'.join([x[1] for x in csv_lines]).encode('utf-8'))

if __name__ == "__main__":
    src = Path('/usr/local/corpus/4th_biz')
    main()