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
    lang_stat = 0
    for txt_file in (src / 'txt').glob("**/*.txt"):
        with open(txt_file) as f:
            seg_list = f.readlines()
        for item in seg_list:
            if len(item.replace('\n', '').split('	', 2)) < 3:
                print(txt_file.stem)
                continue
            start, end, _ = item.replace('\n', '').split('	', 2)
            lang_stat += int((float(end) - float(start))*1000)
        txt_dir = txt_file.parent
        if txt_file.stem[0:2] != txt_dir.name:
            txt_dir_real = src / 'txt' / txt_file.stem[0:2]
            if not txt_dir_real.exists():
                txt_dir_real.mkdir()
                print('mkdir', txt_dir_real)
            txt_file_real = txt_dir_real / txt_file.name
            if not txt_file_real.exists():
                txt_file.rename(txt_file_real)
                print('rename', txt_file_real)
            else:
                txt_file.unlink()

        wav24k_file_real = src / 'wav24k' / txt_file.stem[0:2] / (txt_file.stem + '.wav')
        if not wav24k_file_real.exists():
            print('no wav', wav24k_file_real)

    for wav24k_file in (src / 'wav24k').glob("**/*.wav"):
        wav16_dir = wav24k_file.parent
        wav_dir_real = src / 'wav' / wav24k_file.stem[0:2]
        if not wav_dir_real.exists():
            wav_dir_real.mkdir()
            print('mkdir', wav_dir_real)
        if wav24k_file.stem[0:2] != wav16_dir.name:
            wav24k_dir_real = src / 'wav24k' / wav24k_file.stem[0:2]
            if not wav24k_dir_real.exists():
                wav24k_dir_real.mkdir()
                print('mkdir', wav24k_dir_real)
            wav24k_file_real = wav24k_dir_real / wav24k_file.name
            if not wav24k_file_real.exists():
                wav24k_file.rename(wav24k_file_real)
                print('rename', wav24k_file_real)
            else:
                wav24k_file.unlink()

        txt_file_real = src / 'txt' / wav24k_file.stem[0:2] / (wav24k_file.stem + '.txt')
        if not txt_file_real.exists():
            print('no txt', txt_file_real)

    with open(src/'stat.txt', "wb") as f:
        f.write(format_times(lang_stat).encode('utf-8'))

if __name__ == "__main__":
    src = Path('/usr/local/corpus/jtubespeech/th')
    main()