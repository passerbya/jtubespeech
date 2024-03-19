#!/usr/bin/python
# coding: utf-8
import regex
from pathlib import Path
from tts_norm.normalizer import Normalizer

if __name__ == "__main__":
    lang = 'ar'
    src = Path(f'/usr/local/corpus/4th_biz/{lang}/txt')
    normalizer = Normalizer(lang)
    #zh,ja,ko
    #pattern_mix = regex.compile(r"[0-9\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]")
    #en
    pattern_mix = regex.compile(r"[0-9\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3]")
    #th
    #pattern_mix = regex.compile(r"[0-9\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3]")
    for txt_file in src.glob("**/*.txt"):
        has_mix = False
        with open(str(txt_file)) as f:
            utterance_list = f.readlines()
            for utt in utterance_list:
                utt_start, utt_end, utt_txt = utt.split("\t", 2)
                utt_txt = utt_txt.replace("\n", "").replace('"', "")
                if pattern_mix.search(utt_txt) is not None:
                    has_mix = True
                    continue
            if not has_mix:
                print(txt_file)

