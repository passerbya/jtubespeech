#!/usr/bin/python
# coding: utf-8
import os
import hashlib
import re
import chardet
import codecs
import langid
import shutil
import regex
from pathlib import Path

def is_chinese(content):
    mobj = re.search('[\u4E00-\u9FA5]+', content)
    return mobj is not None

def is_japanese(content):
    mobj = re.search('[\u3040-\u309F\u30A0-\u30FF]+', content)
    return mobj is not None

def is_korean(content):
    mobj = re.search('[\uAC00-\uD7A3]+', content)
    return mobj is not None

pattern_space = regex.compile(r'\s')

def normalize_text(txt):
    #<i>Ulysses</i> as a story about a man
    # {\an8}正在录音
    # {\pos(248,100)}家
    # {\an8}{\fn方正黑体简体\fs18\b1\bord1\shad1\3c&H2F2F2F&}"报应"
    # \N{\3c&0x263FE7&}真人秀
    # {*}Fax, dad.
    # {/an8}Locke的旅程
    # 插曲: {fn方正准圆_GBK\c&HFFFFFF&}「Bust A Move」 by Young MC
    # {fad(530,1030)\an8\bord0\shad0\b1\pos(192,185)\fn方正细圆_GBK\fs19\c&HFFFFFF&}32  小  时  前
    # {\fad(500,500)\fs18\bord1\3c&H0F0F14&\pos(248,208)}1958年8月29日
    # 时间轴：{fs18}草鱼禾生       校对：莎楠    小熊嘟嘟    小坏
    # {fry-30}今晚有约会
    # {bei1}帽子神探还活着
    txt = re.sub(r'(?i)\{[^{}]+\}', '', txt)
    txt = re.sub(r'(?i)【[^【】]+】', '', txt)
    txt = re.sub(r'(?i)\([^()]+\)', '', txt)
    txt = re.sub(r'(?i)\[[^\[\]]+\]', '', txt)
    txt = re.sub(r'(?i)\\N', '', txt)
    txt = re.sub(r'(?i)<br(\s*/)?>', '\r\n', txt)
    txt = re.sub(r'(?i)&nbsp;', ' ', txt)
    txt = re.sub(r'<[^<>]*>', '', txt)
    txt = re.sub(r'(?u)^\w+：', '', txt)
    txt = txt.replace('♪', '')
    txt = txt.replace('\h', '')
    txt = pattern_space.sub(" ", txt)

    #?a love like ours is love that's hard to find?
    if txt.find("?") == 0:
        txt = txt[1:]

    return txt

def parse_times(ts):
    #print ts
    ss = ts.split(",")
    s1 = ss[0].split(":")
    t = int(s1[0]) * 3600 + int(s1[1]) * 60 + int(s1[2])
    t = t * 1000 + int(ss[1])
    return t

def read_sub2txt(sub_file):
    with open(sub_file, 'rb') as fhandle:
        content = fhandle.read()
    if content[:2] == codecs.BOM_UTF16:
        content = content[2:]
        try:
            content = content.decode("utf16")
        except Exception as e:
            print(sub_file)
            raise e

    elif content[:3] == codecs.BOM_UTF8:
        content = content[3:]
        content = content.decode("utf-8")
    elif content[:4] == codecs.BOM_UTF32:
        content = content[4:]
        content = content.decode("utf-32")
    else :
        cs = chardet.detect(content)["encoding"]
        try:
            if cs == "GB2312":
                cs = "GB18030"
            content = content.decode(cs)
        except Exception:
            try:
                content = content.decode("utf-8")
            except Exception:
                try:
                    content = content.decode("GB18030")
                except Exception:
                    try:
                        content = content.decode("mbcs")
                    except Exception:
                        try:
                            content = content.decode("utf16")
                        except Exception:
                            None
    return content

def read_srt_sub(sub_file):
    content = read_sub2txt(sub_file)
    subs = []
    sep = '\r\n'
    if content.find(sep) == -1:
        sep = '\n'
        if content.find(sep) == -1:
            sep = '\r'
    is_started = False
    ts = None
    lines = []
    ss = []
    for line in re.split(sep, content):
        if line.find('\r') != -1:
            ss.extend(line.split('\r'))
        elif line.find('\n') != -1:
            ss.extend(line.split('\n'))
        else:
            ss.append(line)

    i = 0
    for line in ss:
        m = re.match(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', line)
        if m is not None:
            is_started = True
            ts = line.split('-->')
        elif is_started:
            if len(line) == 0:
                if i < len(ss) - 1:
                    if len(ss[i+1]) != 0 and not ss[i+1].isnumeric():
                        i += 1
                        continue

                is_started = False
                if len(lines) > 2 or len(lines) < 1:
                    lines = []
                    i += 1
                    continue
                start = parse_times(ts[0].strip())
                end = parse_times(ts[1].strip())
                subs.append((start, end, lines))
                lines = []
            else:
                line = normalize_text(line)
                if len(line) > 0:
                    lines.append(line)
        i += 1

    line1 = 0
    line2 = 0
    for sub in subs:
        if len(sub[2]) == 1:
            line1 += 1
        else:
            line2 += 1

    if line1 > line2:
        #单语
        subs = [sub for sub in subs if len(sub[2])==1]
    else:
        subs = [sub for sub in subs if len(sub[2])==2]
    subs.sort(key=lambda x:(x[0],x[1]))

    return subs, line1 < line2

def md5sum(f):
    m = hashlib.md5()
    n = 1024 * 8
    with open(f, 'rb') as inp:
        while True:
            buf = inp.read(n)
            if not buf:
                break
            m.update(buf)

    return m.hexdigest()

def format_times(ts):
    msec = ts % 1000
    ts = ts / 1000
    s = ts % 60
    ts = ts / 60
    m = ts % 60
    ts = ts / 60
    h = ts

    return "%04d:%02d:%02d" % (h, m, s)

def main():
    lang_stat = {}
    for lang_dir in src.glob("*"):
        if not lang_dir.is_dir():
            continue
        for srt in lang_dir.glob("**/*.srt"):
            wav = srt.parent / (srt.stem + '.wav')
            md5 = md5sum(srt)
            print(srt)
            subs, is_bilingual = read_srt_sub(str(srt))
            if lang_dir.name.startswith('中'):
                lang = 'zh'
            elif lang_dir.name.startswith('日'):
                lang = 'ja'
            elif lang_dir.name.startswith('英'):
                lang = 'en'
            elif lang_dir.name.startswith('阿'):
                lang = 'ar'
            elif lang_dir.name.startswith('德'):
                lang = 'de'
            elif lang_dir.name.startswith('俄'):
                lang = 'ru'
            elif lang_dir.name.startswith('法'):
                lang = 'fr'
            elif lang_dir.name.startswith('韩'):
                lang = 'ko'
            elif lang_dir.name.startswith('葡'):
                lang = 'pt'
            elif lang_dir.name.startswith('泰'):
                lang = 'th'
            elif lang_dir.name.startswith('西'):
                lang = 'es'
            elif lang_dir.name.startswith('印地'):
                lang = 'hi'
            elif lang_dir.name.startswith('越'):
                lang = 'vi'
            wav_path = dest / lang / 'wav_org' / md5[0:2] / (md5 + '.wav')
            txt_path = dest / lang / 'txt' / md5[0:2] / (md5 + '.txt')
            if is_bilingual:
                line0 = ''
                line1 = ''
                new_subs = []
                for sub in subs:
                    lines = sub[2]
                    if len(lines) < 2:
                        continue
                    line0 += lines[0] + '\n'
                    line1 += lines[1] + '\n'
                    new_subs.append(sub)
                _lang0 = langid.classify(line0)[0]
                _lang1 = langid.classify(line1)[0]
                subs = new_subs
                print(lang, _lang0, _lang1)
                if lang == _lang0:
                    for sub in subs:
                        del sub[2][1]
                elif lang == _lang1:
                    for sub in subs:
                        del sub[2][0]
                else:
                    print(txt_path, srt)
                    continue

            has_txt = False
            for sub in subs:
                if len(sub[2]) > 0:
                    has_txt = True
                    if lang not in lang_stat:
                        lang_stat[lang] = 0
                    lang_stat[lang] += sub[1] - sub[0]

            if not has_txt:
                print('-'*20, srt)

            if txt_path.exists() and wav_path.exists():
                continue
            if not txt_path.parent.exists():
                txt_path.parent.mkdir(parents=True)
            if not wav_path.parent.exists():
                wav_path.parent.mkdir(parents=True)
            if wav_path.exists():
                os.remove(str(wav_path))
            shutil.copy(str(wav), str(wav_path))
            with open(txt_path, "wb") as f:
                f.writelines([f"{t[0]/1000:1.3f}\t{t[1]/1000:1.3f}\t\"{' '.join(t[2])}\"\n".encode('utf-8') for t in subs])
            print(is_bilingual, txt_path, srt, 'ok')
    csv_lines = [(float('inf'), 'lang,duration')]
    for lang in lang_stat:
        csv_lines.append((lang_stat[lang], f'{lang},{format_times(lang_stat[lang])}'))
    csv_lines.sort(key=lambda x:-x[0])
    with open(dest/'stat.csv', "wb") as f:
        f.write('\r\n'.join([x[1] for x in csv_lines]).encode('utf-8'))

def to_wav_org():
    for lang_dir in src.glob("*"):
        if not lang_dir.is_dir():
            continue
        for srt in lang_dir.glob("**/*.srt"):
            wav = srt.parent / (srt.stem + '.wav')
            md5 = md5sum(srt)
            print(srt)
            if lang_dir.name.startswith('中'):
                lang = 'zh'
            elif lang_dir.name.startswith('日'):
                lang = 'ja'
            elif lang_dir.name.startswith('英'):
                lang = 'en'
            elif lang_dir.name.startswith('阿'):
                lang = 'ar'
            elif lang_dir.name.startswith('德'):
                lang = 'de'
            elif lang_dir.name.startswith('俄'):
                lang = 'ru'
            elif lang_dir.name.startswith('法'):
                lang = 'fr'
            elif lang_dir.name.startswith('韩'):
                lang = 'ko'
            elif lang_dir.name.startswith('葡'):
                lang = 'pt'
            elif lang_dir.name.startswith('泰'):
                lang = 'th'
            elif lang_dir.name.startswith('西'):
                lang = 'es'
            elif lang_dir.name.startswith('印地'):
                lang = 'hi'
            elif lang_dir.name.startswith('越'):
                lang = 'vi'
            wav_path = dest / lang / 'wav_org' / md5[0:2] / (md5 + '.wav')
            if wav_path.exists():
                continue
            if not wav_path.parent.exists():
                wav_path.parent.mkdir(parents=True)
            if wav_path.exists():
                os.remove(str(wav_path))
            shutil.copy(str(wav), str(wav_path))
            print(wav_path, 'ok')
    print('done')

if __name__ == "__main__":
    '''
    subs, is_bilingual = read_srt_sub('Z:/38语料/语料盘/语料/第四批语料/英译中/21050011_IYUNO-SDI_系列视频翻译0506/Handy Manny61.srt')
    for sub in subs:
        lines = sub[2]
        if is_chinese(lines[0]):
            if len(lines) > 1:
                del lines[1]
        elif len(lines) > 1 and is_chinese(lines[1]):
            del lines[0]
        print(lines)

    with open('F:/data/2/1/《特别呈现》20160524功夫少林第五集天下.txt', "wb") as f:
        f.writelines([f"{t[0]/1000:1.3f}\t{t[1]/1000:1.3f}\t\"{' '.join(t[2])}\"\n".encode('utf-8') for t in subs])
    '''
    src = Path('/usr/local/corpus/4th_wav')
    dest = Path('/usr/local/corpus/4th_biz')
    if not dest.exists():
        dest.mkdir()
    main()
    #to_wav_org()