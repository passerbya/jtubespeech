设置代理
export http_proxy=http://192.168.3.100:7890
export https_proxy=http://192.168.3.100:7890

git bash获取代码
ssh-agent bash
ssh-add ~/.ssh/id_ed25519
git clone git@hf.co:openai/whisper-large-v2


安装python环境
conda create -n jtubespeech python==3.10
conda activate jtubespeech
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 torch-complex
apt install gcc g++
apt-get install ffmpeg
pip install pathlib soundfile num2words neologdn romkan ffmpeg-python s3prl jaconv underthesea ctc-segmentation soundfile transformers
pip install pyarrow fastparquet
pip install /usr/local/data/tts-norm/
pip install /usr/local/corpus/penghu/work/VocalExtractor/


1）youtube-dl异常
https://github.com/ytdl-org/youtube-dl/issues/31530#Description
下载新版
https://github.com/ytdl-org/ytdl-nightly/releases/tag/2023.09.25

2）从维基词典下载youtube搜索词条，https://kaikki.org/dictionary/
python scripts/make_search_word.py https://kaikki.org/dictionary/Indonesian/kaikki.org-dictionary-Indonesian.jsonl id
python scripts/make_search_word.py https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl ko
python scripts/make_search_word.py https://kaikki.org/dictionary/Persian/kaikki.org-dictionary-Persian.jsonl fa
python scripts/make_search_word.py https://kaikki.org/dictionary/Khmer/kaikki.org-dictionary-Khmer.jsonl km
python scripts/make_search_word.py https://kaikki.org/dictionary/Lao/kaikki.org-dictionary-Lao.jsonl lo

3）从youtube搜索视频id
nohup python scripts/obtain_video_id.py ja word/word/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

4）检查是视频是否有字幕
nohup python -u scripts/retrieve_subtitle_exists.py ja videoid/ja/jawiki-latest-pages-articles-multistream-index.txt --checkpoint sub/ja/jawiki-latest-pages-articles-multistream-index.csv --proxies 127.0.0.1:7890 > ja.log 2>&1 &

5）下载视频
nohup python -u scripts/download_video.py ja sub/ja/jawiki-latest-pages-articles-multistream-index.csv --outdir /usr/local/ocr/jtubespeech/video --proxies 127.0.0.1:7890 > ja.log 2>&1 &

6）对齐文本与声音
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32



python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/zh/wav/ --txtdir /usr/local/corpus/4th_biz/zh/txt/ --output /usr/local/corpus/4th_biz/zh/segments/ --lang zh

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/en/wav/ --txtdir /usr/local/corpus/4th_biz/en/txt/ --output /usr/local/corpus/4th_biz/en/segments/ --lang en

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/ja/wav/ --txtdir /usr/local/corpus/4th_biz/ja/txt/ --output /usr/local/corpus/4th_biz/ja/segments/ --lang ja

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/ko/wav/ --txtdir /usr/local/corpus/4th_biz/ko/txt/ --output /usr/local/corpus/4th_biz/ko/segments/ --lang ko

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/th/wav/ --txtdir /usr/local/corpus/4th_biz/th/txt/ --output /usr/local/corpus/4th_biz/th/segments/ --lang th

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/ru/wav/ --txtdir /usr/local/corpus/4th_biz/ru/txt/ --output /usr/local/corpus/4th_biz/ru/segments/ --lang ru

python -u scripts/align.py \
 --wavdir video/vi/wav16k/ --txtdir video/vi/txt/ --output segments/vi/ --lang vi

python -u scripts/align.py \
 --wavdir video/th/wav16k/ --txtdir video/th/txt/ --output segments/th/ --lang th


python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/de/wav/ --txtdir /usr/local/corpus/4th_biz/de/txt/ --output /usr/local/corpus/4th_biz/de/segments/ --lang de

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/es/wav/ --txtdir /usr/local/corpus/4th_biz/es/txt/ --output /usr/local/corpus/4th_biz/es/segments/ --lang es

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/fr/wav/ --txtdir /usr/local/corpus/4th_biz/fr/txt/ --output /usr/local/corpus/4th_biz/fr/segments/ --lang fr

python -u scripts/align.py \
 --wavdir /usr/local/corpus/4th_biz/pt/wav/ --txtdir /usr/local/corpus/4th_biz/pt/txt/ --output /usr/local/corpus/4th_biz/pt/segments/ --lang pt


7）分离背景音乐
python scripts/separate.py /usr/local/ocr/5th_biz/zh


D:\语料\第四批语料\中文-单语\20100046_猫耳FM_铜钱 S21015
包含以下特殊字符的要去掉
♪
^\w+：
【】
()
[]
