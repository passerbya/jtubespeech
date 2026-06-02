设置代理
export http_proxy=http://192.168.3.100:7890
export https_proxy=http://192.168.3.100:7890

conda activate jtubespeech
python3 -m pip install -U --pre "yt-dlp[default]"

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
pip install pathlib soundfile num2words neologdn romkan ffmpeg-python s3prl jaconv underthesea ctc-segmentation transformers
pip install pyarrow==16.1.0 pandas==2.1.2 fastparquet
conda install -c conda-forge cmake scikit-build-core
pip install ml_dtypes==0.2.0 soxr==0.5.0.post1 onnxruntime-gpu onnxruntime-tools librosa scipy numpy
pip install /usr/local/data/tts-norm/
pip install /usr/local/corpus/penghu/work/VocalExtractor/

conda create -n ytb_cookie python=3.10
conda activate ytb_cookie
pip3 install playwright
python3 -m playwright install chromium

nohup python -u scripts/dnsmos_local.py -t /usr/local/ocr/bilili/zh/flac -o /usr/local/ocr/bilili/zh/dnsmos.csv -p > dnsmos.log 2>&1 &

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
python scripts/make_search_word.py https://kaikki.org/dictionary/Tagalog/kaikki.org-dictionary-Tagalog.jsonl tl
python scripts/make_search_word.py https://kaikki.org/dictionary/Arabic/kaikki.org-dictionary-Arabic.jsonl ar
python scripts/make_search_word.py https://kaikki.org/dictionary/Malay/kaikki.org-dictionary-Malay.jsonl ms
python scripts/make_search_word.py https://kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.jsonl th
python scripts/make_search_word.py https://kaikki.org/dictionary/Vietnamese/kaikki.org-dictionary-Vietnamese.jsonl vi
python scripts/make_search_word.py https://kaikki.org/dictionary/Japanese/kaikki.org-dictionary-Japanese.jsonl ja


3）从youtube搜索视频id
nohup python scripts/obtain_video_id.py ja word/word/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

4）检查是视频是否有字幕
nohup python -u scripts/retrieve_subtitle_exists.py ja videoid/ja/jawiki-latest-pages-articles-multistream-index.txt --checkpoint sub/ja/jawiki-latest-pages-articles-multistream-index.csv --proxies 127.0.0.1:7890 > ja.log 2>&1 &

5）下载视频
nohup python -u scripts/download_video.py ja sub/ja/jawiki-latest-pages-articles-multistream-index.csv --outdir /usr/local/ocr/jtubespeech/video --proxies 127.0.0.1:7890 > ja.log 2>&1 &

export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890 && yt-dlp -v --list-formats https://www.youtube.com/watch?v=yDc0_8emz7M
export http_proxy=http://192.168.8.47:7890 && export https_proxy=http://192.168.8.47:7890 && yt-dlp -v --js-runtimes node --extractor-args "youtube:player-client=default,mweb;po_token=mweb.gvs+MlPA_YR3HhR4wsDBBnSs4Kb5qjFJHmEIvJ_--oUBgYqmHeBtnnqr22Iz6EzvvK49vIwWPeXyqr_dvFl-ZQ1h9J-Pj65pDyjsiU-NqsL95oE5s5Cllg==" https://www.youtube.com/watch?v=yDc0_8emz7M

6）对齐文本与声音
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



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


7）清洗数据
分离背景音乐
nohup python -u scripts/separate.py --path /usr/local/corpus/4th_biz/zh > sep_zh.log 2>&1 &

对长视频进行分段
nohup python -u scripts/segment.py --root /usr/local/corpus/4th_biz/zh > seg_zh.log 2>&1 &

检测mos值
nohup python -u scripts/dnsmos_local.py -t /usr/local/corpus/4th_biz/zh/segs/ -o /usr/local/corpus/4th_biz/zh/segs/dns_mos.jsonl > mos_zh.log 2>&1 &

过滤mos值合格的音频到scp列表
python -u scripts/filter_scp_by_jsonl.py --jsonl /usr/local/corpus/4th_biz/zh/segs/dns_mos.jsonl --output /usr/local/corpus/4th_biz/zh/segs/dns_mos.scp

whisper识别语音文本,生成.whisper.txt文件
nohup python -u scripts/whisper_segs.py --scp /usr/local/corpus/4th_biz/zh/segs/dns_mos.scp > 1.log 2>&1 &

qwen3-asr-flash-filetrans模型识别需要规范化的音频,生成.qwen.txt文件
nohup python -u scripts/qwen_norm_segs.py --root /usr/local/corpus/4th_biz/zh/segs --scp /usr/local/corpus/4th_biz/zh/segs/dns_mos.scp > 1.log 2>&1 &
B站、game数据txt文字准确率高，只需要执行asr进行规范化
nohup python -u scripts/qwen_norm_segs.py --root /usr/local/corpus/game/zh --scp /usr/local/corpus/game/zh/dns_mos.scp --allow-missing-whisper > 1.log 2>&1 &
nohup python -u scripts/qwen_norm_segs.py --root /usr/local/ocr/bilili/zh --scp /usr/local/ocr/bilili/zh/dns_mos.scp --allow-missing-whisper > 1.log 2>&1 &

扫描生成音频与文本文件观对的jsonl
nohup python -u scripts/scan_flac_txt_jsonl.py --root /usr/local/corpus/4th_biz/zh/segs --scp /usr/local/corpus/4th_biz/zh/segs/dns_mos.scp --output /usr/local/corpus/4th_biz/zh/segs/flac_txt.jsonl --skip-empty-txt > 1.log 2>&1 &
开源数据集中直接生成jsonl，不进行whisper\qwen3 asr
nohup python -u scripts/scan_flac_txt_jsonl.py --root /usr/local/corpus/en/hi_fi_tts_v0 --scp /usr/local/corpus/en/hi_fi_tts_v0/dns_mos.scp --output /usr/local/corpus/en/hi_fi_tts_v0/flac_txt.jsonl --skip-empty-txt > 1.log 2>&1 &
nohup python -u scripts/scan_flac_txt_jsonl.py --root /usr/local/corpus/en/LibriTTS-R --scp /usr/local/corpus/en/LibriTTS-R/dns_mos.scp --output /usr/local/corpus/en/LibriTTS-R/flac_txt.jsonl --txt-suffix .normalized.txt --skip-empty-txt > 1.log 2>&1 &
nohup python -u scripts/scan_flac_txt_jsonl.py --root /usr/local/corpus/en/VCTK/wav48_silence_trimmed --txt-root /usr/local/corpus/en/VCTK/txt --scp /usr/local/corpus/en/VCTK/dns_mos.scp --output /usr/local/corpus/en/VCTK/flac_txt.jsonl --skip-empty-txt > 1.log 2>&1 &


统计jsonl中音频时长
nohup python -u scripts/stat_jsonl_flac_duration.py /usr/local/corpus/en/hi_fi_tts_v0/flac_txt.jsonl > 1.log 2>&1 &

