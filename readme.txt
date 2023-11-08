设置代理
export http_proxy=http://192.168.3.100:7890
export https_proxy=http://192.168.3.100:7890

git bash获取代码
ssh-agent bash
ssh-add ~/.ssh/id_ed25519
git clone git@hf.co:openai/whisper-large-v2


安装python环境
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-complex --extra-index-url https://download.pytorch.org/whl/cu117
source /opt/rh/devtoolset-8/enable
pip install pathlib espnet espnet_model_zoo soundfile num2words neologdn  romkan



1）youtube-dl异常
https://github.com/ytdl-org/youtube-dl/issues/31530#Description
下载新版
https://github.com/ytdl-org/ytdl-nightly/releases/tag/2023.09.25

2）从维基词典下载youtube搜索词条，https://kaikki.org/dictionary/
python scripts/make_search_word1.py --wikidict https://kaikki.org/dictionary/Indonesian/kaikki.org-dictionary-Indonesian.json --lang id

3）从youtube搜索视频id
nohup python scripts/obtain_video_id.py ja word/word/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

4）检查是视频是否有字幕
nohup  python scripts/retrieve_subtitle_exists.py ja videoid/ja/jawiki-latest-pages-articles-multistream-index.txt --checkpoint sub/ja/jawiki-latest-pages-articles-multistream-index.csv > ja.log 2>&1 &

5）下载视频
nohup python scripts/download_video.py ja sub/ja/jawiki-latest-pages-articles-multistream-index.csv > ja.log 2>&1 &

6）对齐文本与声音

下载CTCSegmentation模型
https://huggingface.co/espnet/thai_commonvoice_blstm
python scripts/model_downloader.py --asr_model_name espnet/thai_commonvoice_blstm

mkdir -p segments/th/

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--thai_commonvoice_blstm/snapshots/054f24d0eefc6c0822c4bb004f3cbc9256a03fe4/exp/asr_train_asr_rnn_raw_th_bpe150_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--thai_commonvoice_blstm/snapshots/054f24d0eefc6c0822c4bb004f3cbc9256a03fe4/exp/asr_train_asr_rnn_raw_th_bpe150_sp/valid.acc.ave_10best.pth \
 --wavdir video/th/wav16k/ --txtdir video/th/txt/ --output segments/th/ --ngpu 1

cd segments/th/
min_confidence_score=-0.3
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' segments.txt > bad.txt

7）分离背景音乐
python scripts/separate.py --wavdir video/th/wav16k/ --outdir video/th/wav/

