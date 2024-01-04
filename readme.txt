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
apt install gcc g++
apt-get install ffmpeg
pip install pathlib espnet espnet_model_zoo soundfile num2words neologdn romkan ffmpeg-python



1）youtube-dl异常
https://github.com/ytdl-org/youtube-dl/issues/31530#Description
下载新版
https://github.com/ytdl-org/ytdl-nightly/releases/tag/2023.09.25

2）从维基词典下载youtube搜索词条，https://kaikki.org/dictionary/
python scripts/make_search_word1.py --wikidict https://kaikki.org/dictionary/Indonesian/kaikki.org-dictionary-Indonesian.json --lang id

3）从youtube搜索视频id
nohup python scripts/obtain_video_id.py ja word/word/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

4）检查是视频是否有字幕
nohup python scripts/retrieve_subtitle_exists.py ja videoid/ja/jawiki-latest-pages-articles-multistream-index.txt --checkpoint sub/ja/jawiki-latest-pages-articles-multistream-index.csv > ja.log 2>&1 &

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


python scripts/model_downloader.py --asr_model_name kamo-naoyuki/aishell_conformer

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/a1dd2b872b48358daa6e136d4a5ab08b/exp/asr_train_asr_conformer3_raw_char_batch_bins4000000_accum_grad4_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/a1dd2b872b48358daa6e136d4a5ab08b/exp/asr_train_asr_conformer3_raw_char_batch_bins4000000_accum_grad4_sp/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/zh/wav/ --txtdir /usr/local/corpus/4th_biz/zh/txt/ --output /usr/local/corpus/4th_biz/zh/segments/ --ngpu 1


python scripts/model_downloader.py --asr_model_name "Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/29fdff494362014b948fc19e3c753b64/exp/asr_train_asr_raw_en_bpe5000/config.yaml \
 --asr_model_file /root/.cache/espnet/29fdff494362014b948fc19e3c753b64/exp/asr_train_asr_raw_en_bpe5000/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/en/wav/ --txtdir /usr/local/corpus/4th_biz/en/txt/ --output /usr/local/corpus/4th_biz/en/segments/ --ngpu 1


python scripts/model_downloader.py --asr_model_name "Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/1124a7d8d7297e4115691fba79c17478/exp/asr_train_asr_conformer2_latest33_raw_char_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/1124a7d8d7297e4115691fba79c17478/exp/asr_train_asr_conformer2_latest33_raw_char_sp/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/ja/wav/ --txtdir /usr/local/corpus/4th_biz/ja/txt/ --output /usr/local/corpus/4th_biz/ja/segments/ --ngpu 1


python scripts/model_downloader.py --asr_model_name "Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/1124a7d8d7297e4115691fba79c17478/exp/asr_train_asr_conformer2_latest33_raw_char_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/1124a7d8d7297e4115691fba79c17478/exp/asr_train_asr_conformer2_latest33_raw_char_sp/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/ja/wav/ --txtdir /usr/local/corpus/4th_biz/ja/txt/ --output /usr/local/corpus/4th_biz/ja/segments/ --ngpu 1

python scripts/model_downloader.py --asr_model_name "Yushi Ueda/ksponspeech_asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309_valid.acc.best"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/f1b0f522ff3c6aa535403c383916a888/exp/asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309/config.yaml \
 --asr_model_file /root/.cache/espnet/f1b0f522ff3c6aa535403c383916a888/exp/asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309/33epoch.pth \
 --wavdir /usr/local/corpus/4th_biz/ko/wav/ --txtdir /usr/local/corpus/4th_biz/ko/txt/ --output /usr/local/corpus/4th_biz/ko/segments/ --ngpu 1

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--thai_commonvoice_blstm/snapshots/054f24d0eefc6c0822c4bb004f3cbc9256a03fe4/exp/asr_train_asr_rnn_raw_th_bpe150_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--thai_commonvoice_blstm/snapshots/054f24d0eefc6c0822c4bb004f3cbc9256a03fe4/exp/asr_train_asr_rnn_raw_th_bpe150_sp/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/th/wav/ --txtdir /usr/local/corpus/4th_biz/th/txt/ --output /usr/local/corpus/4th_biz/th/segments/ --ngpu 1

7）分离背景音乐
python scripts/separate.py --wavdir video/th/wav16k/ --outdir video/th/wav/


D:\语料\第四批语料\中文-单语\20100046_猫耳FM_铜钱 S21015
包含以下特殊字符的要去掉
♪
^\w+：
【】
()
[]
