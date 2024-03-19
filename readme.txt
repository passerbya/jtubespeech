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
pip install pathlib espnet espnet_model_zoo soundfile num2words neologdn romkan ffmpeg-python s3prl jaconv underthesea ctc-segmentation soundfile transformers



1）youtube-dl异常
https://github.com/ytdl-org/youtube-dl/issues/31530#Description
下载新版
https://github.com/ytdl-org/ytdl-nightly/releases/tag/2023.09.25

2）从维基词典下载youtube搜索词条，https://kaikki.org/dictionary/
python scripts/make_search_word1.py https://kaikki.org/dictionary/Indonesian/kaikki.org-dictionary-Indonesian.json id
python scripts/make_search_word1.py https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.json ko

3）从youtube搜索视频id
nohup python scripts/obtain_video_id.py ja word/word/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

4）检查是视频是否有字幕
nohup python scripts/retrieve_subtitle_exists.py ja videoid/ja/jawiki-latest-pages-articles-multistream-index.txt --checkpoint sub/ja/jawiki-latest-pages-articles-multistream-index.csv > ja.log 2>&1 &

5）下载视频
nohup python scripts/download_video.py ja sub/ja/jawiki-latest-pages-articles-multistream-index.csv > ja.log 2>&1 &

6）对齐文本与声音
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

下载CTCSegmentation模型
langs="ar as br ca cnh cs cv cy de dv el en eo es et eu\
 fa fr fy-NL ga-IE hsb ia id it ja ka kab ky lv mn mt\
 nl or pa-IN pl pt rm-sursilv rm-vallader ro ru rw sah\
 sl sv-SE ta tr tt uk vi zh-CN zh-HK zh-TW"
Corpus combination with 52 languages(Commonvocie + voxforge)
python scripts/model_downloader.py --asr_model_name ftshijt/open_li52_asr_train_asr_raw_bpe7000_valid.acc.ave_10best
/root/.cache/espnet/811ae5a5580d9e5a8dcdc98f16b3c196/exp/asr_train_asr_raw_bpe7000/config.yaml \
/root/.cache/espnet/811ae5a5580d9e5a8dcdc98f16b3c196/exp/asr_train_asr_raw_bpe7000/valid.acc.ave_10best.pth \

145 languages
python scripts/model_downloader.py --asr_model_name espnet/interspeech2024_dsuchallenge_wavlm_large_21_baseline
/root/.cache/espnet/models--espnet--interspeech2024_dsuchallenge_wavlm_large_21_baseline/snapshots/8f5c6af3b2903f46ec80e0f138c1fe9c06ae3697/exp/asr_train_discrete_asr_e_branchformer1_1gpu_lr5e-4_warmup5k_raw_wavlm_large_21_km2000_bpe_rm3000_bpe_ts6000/config.yaml
/root/.cache/espnet/models--espnet--interspeech2024_dsuchallenge_wavlm_large_21_baseline/snapshots/8f5c6af3b2903f46ec80e0f138c1fe9c06ae3697/exp/asr_train_discrete_asr_e_branchformer1_1gpu_lr5e-4_warmup5k_raw_wavlm_large_21_km2000_bpe_rm3000_bpe_ts6000/valid.acc.ave_10best.pth


python scripts/model_downloader.py --asr_model_name "espnet/jiyangtang_magicdata_asr_conformer_lm_transformer"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--jiyangtang_magicdata_asr_conformer_lm_transformer/snapshots/0937e0af018ed7261a939bdcb1b3bd8732bb7ff5/exp/asr_train_asr_raw_zh_char_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--jiyangtang_magicdata_asr_conformer_lm_transformer/snapshots/0937e0af018ed7261a939bdcb1b3bd8732bb7ff5/exp/asr_train_asr_raw_zh_char_sp/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/zh/wav/ --txtdir /usr/local/corpus/4th_biz/zh/txt/ --output /usr/local/corpus/4th_biz/zh/segments/ --ngpu 1 --lang zh


python scripts/model_downloader.py --asr_model_name "Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/29fdff494362014b948fc19e3c753b64/exp/asr_train_asr_raw_en_bpe5000/config.yaml \
 --asr_model_file /root/.cache/espnet/29fdff494362014b948fc19e3c753b64/exp/asr_train_asr_raw_en_bpe5000/valid.acc.ave_10best.pth \
 --wavdir /usr/local/corpus/4th_biz/en/wav/ --txtdir /usr/local/corpus/4th_biz/en/txt/ --output /usr/local/corpus/4th_biz/en/segments/ --ngpu 1 --lang en


python scripts/model_downloader.py --asr_model_name "reazon-research/reazonspeech-espnet-next"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--reazon-research--reazonspeech-espnet-next/snapshots/20f564ce571263ad488379c0cc033e0228a37eea/exp/asr_train_asr_conformer_raw_jp_char/config.yaml \
 --asr_model_file /root/.cache/espnet/models--reazon-research--reazonspeech-espnet-next/snapshots/20f564ce571263ad488379c0cc033e0228a37eea/exp/asr_train_asr_conformer_raw_jp_char/valid.acc.ave_3best.pth \
 --wavdir /usr/local/corpus/4th_biz/ja/wav/ --txtdir /usr/local/corpus/4th_biz/ja/txt/ --output /usr/local/corpus/4th_biz/ja/segments/ --ngpu 1 --lang ja

python scripts/model_downloader.py --asr_model_name "Yushi Ueda/ksponspeech_asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309_valid.acc.best"
python scripts/align.py \
 --asr_train_config /root/.cache/espnet/f1b0f522ff3c6aa535403c383916a888/exp/asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309/config.yaml \
 --asr_model_file /root/.cache/espnet/f1b0f522ff3c6aa535403c383916a888/exp/asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309/33epoch.pth \
 --wavdir /usr/local/corpus/4th_biz/ko/wav/ --txtdir /usr/local/corpus/4th_biz/ko/txt/ --output /usr/local/corpus/4th_biz/ko/segments/ --ngpu 1 --lang ko

#https://huggingface.co/datasets/google/fleurs
#https://huggingface.co/espnet/wanchichen_fleurs_asr_conformer_hier_lid_utt
python scripts/model_downloader.py --asr_model_name espnet/wanchichen_fleurs_asr_conformer_hier_lid_utt
/root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/config.yaml
/root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/valid.acc.ave_3best.pth
vim /root/miniconda3/envs/jtubespeech/lib/python3.9/site-packages/espnet2/bin/asr_align.py
424行
        assert len(enc) >= 1, len(enc)
        # Apply ctc layer to obtain log character probabilities
        if len(enc) > 1:
             lpz = self.ctc.log_softmax(enc[0]).detach()
        else:
             lpz = self.ctc.log_softmax(enc).detach()

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/valid.acc.ave_3best.pth \
 --wavdir /usr/local/corpus/4th_biz/th/wav/ --txtdir /usr/local/corpus/4th_biz/th/txt/ --output /usr/local/corpus/4th_biz/th/segments/ --ngpu 1 --lang th

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/valid.acc.ave_3best.pth \
 --wavdir /usr/local/corpus/4th_biz/hi/wav/ --txtdir /usr/local/corpus/4th_biz/hi/txt/ --output /usr/local/corpus/4th_biz/hi/segments/ --ngpu 1 --lang hi

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/valid.acc.ave_3best.pth \
 --wavdir /usr/local/corpus/4th_biz/ru/wav/ --txtdir /usr/local/corpus/4th_biz/ru/txt/ --output /usr/local/corpus/4th_biz/ru/segments/ --ngpu 1 --lang ru

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/config.yaml \
 --asr_model_file /root/.cache/espnet/models--espnet--wanchichen_fleurs_asr_conformer_hier_lid_utt/snapshots/ab07127c8e882dcaf1936d85480fb9cf51e19a97/exp/asr_train_asr_raw_all_bpe6500_sp/valid.acc.ave_3best.pth \
 --wavdir video/vi/wav16k/ --txtdir video/vi/txt/ --output segments/vi/ --ngpu 1 --lang vi


python scripts/align.py \
 --asr_train_config /root/.cache/espnet/811ae5a5580d9e5a8dcdc98f16b3c196/exp/asr_train_asr_raw_bpe7000/config.yaml \
 --asr_model_file /root/.cache/espnet/811ae5a5580d9e5a8dcdc98f16b3c196/exp/asr_train_asr_raw_bpe7000/valid.acc.ave_10best.pth \
 --wavdir video/th/wav16k/ --txtdir video/th/txt/ --output segments/th/ --ngpu 1 --lang th

English, German, Dutch, Spanish, French, Italian, Portuguese, Polish
python scripts/model_downloader.py --asr_model_name ftshijt/mls_asr_transformer_valid.acc.best
/root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/config.yaml
/root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/29epoch.pth

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/config.yaml \
 --asr_model_file /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/29epoch.pth \
 --wavdir /usr/local/corpus/4th_biz/de/wav/ --txtdir /usr/local/corpus/4th_biz/de/txt/ --output /usr/local/corpus/4th_biz/de/segments/ --ngpu 1 --lang de

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/config.yaml \
 --asr_model_file /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/29epoch.pth \
 --wavdir /usr/local/corpus/4th_biz/es/wav/ --txtdir /usr/local/corpus/4th_biz/es/txt/ --output /usr/local/corpus/4th_biz/es/segments/ --ngpu 1 --lang es

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/config.yaml \
 --asr_model_file /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/29epoch.pth \
 --wavdir /usr/local/corpus/4th_biz/fr/wav/ --txtdir /usr/local/corpus/4th_biz/fr/txt/ --output /usr/local/corpus/4th_biz/fr/segments/ --ngpu 1 --lang fr

python scripts/align.py \
 --asr_train_config /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/config.yaml \
 --asr_model_file /root/.cache/espnet/a08c8578c2074b6848d7b7130d62073b/exp/asr_transformer/29epoch.pth \
 --wavdir /usr/local/corpus/4th_biz/pt/wav/ --txtdir /usr/local/corpus/4th_biz/pt/txt/ --output /usr/local/corpus/4th_biz/pt/segments/ --ngpu 1 --lang pt


cd segments/th/
awk -v ms=-0.3 '{ if ($5 > ms) {print} }' segments.txt > bad.txt

km

7）分离背景音乐
python scripts/separate.py --wavdir video/th/wav16k/ --outdir video/th/wav/


D:\语料\第四批语料\中文-单语\20100046_猫耳FM_铜钱 S21015
包含以下特殊字符的要去掉
♪
^\w+：
【】
()
[]
