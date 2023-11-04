1）youtube-dl异常
https://github.com/ytdl-org/youtube-dl/issues/31530#Description
下载新版
https://github.com/ytdl-org/ytdl-nightly/releases/tag/2023.09.25

2）从维基词典下载youtube搜索词条，https://kaikki.org/dictionary/
python scripts/make_search_word1.py --wikidict https://kaikki.org/dictionary/Indonesian/kaikki.org-dictionary-Indonesian.json --lang id

3）从youtube搜索视频id
nohup python scripts/obtain_video_id.py ja word/word/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

4）检查是视频是否有字幕
nohup python scripts/retrieve_subtitle_exists.py ja videoid/ja/jawiki-latest-pages-articles-multistream-index.txt > ja.log 2>&1 &

5）下载视频
nohup python scripts/download_video.py ja sub/ja/jawiki-latest-pages-articles-multistream-index.csv > ja.log 2>&1 &