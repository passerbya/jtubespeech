#!/usr/bin/python
# coding: utf-8
import os

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


if __name__ == "__main__":
    lang_dir = Path('/usr/local/corpus/fa')
    subtitle_exists = pd.read_csv(str(lang_dir / 'fawiki-latest-pages-articles-multistream-index.csv'))
    vids = set(subtitle_exists["videoid"])
    print(len(subtitle_exists))
    for i in (lang_dir / 'asr-farsi-youtube-chunked-30-seconds/data').iterdir():
        print(i)
        parquet_file = pq.ParquetFile(str(i))
        data = parquet_file.read().to_pandas()
        for index, row in data.iterrows():
            vid = row['video_id']
            if vid in vids:
                continue
            print(vid)
            vids.add(vid)
            subtitle_exists = pd.concat([subtitle_exists, pd.DataFrame([{"videoid": vid, "auto": True, "sub": True}])], ignore_index=True)


    subtitle_exists.to_csv(str(lang_dir / 'fawiki-latest-pages-articles-multistream-index_.csv'), index=None)
    print(len(subtitle_exists))