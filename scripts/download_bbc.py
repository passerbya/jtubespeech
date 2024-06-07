import requests
import json
import zipfile
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
  wav_dir = Path('/usr/local/data/jtubespeech/BBC')
  cat_url = 'https://sound-effects-api.bbcrewind.co.uk/api/sfx/categoryAggregations'
  headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'Referer': 'https://sound-effects.bbcrewind.co.uk/',
    'Content-Type': 'application/json',
    'Origin': 'https://sound-effects.bbcrewind.co.uk',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'Priority': 'u=4',
    'TE': 'trailers'
  }
  cat_json = '{"criteria":{"from":0,"size":80,"tags":null,"categories":null,"durations":null,"continents":null,"sortBy":null,"source":null,"recordist":null,"habitat":null}}'
  response = requests.post(cat_url, headers=headers, data=cat_json)
  cat_resp = json.loads(response.text)['aggregations']
  print(cat_resp)
  for key in cat_resp:
    cat_dir = wav_dir / key
    if not cat_dir.exists():
      cat_dir.mkdir()
    sound_list_url = 'https://sound-effects-api.bbcrewind.co.uk/api/sfx/search'
    start = 0
    size = 80
    while True:
      list_json = f'{{"criteria":{{"from":{start},"size":{size},"tags":null,"categories":["{key}"],"durations":null,"continents":null,"sortBy":null,"source":null,"recordist":null,"habitat":null}}}}'
      response = requests.post(sound_list_url, headers=headers, data=list_json)
      list_resp = json.loads(response.text)
      if 'results' not in list_resp or len(list_resp['results']) == 0:
        print(f'{key} done')
        break
      for item in tqdm(list_resp['results']):
        sid = item['id']
        zip_file = wav_dir / 'zip' / f'{sid}.wav.zip'
        if zip_file.exists():
          continue
        zip_url = f'https://sound-effects-media.bbcrewind.co.uk/zip/{sid}.wav.zip?download'
        #print(zip_url)
        temp_dir = wav_dir / 'temp'
        res = requests.get(url=zip_url)
        with open(temp_dir / f'{sid}.wav.zip',mode='wb') as f:
          f.write(res.content)
        f = zipfile.ZipFile(str(temp_dir / f'{sid}.wav.zip'))
        f.extractall(str(cat_dir))
        f.close()
        (temp_dir / f'{sid}.wav.zip').rename(zip_file)
      print(key, size, list_resp['total'], len(list_resp['results']))
      if len(list_resp['results']) == list_resp['total']:
        break
      start += size

  print('all done')