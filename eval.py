import torch
import numpy as np
from typing import List
import ctc_segmentation
import soundfile
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from tts_norm.normalizer import Normalizer
import pykakasi

#conda create -n ctc_segmentation python=3.9

#pip install ctc-segmentation soundfile transformers pykakasi
#pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# load model, processor and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
#accuracy: 0.8810674028065333 std:0.047563648661616865
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr53-zh-cn-subset-colab"
#accuracy: 0.8359861591695502 std:0.03181788966567708
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-mms-1b-zh-CN"
#accuracy: 0.8190410327339788 std:0.03308122022819622
#lang = 'zh'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-en-colab"
#accuracy: 0.9388235294117647  std:0.012640347359232638
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-english"
#accuracy: 0.9411764705882353 std:0.011899561642030494
#model_name = "/usr/local/data/wav2vec2/wav2vec2-base-librispeech32"
#accuracy: 0.9317647058823529 std:0.007981047418406315

'''
from transformers import AutoFeatureExtractor, Wav2Vec2BertForCTC
model_name = "/usr/local/data/wav2vec2/wav2vec2-bert-CV16-en-libri"
processor = AutoProcessor.from_pretrained(model_name)
model = Wav2Vec2BertForCTC.from_pretrained(model_name).to(device)
tokenizer = processor.tokenizer
#60 120
#accuracy: 0.8917647058823529 std:0.020356629734730163
'''
#lang = 'en'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-japanese"
#accuracy: 0.6477217050465458 std:0.06001493003279648
#accuracy: 0.797747306562194 std:0.026634172017197418 hira
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-japanese-hiragana-katakana"
#accuracy: 0.5024801587301587 std:0.079014231770048
#accuracy: 0.8545543584720862 std:0.02839503180511271 hira
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-japanese"
#accuracy: 0.0 std:0.0
#accuracy: 0.8633692458374143 std:0.024852549161806858 hira
#lang = 'ja'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-korean"
#accuracy: 0.7571656050955414 std:0.011154585890674204
#model_name = "/usr/local/data/wav2vec2/wav2vec2-korean-v3"
#accuracy: 0.7794585987261147 std:0.01577937610269156
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-mms-1b-korean-colab_v0"
#accuracy: 0.8151394422310757 std:0.016671138609240816
#lang = 'ko'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-53-vietnamese"
#accuracy: 0.9846153846153847 std:0.010438078038826102
#model_name = "/usr/local/data/wav2vec2/wav2vec-base-vietnamese-noise-dataset-25-epochs"
#accuracy: 0.9692307692307692 std:0.016304799506361326
#model_name = "/usr/local/data/wav2vec2/wav2vec2-base-vios-google-colab"
#accuracy: 0.9846153846153847 std:0.010438078038826102
#model_name = "/usr/local/data/wav2vec2/wav2vec2-base-vios-commonvoice"
#accuracy: 0.9743589743589743 std:0.020181748130315768
#lang = 'vi'

#model_name = "/usr/local/data/wav2vec2/exp_w2v2t_th_wav2vec2_s729"
#accuracy: 0.7972061657032755 std:0.04576658514140403
#model_name = "/usr/local/data/wav2vec2/exp_w2v2t_th_r-wav2vec2_s930"
#accuracy: 0.7972061657032755 std:0.047374596106336236
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-th-v2"
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-mms-1b-thai-colab"
#accuracy: 0.8294797687861272 std:0.030414083747207964
#lang = 'th'
#accuracy: 0.838150289017341 std:0.0370943559688119
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-th-cv11_0"
#accuracy: 0.8439306358381503 std:0.03652105756318442
#lang = 'th'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-ru"
#accuracy: 0.8019607843137255 std:0.03684878659136927
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xlsr-1b-ru"
#accuracy: 0.8058823529411765 std:0.02910028550974495
#lang = 'ru'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-base-10k-voxpopuli-ft-es"
#accuracy: 0.9083769633507853 std:0.039631752352340545
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-53-es"
#accuracy: 0.9057591623036649 std:0.04472810149139926
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-es"
#accuracy: 0.9162303664921466 std:0.03484640104613415
#lang = 'es'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-french"
#accuracy: 0.9622641509433962 std:0.012820526470213499
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-53-french"
#accuracy: 0.9528301886792453 std:0.01385055240193888
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-french"
#accuracy: 0.9622641509433962 std:0.01318112869405873
#lang = 'fr'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-de-with-lm"
#accuracy: 0.944630248978075 std:0.005961701575753355
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-de-cv8"
#accuracy: 0.9442586399108138 std:0.007212153930618737
#lang = 'de'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-pt-cv7-from-bp400h"
#accuracy: 0.9281886387995713 std:0.01627262036761021
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-gn-pt-colab"
#accuracy: 0.9228295819935691 std:0.017012539582805464
#model_name = "/usr/local/data/wav2vec2/exp_w2v2t_pt_r-wav2vec2_s957"
#accuracy: 0.92497320471597 std:0.01914661110290579
#lang = 'pt'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-arabic-colab"
#accuracy: 0.7829204693611473 std:0.03950562118611839
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xls-r-300m-ar"
#accuracy: 0.8245614035087719 std:0.033106673133599204
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-arabic"
#accuracy: 0.8096166341780376 std:0.03187737182100164
#lang = 'ar'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-xlsr-multilingual-56"
#lang = 'zh'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-persian-v3"
#accuracy: 0.9515151515151515 std:0.011828147491267508
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-persian"
#accuracy: 0.9613053613053613 std:0.007100461025272682
#model_name = "/usr/local/data/wav2vec2/wav2vec2-large-xlsr-persian-shemo"
#accuracy: 0.9568764568764568 std:0.010619279723131016
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-fa-colab"
#accuracy: 0.9554778554778555 std:0.010990614392290057
#lang = 'fa'

#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-300m-khmer"
#accuracy: 0.9814814814814815 std:0.0
#lang = 'km'
#model_name = "/usr/local/data/wav2vec2/wav2vec2-xls-r-1b-khmer"
#accuracy: 0.9814814814814815 std:0.0
#lang = 'km'
model_name = "/usr/local/data/wav2vec2/wav2vec2-xlsr-khmer"
#accuracy: 0.9814814814814815 std:0.0
lang = 'km'


print('loading ...')
processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
print('load done')

accuracy = 0
total = 0
percent = []
def align_with_transcript(md5):
    global accuracy, total, percent
    data_path = f'/usr/local/data/wav2vec2/{lang}/{md5}'
    audio, sample_rate = soundfile.read(data_path + '.wav')
    with open(data_path + '.txt') as f:
        utterance_list = f.readlines()
    transcripts = []
    timestamps = []
    normalizer = Normalizer(lang)
    #kks = pykakasi.kakasi()
    for utt in utterance_list:
        utt_start, utt_end, utt_txt = utt.split("\t", 2)
        utt_txt = utt_txt.replace("\n", "").replace('"', "")
        utt_txt, _, _ = normalizer.normalize(utt_txt)
        '''
        result = kks.convert(utt_txt)
        if result is not None and len(result) >0:
            spell = ''
            sep = ''
            for item in result:
                #hira = item['hira']
                hira = item['kana']
                spell += sep + hira
                sep  = ' '
        else:
            spell = text
        transcripts.append(spell)
        '''
        transcripts.append(utt_txt)
        timestamps.append((float(utt_start), float(utt_end)))
    assert audio.ndim == 1
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    if "<unk>" in vocab:
        unk_id = vocab["<unk>"]
    elif "<UNK>" in vocab:
        unk_id = vocab["<UNK>"]
    elif "[unk]" in vocab:
        unk_id = vocab["[unk]"]
    elif "[UNK]" in vocab:
        unk_id = vocab["[UNK]"]

    # Run prediction, get logits and probabilities
    start = 0
    _total = 0
    _accuracy = 0
    while True:
        try:
            start_time = timestamps[start][0]
            end_time = timestamps[start][0]
            end = start
            while end < len(timestamps):
                end = end + 1
                end_time = timestamps[end-1][0]
                if end_time - start_time > 120:
                    break
            if start > 0:
                if end_time - timestamps[start-1][0] <= 180:
                    offset = 1
                    start -= offset
                    start_time = timestamps[start][0]
                else:
                    offset = 0
            else:
                offset = 0
            if end_time - start_time > 180:
                if end-1 > start:
                    end -= 1
                    end_time = timestamps[end-1][0]
                else:
                    if start >= len(timestamps)-1:
                        #忽略并中止
                        print('-'*10)
                        break
                    #忽略本条字幕
                    print('='*10, start, end, transcripts[start:end], end_time - start_time)
                    start = end
                    continue

            timestamp_slice = timestamps[start:end]
            transcripts_slice = transcripts[start:end]
            start_time -= 0.1
            end_time += 0.1
            #print(start, end, end_time - start_time)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            if start_idx < 0:
                start_idx = 0
                start_time = 0
            if end_idx > len(audio):
                end_idx = len(audio)
            audio_slice = audio[start_idx:end_idx]
            inputs = processor(audio_slice, sampling_rate=sample_rate, return_tensors="pt", padding="longest")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                #logits = model(inputs['input_values'].to(device)).logits.cpu()[0]
                logits = model(**inputs).logits.cpu()[0]
                probs = torch.nn.functional.softmax(logits,dim=-1)

            # Tokenize transcripts
            tokens = []
            texts = []
            indexes = []
            for i, transcript in enumerate(transcripts_slice):
                assert len(transcript) > 0
                tok_ids = tokenizer(transcript.replace("\n"," ").lower())['input_ids']
                '''
                for j, tok_id in enumerate(tok_ids):
                    if tok_id == unk_id:
                        print(transcript, unk_id, transcript[j])
                '''
                tok_ids = np.array(tok_ids,dtype=np.int64)
                know_ids = tok_ids[tok_ids != unk_id]
                if len(know_ids) != 0:
                    tokens.append(know_ids)
                    texts.append(transcript)
                    indexes.append(i)

            # Align
            char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
            config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
            config.index_duration = audio_slice.shape[0] / probs.size()[0] / sample_rate

            ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
            timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
            #print(len(tokens), len(texts), len(ground_truth_mat), len(utt_begin_indices), utt_begin_indices, len(timings))
            segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, texts)
            for i, t, p in zip(indexes, texts, segments):
                if i < offset:
                    continue
                if p[2] > 1.5:
                    total += 1
                    _total += 1
                    print("1111", {"start" : p[0]+start_time, "end" : p[1]+start_time, "conf" : p[2], "text" : t})
                    continue
                total += 1
                _total += 1
                _utt_start, _utt_end = timestamp_slice[i]
                if abs(_utt_start - (p[0]+start_time)) > 2 or abs(_utt_end - (p[1]+start_time)) > 2:
                    #print("2222", abs(_utt_start - (p[0]+start_time)), abs(_utt_end - (p[1]+start_time)), t)
                    pass
                else:
                    accuracy += 1
                    _accuracy += 1
                    #print("3333", abs(_utt_start - (p[0]+start_time)), abs(_utt_end - (p[1]+start_time)), t)
            percent.append(accuracy/total)
        except Exception:
            print(transcripts_slice, timestamp_slice)
            import traceback
            traceback.print_exc()
        start = end
        if end >= len(timestamps):
            break
    print(md5, _accuracy/_total)
    print('accuracy:', accuracy/total, accuracy, total)

data = {'zh':['c1a8ee7a2db61a898a3a25ad548a3a61','c11af072f0d66300dd6147cb314018b5','c147803c12c5ea45562caf2dd8ff0c53','c192d0ac6672adb25facc56ebab73551','41ea14397e661aa1381aad8c0a26d1a8','413b484a0ba76c535e369b2597904a76','652d34271d1ec076c6c483e0c68692c6','65f0c870c280c5eaa60cf64ae931fc73','2bbe545556ed29bcb3ca780f14a5dfd8'],
        'en':['cc1844ef241b16d476664e8116521ef6','cc42002bc6fd9ec75bbd9c2980a3741f','efb2782f5a0798494df1eb33c63ff967','ef0ca5dd2a2d3e9afab77c24c961d7f0','b100939046bec839c70379f84a430baa','304abc46b649b4b7bfc88e24ebe02480','309e20897eb2aa12a1be03c0f70babe9','c678d5b021eb98a0ac8feb4d74271d18','c605faf4f581ee9ed94d42de5818bf5b'],
        'ja':['07c088f45055baf6b11bad159bdd9488','7d12569b4e3bd1a4a7cf9152ecb703d8','279c72f2b798953e4e4e4e8497f2fa06','26aef49a425953df7a234fba92f89def','15265c9145149c21b408d53bacfbf22c','153ea910d75faa1696972cb27d283f3d','ccb97584267a760cde96fc1a3ad67f35','be7c5da3f77ce775f01b80b6a1bb45af'],
        'ko':['24c3f5456af523ea257463ae5de62b5c','968f6f091ccbc02e38e7ff817eba22cc','31e8ea363bbd6e2665660d86388ef5ef','39ebeb3bc603463204b8c655ca42afbf','57ddf6161f2a06ff7a3ccc56c40ab63b','ffb75a46325bfc62a52b190b974fa764'],
        'vi':['b18447ee7d7e640858ba4c8df044f286','e888d3fda881136651b71694ed4c6172','4daaf75af7784ef6b8ef5085dd555071','2ce892021ce5053ebfb174aaf040f375','1786c673ec128b563b787852a3956fcc','f520506dce9ef192d0e461bee49b3243','971fd5aa6f53eb5dfe34070924fa1d40','59e88466a6b75b4b18312e6075f0da71','733276f9de44e67c0a8f284e9256b0cd','c00a329c5cec2846c38b7c386f7d9483','ae4e63aa1185dc92d2e46277efee08f8','3d3fc9bedc5b03b4eb1dda352bf05b64','8473112279a14fe4434435f437de2db3','bcf3c1a6309dd143150d8d7bd7f60872','12f0e85bceed58637e3900d128a692f0','100716501521da20e522d3403b91edb4','9eb5374febafdc4e24618a9343c65f3a','e7041c01fad5562043359ecc9bf56285','7b8c03faff436c3b34afed9306c0715b','c1d208481c82dce6c3c9fb96fc75152e'],
        'th':['4b7ef406b50d6d960697c11d90b2ca2d','7a55552a9ed2cd4bfec06db2503b2204','8c10fea079034f93787ad83fcf74c676','99caef00cd1493c976fc735bafbcd2bf','221f13080a6da9fee18a6ab5b88e717f','7408d1402d3d6c3e9275683298e22466','67117eb5ebbefb37d3555e9baea474dc','c6eaa6c33420311f4a1cfa952324129a'],
        'ru':['23124aca454d47d88331035eee8c6648','cf160a5357820fd94e4860c03eb80e40','83bd2b955b0f9b3df21948a05da2d3a0','ff5444e7558bd6f3c8401f2491536f64','7be2e98e3a8338aecfde6959e75252e1'],
        'es':['8b0e978bcb2ddf6e002de0c169c5d790','e552cd6a74c8e394bb0112e3b8d6489f','19e4da5efa6b0c6139a0d10ddde8389d','32f46ac6e05ae391142cc7098ea09b19','327d1eb1f869b26ae7d772f509ccd73a','648a16d3edd2fb721bea265583708aee','ce320363551e5206711a6a728c98adb4','7ef9a456df2b036c9b6123098c9017c2','3b1a22d194228e0bc808b36fc5272a9f','ab83cf5ededec8f6bcbfb7ae37c01eb7'],
        'fr':['ee692a7f2be3c1aab698af8f235a0a77','06eb682b713904eb0824cb8c8cf3e687','11be185fa018f82bf1913e503d2d72ef','f2c06c7253112fd2a987a0092f140c63','d8bb6fd1b92f73d2a1bee1e1f48d3282','da1c71c8d1059c45c28c981d8232bba0','19175517955c09d0ce571ea494ae7136','d0efe4c4f1aa76fb8caa5a58b846c220','3e23d75dcffe70960a529c1267588a77','53a0ae93a6a0e65c55ea0e8c3731efa7','e7e2e0cef70519c150953bc27483e959','faaddb7ec5fd4b7e6a431f9e529dfd90'],
        'de':['427c9c8473019bb033151234437b7dd1','0375601dce4ec5ee233e7975acb955cb','0402380f8edb1fd51ebd6f3f4349e4d6','beb65e2d98347107800e87249944c1d4','b36e087fc70fec89c41ec4d60ca5815b','59f454ae2c0e835e3d52cc98fed77a53','01f16b87270670701780ff8adf7eafe0','475889b203d077c7ce61d284b089347f','64585c8cbb07af26a0739d196b35a606','56c129b70de542423edc868cba3d25f4','56b8515b5a6467045a8d1b9e3379823c','3a2061f6fa043f5facb7ddf37d26c503','a322a0625136f24740a86b9883c965d7','77f2bcdfe8e5e3f528f437a50f055b2f','c9d61dbaebc32ab90ca62a59dc00e66c','80a75ea0bf43d70aebb1a1471a7f9542','4ba5b77b08a3c3fa65ad77530af1debb','d5ea5a3f1ad7f9664a936ed1f89cc20b'],
        'pt':['5aa1518ed74aede11efde41ce51d3f03','5548800a9409a56f0de3b8baf69802c6','60c6db09c2ad4a66fcccee7dc4e421f7','73b9a030f563a56e31ce21ab2878bea1','1b78ebfd703606932c2a746306f96d99','dca6a2ab33920f0631868610119f2c72','8c75602e986e25174d10c7933b39b560','3a5e28dfff9c0475a6c9e8ab8aa65bf5','1f36c1a48abdfe2d2a75e0ee89b3253c','7e993139c36f92f4914778c5b2d26c3f','e71bd174ebeeb87e8326786646db79db','430965aeba72e3290d1ba6ab6c4aace5'],
        'ar':['dd245f768613ee840040a87291362c90','b239a8486ef16272e8330686255c7de3','31473bb25b94ccb24be91e2ed27bc356','a2c34d4f39c0eaee0fc3cf2b0264335b','a35ea818f846d8c9e9d99369198d8b4a'],
        'fa':['6HrKJ87RPi0','aJU8uGdbAaE','gUY8CGcnEhE','iIgfI0UaQs0','KAVaN8XGX7k','wK94ZHGbHIo'],
        'km':['Bk74dS7mSxQ'],
        }
for d in data[lang]:
    align_with_transcript(d)

print('std:', np.std(percent))


