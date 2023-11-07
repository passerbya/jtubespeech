#!/usr/bin/python
# coding: utf-8

import argparse
import sys
from espnet_model_zoo.downloader import ModelDownloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download espnet model from https://github.com/espnet/espnet_model_zoo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    #日语Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave
    #泰语espnet/thai_commonvoice_blstm
    parser.add_argument("--asr_model_name", type=str, default='espnet/thai_commonvoice_blstm')
    args = parser.parse_args(sys.argv[1:])

    d = ModelDownloader("~/.cache/espnet")
    ret = d.download_and_unpack(args.asr_model_name)
    print(ret['asr_train_config'])
    print(ret['asr_model_file'])
