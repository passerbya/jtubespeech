import argparse
import torch
import librosa, vocal
import soundfile as sf
from pathlib import Path
import itertools

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("/usr/local/corpus/penghu/work/VocalExtractor/egs/all/testset/mix-44k"))
    parser.add_argument("--out", type=Path, default=Path("/usr/local/corpus/penghu/work/VocalExtractor/egs/all/testset/uvr"))
    args = parser.parse_args()
    flac_dir = args.path
    out_dir = args.out

    device = "cuda:6"
    script_dir = Path(__file__).parent
    voc_ft_model_path = str(script_dir / 'UVR-MDX-NET-Voc_FT.cuda.pt')
    vocal_separator = torch.jit.load(voc_ft_model_path).to(device)  # https://github.com/seanghay/vocal

    out_dir.mkdir(parents=True, exist_ok=True)

    # 处理所有的 flac 和 wav 文件
    for audio_path in itertools.chain(flac_dir.glob("*.flac"), flac_dir.glob("*.wav")):
        print(audio_path)
        flac, sr = librosa.load(str(audio_path), mono=True, sr=44100)

        wav_vocal = vocal.separate_vocal(vocal_separator, flac, device, silent=False)[0]

        sf.write(str(out_dir / f'{audio_path.stem}.flac'), format="flac", data=wav_vocal.T, samplerate=44100)
        print(out_dir / f'{audio_path.stem}.flac')
