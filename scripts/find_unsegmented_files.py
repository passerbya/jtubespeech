import os
import argparse

def find_unsegmented_files(base_dir):
    long_dir = os.path.join(base_dir, "long")
    long_segments_dir = os.path.join(base_dir, "long_segments")

    if not os.path.exists(long_dir) or not os.path.exists(long_segments_dir):
        print("Error: One or both directories do not exist.")
        return

    # 获取 long 目录下的所有 .flac 文件
    long_files = {f for f in os.listdir(long_dir) if f.endswith(".flac")}

    # 获取 long_segments 目录下的所有切片文件的前缀
    long_segments_files = set()
    for f in os.listdir(long_segments_dir):
        if f.endswith(".flac"):
            base_name = f.rsplit("_speech_", 1)[0] + ".flac"
            long_segments_files.add(base_name)

    # 找出 long 目录中没有被切分的音频文件
    unsegmented_files = long_files - long_segments_files

    # 生成 mv 命令
    for file in unsegmented_files:
        src = os.path.join(long_dir, file)
        dest = os.path.join(base_dir, file)
        print(f"mv \"{src}\" \"{dest}\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and move unsegmented audio files.")
    parser.add_argument("base_dir", help="Base directory containing 'long' and 'long_segments'")
    args = parser.parse_args()

    find_unsegmented_files(args.base_dir)
