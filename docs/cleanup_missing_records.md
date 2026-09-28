# 清理 SCP 和 JSONL 中的失效记录

脚本：`scripts/cleanup_missing_records.py`，仅使用 Python 标准库。

检查 `filter_scp_by_jsonl.py` 输出的 SCP，以及 `filter_quality_jsonl.py`
输出的 JSONL。只校验记录中写出的路径：SCP 的 `.flac` 不存在时移除该行；
JSONL 中记录的 `.flac` 或 `.txt` 任一不存在时移除该行。
脚本只修改清单，不会删除清单引用的音频或文本。

## 使用

在数据所在服务器运行；检查的是运行机器上能访问的路径。

不指定输入时，递归扫描默认目录
`/usr/local/ocr/jtubespeech/video/ms/segs` 下的 `*.scp`、
`flac_txt.jsonl` 和 `flac_txt.<语言>.jsonl`：

```bash
# 预览，列出将被移除的记录
python scripts/cleanup_missing_records.py --verbose

# 实际原地更新清单
python scripts/cleanup_missing_records.py --apply
```

也可以指定一个目录：

```bash
python scripts/cleanup_missing_records.py \
  --root /usr/local/ocr/jtubespeech/video/ms/segs --apply
```

单独指定文件时，`--scp`、`--jsonl` 都支持多个路径，可以只使用其中一个参数：

```bash
python scripts/cleanup_missing_records.py \
  --scp /path/to/train.scp /path/to/valid.scp \
  --jsonl /path/to/flac_txt.en.jsonl /path/to/flac_txt.zh.jsonl \
  --apply
```

只有未指定 `--root`、`--scp`、`--jsonl` 时才使用默认目录。
目录扫描忽略评分 JSONL、临时文件和备份；其他文件名的质量清单可以用
`--jsonl` 显式指定。同一个文件不会重复处理。

## 文件与路径规则

- SCP 每行是一个完整的 `.flac` 路径，路径中的空格会保留。
  只检查这个音频文件是否存在，不检查同名 `.txt` 或其他文本文件。
- JSONL 每行必须是两个路径组成的数组，例如
  `["/data/a.flac", "/data/a.qwen.txt"]`。严格检查数组中的两个实际路径，
  不用其他同名文本替代缺失的已选文本。
- 相对路径默认以运行时工作目录为基准，与质量筛选脚本一致。
  可以用 `--path-base /原运行目录` 显式指定；清单所在目录不会被自动猜作基准。
- 文件必须确实存在且是普通文件。指向有效数据文件的符号链接可以保留，
  断开的数据链接视为缺失；清单本身为符号链接时拒绝改写。
- 只检查是否存在，不重新判断语言、质量或文本内容。
  有效记录的原始字节、顺序、重复行、换行和 UTF-8 BOM 保持不变。
  空行也保持不变。

## 更新与错误处理

默认仅预览，`--apply` 才会更新。采用逐行读取、同目录临时文件和原子替换，
不把全部记录装入内存；没有失效记录的清单不替换。

格式错误、编码错误、权限错误、写入失败都不会覆盖该清单。
脚本报告错误后继续处理其他清单；有失败时退出码为 1。
更新前还会检查清单是否被其他进程改动。运行时应暂停该清单的生成任务。

完成后显示各文件及总计的保留、移除、失败数量。
JSONL 中音频和文本同时缺失的记录只移除一次，但两个缺失计数都会增加。
SCP 不检查文本，`missing_txt` 始终为 0。

可先运行语言清理，再清理失效记录：

```bash
python scripts/cleanup_unsupported_languages.py --delete
python scripts/cleanup_missing_records.py --apply
```
