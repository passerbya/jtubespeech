# 清理非训练语言数据

脚本：`scripts/cleanup_unsupported_languages.py`，仅使用 Python 标准库。

默认目录为 `/usr/local/ocr/jtubespeech/video/ms`，固定保留以下 16 种语言：

`ar fa id ja km ko lo ms th tl vi es fr ru zh en`

运行前停止对该目录执行分段、转写和质量筛选的任务，避免它们重新生成文件。

## 使用

在服务器的 jtubespeech 项目目录执行。默认只预览；加 `--verbose` 可以查看每个待删除文件：

```bash
python scripts/cleanup_unsupported_languages.py --verbose
```

确认预览后，实际删除：

```bash
python scripts/cleanup_unsupported_languages.py --delete
```

也可以显式指定数据集目录：

```bash
python scripts/cleanup_unsupported_languages.py \
  --root /usr/local/ocr/jtubespeech/video/ms --delete --verbose
```

## 清理规则

1. 递归扫描 `segs/**/*.lang.txt`，以 UTF-8（兼容 BOM）读取第一个非空行，
   去除两端空白并转为小写。代码不在保留列表时，删除同名的
   `.flac`、`.txt`、`.whisper.txt`、`.qwen.txt`，相关清理成功后删除 `.lang.txt`。
2. 依据 `segment.py` 的普通目录规则，将
   `segs/ab/video_id_0000.flac` 对应回 `flac/ab/video_id.flac`。
   只移除文件名末尾的分段序号，支持四位及以上序号、原名中的下划线和多层子目录。
3. 仅对发现非保留语言的原文件进行检查。当该原文件没有剩余分段音频或文本时，
   删除以下对应文件（相对目录和原文件名必须完全相同）：
   - `flac/<相对路径>.flac`
   - `txt/<相对路径>.txt`
   - `vtt/<相对路径>.vtt`
   - `wav_org/<相对路径>.flac` 和 `.wav`
   - `wav/<相对路径>.flac` 和 `.wav`
4. 删除 `segs` 下（含子目录）语言不在列表中的
   `flac_txt.<语言>.jsonl`，例如 `flac_txt.pl.jsonl`。
   文件名只匹配两到三位字母的语言码，以及已知的 `_single_speaker` 系列后缀。
   对这些带处理后缀的文件，以下划线前的代码判断语言：
   `flac_txt.en_single_speaker.jsonl`、`flac_txt.en_single_speaker_review_*.jsonl`
   都属于 `en`，会保留；其他保留语言的此类文件也会保留。
   `flac_txt.pl_single_speaker.jsonl` 属于 `pl`，会删除。
   保留未按语言拆分的 `flac_txt.jsonl`，不改写其中的记录。
   `stat_jsonl_flac_duration.py` 的 `.duration.jsonl`、`.stat.csv`，以及
   `classify_flac_txt_accent.py` 的 `.accent.progress.jsonl`、口音标签 JSONL、CSV
   均保留。例如 `flac_txt.duration.jsonl`、`flac_txt.en.us.jsonl`、
   `flac_txt.en.o.jsonl` 不会被删除；其他不符合上述语言清单命名的文件也保留。
   对未按语言拆分的 `flac_txt.jsonl` 做口音分类后，标签可能与语言码重名；
   如果同目录存在 `flac_txt.accent.progress.jsonl`，或对应的
   `flac_txt.stat.<标签>.csv`，相关有歧义的 JSONL 会保留。
5. `.lang.txt` 最后删除；分段或应清理的原文件删除失败时，暂留标记供重试。
   它不算作剩余分段文本。
   任一分段仍存在音频或其他文本（包括转写文本）时，都会保留原文件。
   缺失、空白或无法读取的语言标记不会触发对应分段删除。
   不根据空目录推断其他未分段原文件也应被删除。

不跟随符号链接，不删除整个目录。文件删除失败会报告错误并继续处理其他文件；
失败的分段会阻止对应原文件删除。失败时暂留的语言标记使脚本可以反复执行，
包括上次已删除分段、但原文件删除失败的情况；成功后标记也会清除。

预览模式会模拟删除后的剩余分段情况，因此也会统计满足条件的原文件。
输出按目录显示进度，最后列出语言、分段、原文件、JSONL 的计数和文件大小合计。
大小合计是文件逻辑字节数；硬链接、压缩等可能使实际释放空间不同。
有读取、扫描或删除错误时退出码为 1。

此脚本针对 `video/<语言>` 的普通分段布局，不支持 `bilili/zh` 的特殊分桶布局。
