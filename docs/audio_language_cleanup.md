# 按语言即时清理 YouTube 音频

每次只处理一个 jtubespeech 根目录和一个语言，`--root`、`--lang` 均为必填，
不能重复指定。脚本边查询边删除，运行后即会删除文件，不再分 scan/apply，
也不需要 SQLite、候选清单或 `--delete` 参数。

## 运行

把更新后的脚本同步到服务器，在包含 `scripts/` 和 `cookies.txt` 的工作目录运行。
以日语为例，先处理 ocr：

```bash
python -u scripts/cleanup_audio_language.py \
  --root /usr/local/ocr/jtubespeech \
  --lang ja \
  --cookies cookies.txt \
  --proxy 192.168.8.23:7890 \
  --proxy 192.168.8.123:7890 \
  --proxy 192.168.8.25:7890 \
  --workers 3
```

处理完后再单独处理 corpus：

```bash
python -u scripts/cleanup_audio_language.py \
  --root /usr/local/corpus/jtubespeech \
  --lang ja \
  --cookies cookies.txt \
  --proxy 192.168.8.23:7890 \
  --proxy 192.168.8.123:7890 \
  --proxy 192.168.8.25:7890 \
  --workers 3
```

下一种语言替换 `--lang ja`。根目录是包含 `video/` 的 jtubespeech 目录。
需要 Python 3.9+、当前下载环境的 yt-dlp 和 Node；无需新增 Python 第三方依赖。
运行期间暂停所选目录、语言的下载、分离、分段和识别进程，避免重新生成文件。

## 处理和断点续跑

启动读取所选根目录对应的三个持久列表：

```text
<root>/videoid/empty/<lang>wiki-latest-pages-articles-multistream-index.txt
<root>/videoid/error/<lang>wiki-latest-pages-articles-multistream-index.txt
<root>/videoid/unknown/<lang>wiki-latest-pages-articles-multistream-index.txt
```

逐桶收集所选语言的文件，在当前桶中：

1. VID 已在 empty：直接删除仍存在的对应文件，不再访问 YouTube。已有记录不区分当初写入原因。
2. VID 已在 error 或 unknown：保留本地文件，跳过查询。同一 VID 同时在 empty 中时，优先按 empty 补删。
3. 其余 VID 查询 YouTube 音轨语言；确认没有目标语言音轨后，先把 VID 去重追加到 empty，并 flush/fsync，随后立即删除该 VID 的文件。
4. Video unavailable、This video is unavailable、私有/已删除/付费视频等进入 error_queue，并立即去重追加到 error 文件；保留本地文件，下次跳过。
5. UNKNOWN AUDIO 进入 unknown_queue，并立即去重追加到 unknown 文件；保留本地文件，下次跳过。
6. 某个文件删除失败时打印路径和原因，继续处理其他文件。VID 已在 empty 中，下次重跑会直接补删。

结果按完成顺序处理，完成的请求不等待其他慢请求。临时超时、限流、机器人验证和地区限制不会写入永久 error，下次仍可重试。

不必等所有视频查询完才释放空间。标准前两位分桶布局只收集当前桶，
不会先扫描全盘生成文件清单。每处理 100 个新查询及每次删除都会打印进度。
同一 VID 在当前运行中只查询一次；跨桶发现的对应文件也会依据 empty 继续清理。
中断后重跑相同命令即可。已删除完且没有残留文件的 VID 不需要再处理。

empty/error/unknown 是持久进度依据，不需要数据库。匹配语言的正常视频没有额外缓存，下次运行仍会查询。
想重新检查某个 error/unknown 视频时，先从相应列表移除该 VID，再重跑清理脚本。
还会有一个很小的 `.cleanup.lock` 文件用于协调同一排除列表上的清理进程，不存储进度。

`--limit N` 只限制本次新增的元数据查询数量，已有 empty 命中的文件仍会删除，
它不是预览模式。需要重试的查询失败或文件删除失败会返回非零退出码；
已成功记入 error/unknown 的结果不算本次未处理的错误。

如果采集程序使用其他代码目录中的 videoid，可以指定：

```bash
python -u scripts/cleanup_audio_language.py \
  --root /usr/local/ocr/jtubespeech \
  --lang ja \
  --videoid-dir /实际工作目录/videoid \
  --cookies cookies.txt \
  --proxy 192.168.8.23:7890
```

这会使用指定的 videoid 排除列表；数据删除范围仍只有所选 root 和 lang。
清理脚本直接持久化 empty_queue 对应的文件，后续下载/字幕检索会加载并跳过这些 VID。

## 判定与删除范围

新查询以 `video/<lang>/` 的目录语言为目标，检查 yt-dlp 返回的
audio-only formats 的 language：

- 有匹配音轨：保留，支持 en/en-US 等标签及 iw/he、jw/jv、in/id。
- 查询成功，所有音轨都有明确语言，但均不匹配：立即记入 empty 并删除。
- 没有纯音频格式、缺失/未知/多语言标签、格式提取不完整：保留，写 unknown，下次跳过。
- 私有/下架/不可用/需付费的视频：保留，写 error，下次跳过。
- 临时网络错误、限流、机器人验证、地区限制、JSON 异常：保留，不写排除列表，下次重试。

UNKNOWN AUDIO 日志包含目标语言、返回的音轨语言和原因：

| reason | 含义 |
| --- | --- |
| no_audio_only_formats | 没有返回带音频编码的纯音频格式 |
| incomplete_format_list | yt-dlp 提示部分格式被跳过或缺少 URL |
| missing_or_ambiguous_language | 至少一条音轨没有 language，或为 und/unknown/mul/zxx |

字幕语言不能补足这些缺失信息；UNKNOWN 不等于已确认音频语言不匹配。

只查询 YouTube 元数据，不下载或识别本地音频。当前音轨信息不能证明历史文件
实际保存的是哪条音轨；例如现在有日语配音的视频，历史下载的英语音轨仍可能被保留。
字幕翻译列表和顶层 language 字段不作为音轨证据。

删除范围为所选根目录下：

```text
video/<lang>/wav_org/
video/<lang>/wav/
video/<lang>/wav16k/
video/<lang>/flac/
video/<lang>/txt/
video/<lang>/vtt/
video/<lang>/segs/
```

按完整 11 位 VID 和 `VID_0000.*` 分段文件名匹配，包含 flac、wav、vtt、txt，
以及 `.lang.txt`、`.whisper.txt`、`.qwen.txt` 等附属文件。
也兼容平铺及桶内多级目录；不递归删除目录，不跟随符号链接，不处理其他语言或另一块阵列。
日志中的字节数是已删除文件的逻辑长度，实际空间释放量以 df 为准。
完成后重建训练使用的 dns_mos.scp、flac_txt.jsonl 等汇总列表，避免引用已删除文件。

## 删除数量与排查

旧日志中的 deleted=3 表示这个 VID 在当前批次实际 unlink 成功的文件数，
不表示删了三个目录，也不一定是这个 VID 在整个运行中的总数。

新日志同时输出：

- matched：当前批次传入删除操作的去重文件数。
- deleted：当前批次实际成功删除的文件数。
- vid_deleted_total：本次运行中这个 VID 的累计成功删除文件数，跨桶累加。
- by_dir：wav_org、wav、wav16k、flac、txt、vtt、segs 各自成功删除的文件数。
- failed：当前批次删除失败数。运行结束的 DONE 行还有全局分类汇总。

加 --verbose 会逐个输出 [UNLINK] 的完整文件路径、大小，并显示因 error/unknown 跳过的 VID。
DATA DIR 日志会列出实际扫描的目录，缺失目录、符号链接也会单独打印。
如果 segs:0，但你确认还有分段文件，应检查是否属于同一个 root/lang，
是否在符号链接后面、是否采用 VID_0000.* 命名，或是否位于尚未扫描到的其他桶。
跨桶的残留文件会在后续扫描时依据 empty 补删；统计只累计实际成功删除的文件。

## 与现有采集脚本配合

download_video.py 和 retrieve_subtitle_exists.py 保持原样。它们原有的 videoid 路径
相对于启动时的工作目录，清理时必须使用同一份列表。
清理脚本的 --videoid-dir 可指定该位置；不要给两个原采集脚本传这个新增参数。
unknown 列表由清理脚本读取，不改变原采集脚本的行为。

原采集脚本会重写排除列表，因此清理期间应暂停它们，清理完成并补删失败文件后再启动。
例如采集工作目录是 /usr/local/data/jtubespeech，数据放在 ocr：

```bash
python -u scripts/cleanup_audio_language.py \
  --root /usr/local/ocr/jtubespeech \
  --lang ja \
  --videoid-dir /usr/local/data/jtubespeech/videoid \
  --cookies cookies.txt \
  --proxy 192.168.8.23:7890
```

运行清理工具的测试（不导入或修改两个采集脚本）：

```bash
python -m unittest discover -s tests -p test_audio_language_cleanup.py -v
```
