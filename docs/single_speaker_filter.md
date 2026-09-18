# 过滤多人说话的音频

入口：scripts/filter_single_speaker.py。支持：

- filter_scp_by_jsonl.py 生成的 SCP：每行一个音频路径。
- filter_quality_jsonl.py 生成的 JSONL：每行 [音频路径, 文本路径]。

对每个音频完整执行 pyannote 说话人分离。默认使用保守判定：模型单标签的音频进入主输出，
满足多人时长和独立发言条件的音频过滤，短片段、短暂第二标签及缺少独立发言证据的片段单独进入待复核列表。
待复核不表示已证明单人或多人。原音频、文本和输入列表保持原样，输出保留原始记录格式。

## 环境和模型

使用已有的 jtubespeech Python 环境，准备 pyannote.audio、soundfile，
以及互相匹配的 torch/torchaudio。默认模型是 pyannote/speaker-diarization-3.1。

pyannote 3.x 使用 torchaudio.AudioMetaData 等旧 API，不适用于 torchaudio 2.11。
如需调整 PyTorch 环境，使用相互配套的版本；已安装这组版本时无需重复安装：

```bash
python -m pip install --no-cache-dir \
  "torch==2.7.1" "torchaudio==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128
```

3.x 的安装示例（适用于项目记录的 NumPy 1.26 环境；保留当前匹配的 torch/torchaudio）：

```bash
python -m pip install --no-cache-dir "pyannote.audio==3.3.2" "huggingface_hub<1" \
  "pyannote.core<6" "pyannote.database<6" "pyannote.metrics<4" "numpy<2" soundfile
```

按照 [官方说明](https://github.com/pyannote/pyannote-audio/blob/3.3.2/README.md)，
先接受以下两个模型的访问条件，再通过 HF_TOKEN 环境变量提供有权限的 Hugging Face token：

- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

token 不写入检查点或命令行参数。也可以使用已登录的 Hugging Face 缓存。
本地推理使用完整音频，不设置 num_speakers=1，以免强制把多人结果合并成一人。

已有本地模型时，传入 --model /模型目录/config.yaml，或者包含 config.yaml 的目录。
离线运行时，配置内引用的分割模型、embedding 模型等也需要已下载，
并按 pyannote 的离线配置方式指向本地路径。
工具支持读取 pyannote 3 的 Annotation 和新版的 speaker_diarization 输出字段。

## 在联网机器下载，复制到无外网服务器

普通过滤现在始终离线，包括加载模型和推理；不需要再加 --offline。
这个参数保留兼容旧命令。只有 --download-model 模式允许下载。

先在一台能够访问 Hugging Face、安装了相同推理依赖的机器上配置 HF_TOKEN，
接受前面的模型访问条件，然后执行：

```bash
python -u scripts/filter_single_speaker.py \
  --download-model \
  --model pyannote/speaker-diarization-3.1 \
  --cache-dir /usr/local/data/models/pyannote
```

命令会下载配置和两份权重，导出以下可搬运目录，并在 CPU 上按本地配置验证能否加载：

```text
/usr/local/data/models/pyannote/local/pyannote--speaker-diarization-3.1/
  config.yaml
  pyannote_segmentation.bin
  pyannote_embedding.bin
```

把整个 local/ 目录复制到服务器相同的 --cache-dir 根目录下即可。
导出的是实际文件，不是指向 Hugging Face 缓存的符号链接。
可以通过内网、移动磁盘或你现有的文件传输方式复制；无外网服务器不运行下载命令。

之后保留默认模型名并传入 --cache-dir，代码会直接找到本地导出的配置。
也可以把模型包搬到任意目录，并使用 --model /新目录/config.yaml。
配置中的权重路径相对配置文件解析，不依赖启动目录。
运行时会验证配置和两份本地权重，生成仅含绝对本地路径的临时配置，
再交给 pyannote；不会把远程模型 ID 传给 Pipeline.from_pretrained。

--cache-dir 在导入模型库前设置 PYANNOTE_CACHE、HF_HOME、HF_HUB_CACHE、
TORCH_HOME、XDG_CACHE_HOME、HF_XET_CACHE、编译缓存和临时目录，避免写入 /root。
推理自动设置 HF_HUB_OFFLINE、TRANSFORMERS_OFFLINE，关闭相关遥测；
主进程与每个 GPU 工作进程还会阻断 Python DNS 和互联网 socket 连接，
包括经 localhost 代理转发的连接。缺少本地文件直接报错，没有在线回退。

旧版下载到该 --cache-dir 的 Hugging Face 缓存也可以复用：
若没有 local/ 模型包，程序只通过 local_files_only=True 查询本机缓存，
转换成本地包；不会检查在线版本。缓存不完整时提示去联网机器准备模型。
已有 local/ 包时，连 Hugging Face 缓存查询都不会执行。

下载阶段仍使用 pyannote 3.x 的 SpeakerDiarization 配置及 PyTorch 模型权重。
WeSpeaker 的 vblinkp/vblinkf zip 是另一种模型格式，不能直接填进这个 --model。
模型选择及多语言对比见 [speaker_model_comparison.md](speaker_model_comparison.md)。

## PyTorch 2.6+ 检查点兼容

PyTorch 2.6 起 torch.load 默认使用 weights_only=True。pyannote 3.x 旧检查点
包含 TorchVersion、Specifications、Problem、Resolution 元数据，可能出现
Weights only load failed / Unsupported global 错误。

本工具在 Pipeline.from_pretrained 期间临时允许上述四个已知类型，加载后恢复原有白名单，
包括异常退出时也会恢复。每个 GPU 工作进程都使用相同处理。
不会全局替换 torch.load，不会自动切换到不受限的 pickle 加载。
若仍出现其他 Unsupported global，会打印具体类型并停止。

同步 scripts/filter_single_speaker.py 和 scripts/pyannote_local.py 后，
对已经下载完成的模型直接做离线验证，无需再次运行 --download-model：

```bash
python - <<'PY'
import sys
sys.path.insert(0, "scripts")
from filter_single_speaker import PyannoteDiarizer, configure_model_cache
configure_model_cache("/usr/local/data/models/pyannote", offline=True)
PyannoteDiarizer("pyannote/speaker-diarization-3.1", "cpu", "HF_TOKEN")
print("本地模型加载成功")
PY
```

## 8 张 A800 同时处理同一个列表

SCP：

```bash
python -u scripts/filter_single_speaker.py \
  --scp /usr/local/ocr/jtubespeech/video/en/segs/dns_mos.scp \
  --devices 0,1,2,3,4,5,6,7 \
  --cache-dir /usr/local/data/models/pyannote \
  --offline
```

JSONL：

```bash
python -u scripts/filter_single_speaker.py \
  --jsonl /usr/local/ocr/jtubespeech/video/en/segs/flac_txt.en.jsonl \
  --devices 0,1,2,3,4,5,6,7 \
  --cache-dir /usr/local/data/models/pyannote \
  --offline
```

--devices 指定的是当前进程可见的逻辑 GPU 编号，受 CUDA_VISIBLE_DEVICES 影响，
不要和单卡 --device 同时使用。每张 GPU 启动一个 spawn 子进程，各自加载一份模型；
共享任务队列动态分配音频，某张卡处理完就领取下一条。Torch CPU 线程默认每进程 2 个，
可以使用 --cpu-threads 调整，避免八个进程各自开满服务器 CPU。

主进程是唯一的检查点写入者，每张卡完成结果后立即追加并 fsync。
在途任务数量受限，文件内容在各工作进程内读取，不通过队列传输整段波形。
所有检测结束后再流式读取原列表，从缓存中按原始顺序生成正式输出，
因此推理完成顺序不会打乱 SCP/JSONL；重复输入行仍会保留，相同音频只推理一次。
生成正式列表不再调用模型。

异常退出或 CUDA 内存不足会停止本次任务并清理子进程，已保存的结果继续有效。
再次执行同一命令即可续跑。已有单卡检查点也可直接切换到多卡；
改变卡数、卡编号或缓存目录不使检测缓存失效。
若所有音频都有有效结果，不会启动 GPU 工作进程。

## 单卡运行

SCP：

```bash
python -u scripts/filter_single_speaker.py \
  --scp /usr/local/ocr/jtubespeech/video/en/segs/dns_mos.scp \
  --device cuda:0
```

生成：

```text
dns_mos_single_speaker.scp
dns_mos_single_speaker_review.scp
dns_mos_single_speaker.state.jsonl
```

JSONL：

```bash
python -u scripts/filter_single_speaker.py \
  --jsonl /usr/local/ocr/jtubespeech/video/en/segs/flac_txt.en.jsonl \
  --device cuda:0
```

生成：

```text
flac_txt.en_single_speaker.jsonl
flac_txt.en_single_speaker_review.jsonl
flac_txt.en_single_speaker.state.jsonl
```

本地模型示例：

```bash
python -u scripts/filter_single_speaker.py \
  --scp /数据目录/dns_mos.scp \
  --model /模型目录/config.yaml \
  --device cuda:0
```

JSONL 保留原始记录，包括原来的文本路径、附加字段、Unicode、顺序。
SCP 同样保留原行顺序；音频路径可包含空格。输入中的重复行仍会出现在输出，
但同一路径、文件签名相同的音频只推理一次。两个输入选项不能同时指定。

相对音频路径默认相对于启动命令时的当前工作目录，与现有筛选脚本一致。
需要指定其他基准目录时使用 --path-base /目录。
--device auto 默认自动选 CUDA 或 CPU，也可指定 cpu、cuda:1 等。
同一个列表使用 --devices 在多卡间分发；不同列表也可分开启动，各用自己的输出和检查点。

## 断点续跑

重跑同一条命令即可，不需要额外 resume 参数：

1. 每个处理结果都立即追加到 .state.jsonl，flush/fsync 后再继续。
2. 下次启动读取已处理结果，跳过对应音频的模型推理。
3. 中间文件记录说话人时长、语音总时长、状态和音频文件信息。
   音频大小、mtime、设备或 inode 改变时自动重新处理。
4. 输入新增记录会继续处理；输入记录重排或 JSONL 文本路径改变时，
   会按当前输入重新生成输出，已缓存的音频检测结果仍可复用。
5. 检查点最后一行因中断未写完时会自动修复；
   中间完整行损坏或模型配置不一致会明确报错，不静默丢弃。
6. 换模型或修改本地 config.yaml 时应使用新的 --state 路径。
   如果仅替换本地权重但 config.yaml 不变，也应选新的检查点。
7. 多个进程通过 .lock 文件协调输出和检查点写入。正常完成、普通异常和 Ctrl+C 后清理锁文件；
   SIGKILL/断电可能留下空文件，下次运行可重新获取并清理。等待者会校验锁文件身份，避免删除锁文件导致两个任务同时写入。
   更新前先停止处理同一输入列表的旧版任务。
8. 音频、输入列表、输出和状态路径保留输入时的软链接写法，例如 /usr/local 不会改写为 /vdus。
   状态中的路径由用户完成替换后，直接按输入路径和文件签名精确匹配。
   不再按 inode 查找路径别名，不做旧路径迁移，也不会为整理路径重写状态文件。

推理阶段逐条保存检查点；推理完成后按原始输入顺序生成主输出和待复核列表的 .tmp 文件，成功后分别替换正式输出。
中断时 .tmp 和 .state.jsonl 会留下，重跑会从检查点重建结果，不会追加重复输出。
即使在保存检测结果后、写输出前中断，也可恢复。
如果所有输入都有有效缓存，可以直接重建输出，无需加载模型。

没有语音的音频不作为单人语音保留。待复核项默认只写到 *_single_speaker_review.scp/jsonl，
不会自动混入单人主输出，也不再直接记作多人。读取或推理失败的音频也不加入通过列表，
错误会写入检查点，下次默认跳过。需要重试失败项时加 --retry-errors：

```bash
python -u scripts/filter_single_speaker.py \
  --scp /数据目录/dns_mos.scp \
  --device cuda:0 \
  --retry-errors
```

模型加载失败会直接退出，不会把整个输入误写成音频错误。
CUDA 内存不足等运行环境错误也会中断，之前已完成的结果仍可续跑。
完成时如有未解决的文件错误，退出码为 1，同时输出成功筛选的记录。

## 可选参数

- --devices 0,1,2,3,4,5,6,7：每张 GPU 一个进程，共同处理输入列表。
- --cpu-threads 2：每个 GPU 工作进程使用的 Torch CPU 线程数。
- --cache-dir /usr/local/data/models/pyannote：指定模型及相关缓存根目录。
- --download-model：仅在联网机器下载并导出可搬运的本地模型包，要求指定 --cache-dir。
- --offline：保留兼容；所有普通过滤运行都已强制离线，即使不传也不会联网。
- --suffix _single_speaker：输出文件名后缀，放在 .scp/.jsonl 前。
- --state /路径/progress.jsonl：指定中间文件，可跨 SCP/JSONL 复用同一模型的音频结果。
- --verbose：打印每条音频的判定、说话人数和各说话人时长。
- --limit 100：只处理输入前 100 条，用于试跑；去掉后会复用已完成结果并处理完整列表。
- --decision-policy conservative：默认保守分流；strict 使用原来的标签数/阈值判定。
- --min-speaker-seconds 0.5：默认多人证据要求，每个说话人累计至少 0.5 秒。
- --min-speaker-ratio 0.1：默认多人证据要求，每个说话人至少占有效语音时长的 10%。
- --min-multispeaker-speech 3：默认总有效语音不足 3 秒且有多个标签时进入待复核。
- --min-exclusive-seconds 0.3：默认至少两人分别有 0.3 秒非重叠发言，才满足多人过滤条件。
- --uncertain-action review：默认待复核单列；keep 会同时将待复核项放入主输出，应由用户明确选择。
- --rebuild-only：仅使用完整状态文件重建主输出和待复核列表，不启动模型、不修改缓存路径。缺少有效缓存时明确报错。

旧版默认两个阈值都为 0，极短的第二标签也会被当作多人，容易误伤短音频。
新版保守规则是工程上的待复核策略，阈值并未在你的全量数据上标定，不代表已提升模型本身的准确率。
真实多人重叠、短对话也可能进入待复核；不能将待复核列表直接认定为单人。
若要完全恢复旧规则，使用 --decision-policy strict --min-speaker-seconds 0 --min-speaker-ratio 0。
判定策略、阈值变化不影响模型结果缓存，可以直接重建输出。
新推理额外记录各人的独立发言时长及重叠时长；旧版两人结果可从累计时长和语音并集推导。
旧版三人且有重叠但无法还原细节时，会进入待复核。

pyannote 的判定存在误差，尤其是很短的片段、背景人声、歌声或音色变化明显的音频。
可先使用 --limit 和 --verbose 做小批量检查，再运行全量。
工具对完整音频逐条推理，每次加载一段；特别长的音频需考虑内存占用。

## 按判定原因独立输出

每次处理或使用 --rebuild-only，除了主输出和汇总 review，还会自动生成以下分类文件。
以 flac_txt.zh.jsonl 为例：

```text
flac_txt.zh_single_speaker_review_short_speech.jsonl
flac_txt.zh_single_speaker_review_brief_or_low_share_secondary.jsonl
flac_txt.zh_single_speaker_review_insufficient_nonoverlap_speech.jsonl
flac_txt.zh_single_speaker_review_legacy_overlap_ambiguous.jsonl
flac_txt.zh_single_speaker_multiple.jsonl
```

这些文件保留输入中的原始 [音频路径, 文本路径] 行、附加字段、顺序和重复记录，
方便分别抽样盲听。SCP 输入则生成同名分类的 .scp 文件。
各 review 分类互斥，合起来与汇总 review 内容对应；multiple 单独存放。
待复核和 multiple 都是当前模型与规则的结果，不代表人工核实的标签。

结束日志增加每个分类文件的 [OUTPUT] 路径和条数。每次成功运行都更新全部分类文件，
某类为 0 条时写空文件，避免上一次的旧结果残留。状态文件和判定阈值不受分类输出影响。

## 使用已有状态重新筛选

在你已跑完的服务器上，同步最新脚本后执行：

```bash
python -u scripts/filter_single_speaker.py \
  --jsonl /usr/local/corpus/4th_biz/zh/segs/flac_txt.zh.jsonl \
  --state /usr/local/corpus/4th_biz/zh/segs/flac_txt.zh_single_speaker.state.jsonl \
  --cache-dir /usr/local/data/models/pyannote \
  --rebuild-only
```

这条命令不加载 GPU 模型，按新规则重建 *_single_speaker.jsonl 和 *_single_speaker_review.jsonl，
状态路径已手动替换为 /usr/local/... 时直接复用，不再执行路径兼容或迁移。
使用服务器上完整的 .state.jsonl（含 header）；用户提供的 1.jsonl 是结果子集，不能直接当完整检查点使用。
如音频已变更或输入新增了未处理记录，命令会指出缺少缓存的路径；去掉 --rebuild-only 才会继续推理。
六个样本及该结果子集的分析见 [single_speaker_sample_review.md](single_speaker_sample_review.md)。

## 验证

```bash
python -m unittest discover -s tests -p test_filter_single_speaker.py -v
python -m unittest discover -s tests -p test_pyannote_local.py -v
python -m unittest discover -s tests -p test_single_speaker_regressions.py -v
```

测试覆盖两种输入、原始格式保留、轮流/重叠说话、断点续跑、末行修复、
输入或音频变化、错误重试及缓存兼容性。多卡测试实际创建 spawn 子进程，验证任务分发、
乱序结果落盘、单卡转多卡续跑、重复音频、异常退出和内存不足后的恢复。
新增 test_pyannote_local.py 验证模型包换目录、缺失文件拒绝、旧缓存本地转换、默认离线和联网阻断。
测试使用模拟模型，没有实际下载受限权重或在 A800 上运行真实 pyannote 推理；实际吞吐和识别效果需在服务器评估。
