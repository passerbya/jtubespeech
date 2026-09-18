# pyannote 3.1 与 WeSpeaker VoxBlink2 的选择

本工具目前使用 pyannote/speaker-diarization-3.1。下面区分完整说话人分离系统、
声纹模型及其训练数据，避免把声纹验证成绩直接当作多人片段过滤成绩。
资料核对日期：2026-09-17。

## 型号对应

| 选项 | 实际模型 | 训练数据 | 官方标注 |
| --- | --- | --- | --- |
| 当前 pyannote 3.1 的 embedding | pyannote/wespeaker-voxceleb-resnet34-LM | VoxCeleb，ResNet34-LM | WeSpeaker 对对应 VoxCeleb 系列标 EN |
| WeSpeaker --vblinkp | voxblink2_samresnet34.zip，SimAM-ResNet34 | VoxBlink2 预训练 | Multilingual |
| WeSpeaker --vblinkf | voxblink2_samresnet34_ft.zip，SimAM-ResNet34 | VoxBlink2 预训练，再在 VoxCeleb2 微调 | Multilingual |

vblinkp/vblinkf 都是声纹表示模型的选项。WeSpeaker 自身有 diarization 实现，
但把模型 zip 和现成 pyannote pipeline 放在一起比较时，需要统一分段、聚类和评测设置。

## 支持多少语言

不能把模型表中的 EN 理解成“非英语不能运行”。这些模型根据声音区分人，
并非依赖单一语言词表转写文本。pyannote 3.1 官方的完整系统基准包括
AISHELL-4、AliMeeting、CALLHOME 等不同语言/场景的语料，因此“完全不支持多语言”不成立。

另一方面，能够运行不代表在每种语言上性能一致。VoxBlink2 两个模型有更明确的多语言
训练和模型标注，对语言种类很多的语料是值得比较的声纹候选。
官方上述模型列表没有给三者统一的支持语言清单、逐语言准确率或可直接比较的语言数量。
也不能因为 F 多了一次 VoxCeleb2 微调就宣称 F 支持的语言数比 P 更多。

## 准确率如何比较

WeSpeaker 的 VoxCeleb recipe 报告的是说话人验证 EER：
输入两段音频，判断是不是同一个人。pyannote 完整系统通常报告 DER：
衡量什么时候是谁在说话的错误。EER 和 DER 的分母、任务不同，数值不能直接排名。

VoxBlink2 系列有很强的公开声纹验证成绩，但这不证明它在你的全部语言、短片段、
音乐背景及重叠语音上，都比 pyannote 3.1 的整套流程更适合。
P 与 F 的区别是训练/微调域；F 可能更适合 VoxCeleb 类分布，是否改善你的多语言数据
需要同条件验证，不能仅凭 ft 文件名确定。

本项目关心的是“单人保留、多人排除”，建议直接评测：

- 多人音频漏进单人输出的比例，尤其是很短的第二人声音及重叠语音。
- 单人音频被误删出列表的比例。
- 按语言和片段长度分别统计，避免主流语言掩盖少数语言的问题。
- 在相同分割、重叠检测、聚类方法下比较当前 embedding、vblinkp、vblinkf。

如果优先考虑多语言覆盖，先用 vblinkp 作为候选、vblinkf 作对照是合理的起点；
这是一项待验证的选择，不是已经证明的准确率排名。
现成 pyannote pipeline 提供分段、重叠处理和聚类，可作为完整基线。

## 为什么不直接替换当前模型文件

当前配置中的分割模型是 pyannote/segmentation-3.0，
声纹模型是 pyannote 重新封装的 WeSpeaker ResNet34-LM。
WeSpeaker 原始 zip 中的 avg_model.pt/config.yaml 不能直接当作 pyannote checkpoint 使用。
更换 embedding 还会改变距离分布，应重新校准聚类阈值；
照搬 pyannote 3.1 原阈值不能保证两人识别正确。

本次代码只强化离线加载，没有未经验证就替换声纹后端。
默认模型不是最新或所有场景最优的保证，实际选择应由上述多人过滤指标决定。

## 官方依据

- [WeSpeaker Python 包说明：vblinkp/vblinkf 的确切含义](https://github.com/wenet-e2e/wespeaker/blob/master/docs/python_package.md)
- [WeSpeaker 预训练模型表：VoxBlink2 两个版本均标 Multilingual](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- [WeSpeaker VoxCeleb recipe：EER 及评分条件](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md)
- [pyannote 3.3.2 README：speaker-diarization-3.1 的完整系统 DER 基准](https://github.com/pyannote/pyannote-audio/blob/3.3.2/README.md)
- [pyannote 官方仓库离线示例：3.1 的两份模型及本地路径规则](https://github.com/pyannote/pyannote-audio/blob/3.3.2/tutorials/community/offline_usage_speaker_diarization.ipynb)
