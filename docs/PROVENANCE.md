# 来源与复用范围

检索/核对日期：2026-11-25。精确提交与快照 SHA-256 见项目根目录 sources.lock.json。

- Grove：https://github.com/SitongGong/Grove 。已 git fetch 核对，上游 main 与本地基线一致。实验副本修改 episode segmenter、builder、types、structure，并增加 character_identity 模块。没有改原兄弟目录。
- person-identity：https://github.com/jwygit/person-identity （用户私有项目）。复用 identity.py 为 lab/reused/identity.py；复用未提交 av_compare 的 backends.py/core.py 为真实 AV 适配与时间积分指标。旧实验目录保留原样，新的三阶段实验没有沿用旧实验目标。
- M3-Agent：https://github.com/ByteDance-Seed/m3-agent 。参考 voice_processing.py 的 ERes2NetV2（192 维）与 FBank（80 维、16kHz）；新 perception.py 独立初始化，避免原文件导入时立即加载 CUDA。该代码来源采用 Apache-2.0；相关模型权重的使用条款另依原发布者。
- 人脸更新：`mmagent/src/face_clustering.py` 原样复制为 `lab/reused/m3_face_clustering.py`（保留版权头，SHA-256 记录在 sources.lock.json）。`lab/faces.py` 适配 `face_processing.py` 的两阶段筛选/排序与 `videograph.py` 的多代表脸平均余弦匹配。固定采样 seed、图证据结构和 episode 范围是适配差异，详见 DESIGN.md；旧简化去重实现留在 Git 历史。
- 声音更新：`lab/voices.py` 适配 `voice_processing.py` 的时长筛选和逐条声音节点归并、`videograph.py` 的平均相似度匹配与最多 20 条特征。`lab/reused/backends.py` 已在原快照上增加 voice_association 分组协议，不再是原文件逐字节副本；sources.lock.json 记录原始快照哈希与当前适配版哈希。
- M3 的 Gemini 只做轮次/ASR、稳定 voice ID 依赖后续声纹及图；本实验未把 Gemini 输出当作全局 speaker ID。没有复制 M3 的整套 memorization/reasoning agent。

核对本地替代实现时使用的官方资料：

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)：词级时间戳与 VAD 接口。
- [WhisperX](https://github.com/m-bain/whisperX)：ASR、对齐、diarization 分工；本实现未引入其 forced alignment。
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)：本地 diarization 与离线模型说明。
- [FunASR](https://github.com/modelscope/FunASR)：AutoModel 组合及 sentence_info 结构。中文 Paraformer 的结果只在中文子集解释。
- [Gemini 音频接口](https://ai.google.dev/gemini-api/docs/audio)：音频输入与转录；model ID 由运行者显式填写，未写死 M3 已较旧的模型版本。

背景参考：[WorldMM](https://github.com/wgcyeo/WorldMM) 的 Video-MME 预处理使用 faster-whisper；这是“本地转录可作为记忆系统前端”的工程参考，不证明其人物关联准确率。

测试视频来自 AMI Corpus 的 IS1009a，视频 Central camera 与 Mix-Headset 音轨合成，来源索引 https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1009a/ 。AMI Consortium，CC BY 4.0。原来源与区间记录在 data/ami-source-manifest.json；新项目复制现有两段 clip，没有生成或替换真实内容。上游 manifest 内旧路径仅表示原下载记录。
