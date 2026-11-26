# 来源与复用范围

2026-11-26 人物图更新：新增 `lab/character_links.py` 纯投影/索引契约，逐字节复制到 Grove `modules/character_links.py`（测试检查一致），由本项目维护；不是 M3 原始实现。新增 Grove `character_projection.py` 和 `character_cache.py`，扩展原 writer、图类型、builder 和两条检索入口。`lab/identity_state.py` 继续复用原 `IdentityGraph` 的观测、确认/撤销和审计机制；未改 M3 的聚类源码或联合提示。

视觉检索更新：在同一固定 Grove 副本中补齐 `mmlifelong_week_retrieval_v2.py` 的 Stage E TODO；新增共享的 `retrieval/multimodal.py`，复用 Grove 真实图节点、文本索引接口与跨层得分传递，接入 `HierarchicalRetriever`。改进 `visual_retriever.py` 的候选过滤、错误处理、远程 encoder 接口与索引校验；三个 `visual_indexing.py` 构建入口修正 day offset，现有可选 visual rerank 限定为已命中 clip。没有复制或伪造缺失的 BM25/embedding 索引与 GPU server 实现，具体范围见 VISUAL_RETRIEVAL.md。

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

联合记忆更新：`lab/reused/m3_prompt.py` 的 `M3_PROMPT` 与固定上游 `mmagent/prompts.py:prompt_generate_memory_with_ids_sft` 字符串一致，保留 Apache 版权头，文件与字符串哈希写入 sources.lock.json。`lab/m3.py` 适配 `memory_processing_qwen.py:generate_video_context/generate_all_memories` 和 `utils/chat_qwen.py`，按原输入顺序使用 face_only 代表图、分组 voice、视频音轨，以及相同 generation token IDs、采样与默认 temperature=1e-6；本地媒体直接传路径，标签单独注册映射，时间保留毫秒。新增严格解析与 Grove 版本化映射不是原源码逐字节复制。

默认权重：[ByteDance-Seed/M3-Agent-Memorization](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)。安装指引沿用固定 M3 README 的 Transformers commit `f742a644ca32e65758c3adb36225aef1731bd2a8`、qwen-omni-utils==0.0.4。适配器检查 processor 的显式音频形参以支持 audios/audio，不依靠未知 kwargs；[Transformers 官方 Qwen2.5-Omni 文档](https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/qwen2_5_omni) 作为接口核对资料，未声称已测试所有版本。完整 GPU 依赖锁定和真实推理仍待内网服务器验证。
