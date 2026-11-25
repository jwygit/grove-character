# 实际验证记录

日期：2026-11-25。环境：macOS，Python 3.13.9；隔离环境 NumPy 2.5.5、Pydantic 2.14.4、HDBSCAN 0.8.44、scikit-learn 1.9.1、SciPy 1.18.1。未运行任何 GPU 模型。

| 验证 | 结果 | 可以说明什么 |
|---|---|---|
| `python -m unittest discover -s tests -t . -v` | 98 项通过：65 项模块测试、33 项 Grove 测试 | 真实 HDBSCAN 计算、声音分组/匹配、M3 联合协议、图操作、指标、边界后接入、标签转写及序列化契约 |
| M3 原始联合 prompt 比对 | 与固定提交的字符串逐字符一致，文件/字符串 SHA 入锁文件 | 没有改写为 association-only 提示 |
| M3 聚类源码比对 | 与上游文件逐字节一致 | 未重新编写或替换其 HDBSCAN 算法 |
| `scripts/smoke.py` | 生成 snapshot、真实 Grove hypergraph、synthetic report | 假 LLM / 合成向量下，写入路径可跑，结果明确不用于选型 |
| AMI 真实视频 ffmpeg 解码 | 音频长度 60.000 秒 | 媒体可读取；已处理 AAC 解码尾部 padding，否则为 60.032 秒 |
| AMI PCM 的 1.125–4.625 秒切片 | 3.5 秒、56000 个样本，与原 PCM 对应范围逐字节一致 | 声纹切片不再经过视频 AAC 重编码；尚未运行 ERes2NetV2 权重 |
| 无 Gemini 配置/密钥运行 | 抛错并写 status=failed | 缺依赖时不会伪造预测或偷用标注 |
| compileall | 通过 | 新代码及修改的 Grove 文件语法有效 |
| Grove git diff --check | 通过 | 无 diff 空白错误 |
| 清洁 Grove 副本应用补丁并逐文件比较 | 12 个文件完全一致 | 补丁可重现当前实验改动，包括新增模块 |
| 本次完整导出目录部署 | 离线还原 Grove、98 项测试、人物/视觉两条 smoke 通过，12 个补丁文件逐字节一致 | 人物写入和视觉检索可从仓库独立初始化测试，GPU 推理仍未验证 |
| 首次发布的 Git 全新导出目录部署 | 离线还原 Grove、当时 31 项测试及 smoke 全部通过，两段视频齐全 | 仓库无需原本机兄弟目录或访问 Grove 上游即可初始化实验代码；GPU 环境未验证 |

Grove 测试使用其真实 VideoEpisode/VideoFact、Pydantic 图节点、VideoHypergraph.save/load，以及新增 writer/provider。boundary hook 测试执行真实修改后的 `_finalize_episode`，**仅用测试替身补上公开仓库缺失的 prompt/llm_utils**；它验证调用顺序，不能证明真实模型的动态边界划分准确。

覆盖的失败场景包括：漏项/重复 AV 输出、虚构 face/utterance ID、跨模态 embedding 比较、候选边不合并、历史画外音复识别、未知短发言、图版本/时间范围不符、错误角色归属、错误关联、全拒识导致召回为零、跨边界句子不重复写入。

新增人脸测试使用构造的归一化特征，实际执行 HDBSCAN（非 mock），检查两个人物成簇、质量门限、噪声和不足 20 个样本的拒识、最佳 3 张代表脸、均值匹配区别于最近邻、模型空间隔离、最多 10 个匹配特征、快照重载/重放一致，以及多代表脸到 character 的映射。首次沙箱内运行卡在 SciPy 原生库加载，沙箱外运行最终代码的 41 项测试全部通过；本机库加载问题不作为模型准确率结论。

声音更新后本机完整运行 56 项测试通过。新增声音测试验证共同的 >=2 秒规则、忽略前端独有 overlap 标签进行筛选、无损 PCM 切片、多声纹均值区别于最大值、20 条上限和可复现采样、不同模型隔离、旧图迁移不按共享人脸误合声音、先分组后推理、短句无身份但保留文字、画外发言继承 voice/character、输出协议完整性与组内一致性，以及声音分组贯穿 Grove 写入/重载。Gemini 协议测试使用假 HTTP 响应，GPU 声纹和 Omni 推理仍未执行。

尚未验证：Gemini 实际 API 响应、Whisper/pyannote/FunASR 的安装和模型推理、InsightFace/ERes2NetV2 输出、Omni GPU 推理、真实 writer 的事实忠实度、真实 ASR/AV/character 准确率、全 Grove topic/retrieval 链路、实时并行调度与显存/费用。

`scripts/preflight.py` 会列出缺失上游文件，当前 full_grove_pipeline_ready=false。没有把测试替身补成生产源码来冒充完整 Grove 已可运行。

联合记忆更新：75 项 CPU 测试通过（58 模块、17 Grove）。新增覆盖：原始 prompt 和输入顺序、先分组后联合生成、MM:SS 毫秒时间、单复数描述键兼容、未知/矛盾等价结论拒绝、candidate 与 confirmed 的不同映射、无声音视觉记忆、注册标签重放幂等、身份对计分；GPU 调用通过测试替身验证 audios/audio 两种签名、音轨传入和 generation 参数，这不代表真实 GPU 推理通过。

Grove 新增测试执行实际边界 finalize → 取完整 M3 evidence → 摘要转写；验证正文与结构化引用共用映射、无证据标签拒绝、完整标签替换、unknown 对照重渲染、原文/图版本保存，以及真实类型/超图重载。新版 smoke 使用合成 M3 输出和假 writer，生成可检查的原文/character 正文，仍明确 synthetic，不能用于准确率选型。

视觉检索更新：新增 23 项测试（7 项视觉索引、16 项 Grove 检索），完整 98 项 CPU 测试。覆盖真实 NPZ 与 cosine、候选先过滤再 top-k、各版 clip ID metadata、编码空间不匹配/服务故障、fact 与 visual clip 共享预算、无 fact 的视觉 episode、严格 day 范围、跨层传播、旧时间戳兼容、历史 cutoff、图超边一致性、Grove 保存/加载、核心 API 开关，以及 week retrieval 的 Stage 1 不能跳过视觉通道、结果类型和错误路径。文本索引/encoder/LLM 使用明确测试替身，未执行真实模型。

`smoke_visual.py` 生成 2 个 fact + 2 个 visual_clip 的统一 evidence 和回答上下文，使用真实 Grove 图与 NumPy 检索；合成报告禁止用于模型选型。新版 CLI `--help` 可在没有 torch 的 CPU 环境读取；实际全 CLI 仍依赖上游缺失的文本索引/builder 模块，preflight 已增加相关报告。GPU/API、真实召回率、视觉答案忠实度仍未验证。
