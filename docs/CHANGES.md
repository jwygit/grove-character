# 从原版 Grove 到当前项目：补了什么

对照基准是仓库锁定的 Grove 提交 `5e437cff027ab7df969c17bdb8fc243a6e70961d`，不代表作者私有环境或所有后续版本。当前完整主循环实现提交为 [`ca7222e`](https://github.com/jwygit/grove-character/commit/ca7222ee0325b13a698d91f71e9c28bfef829bf0)。

后续内网部署补充：增加 `lab.visual_server` 本地 SigLIP 服务，修正此前误以为上游已经提供可启动 encoder server 的描述；增加分开的 core/visual/ASR/character Docker 配方、离线资产准备与分阶段配置生成器。新增 5 项视觉服务测试，总计 178 项；GPU 镜像构建、真实权重推理和准确率仍待目标服务器验收。操作见 [内网部署手册](OFFLINE_DEPLOYMENT.md)。下表保留主循环版本 `ca7222e` 的前后对照。

需要分清三个阶段：**原版公开代码已有的架构**、**此前已完成的人物/视觉组件**，以及**本次把这些组件串起来的兼容主循环**。人物和视觉功能并非这次才从零实现。

## 1. 总体前后对照

| 能力 | 固定的原版 Grove | 此前项目，至 `4dd2a8e` | 当前，`ca7222e` |
|---|---|---|---|
| 动态 episode 组织 | 已有 Segmenter 和 Builder 代码，但引用的提示词/抽取工具等缺失 | 能测试边界后 hook，不能据此证明完整主循环可运行 | 补齐依赖，真实调用原 Segmenter 按 caption 流动态关闭 episode |
| episode 摘要 | 已有调用位置，缺公开 prompt/LLM utility | 人物独立 writer 可写摘要；边界 hook 测试曾临时注入缺失依赖 | 兼容 prompt + 重试/严格失败处理，完整循环直接使用生产模块 |
| fact 抽取 | Builder 有调用与入图流程，`fact_extractor.py` 缺失 | 已有证据约束的 character writer，按给定 episode 验证写入 | 新增基础 fact extractor；无人物和有人物两种写入路径都可接主循环 |
| topic 聚合 | 已有匹配、创建、更新与超边编排，缺 prompt/utility | 人物隔离写入不模拟 topic 聚合 | 调用真实 TopicAggregator，验证新建、更新与角色分配 |
| 文本检索 | 已有 DualIndex 与层级检索编排，BM25/embedding/RRF 实现缺失 | 检索组件测试使用明确的测试索引 | 新增实际 BM25、向量索引、超边聚合与 RRF，支持保存、重载和真实查询计算 |
| 人物身份 | 本项目所需的 M3 脸声映射与写入 character 不是该基线已完成的能力 | 已有独立 ASR、脸/声取证、M3 联合记忆、身份图和 writer | 在动态边界后自动调用已有组件，并用已提交身份图作为 prior |
| visual clip 联合检索 | 有视觉索引/节点基础，目标联合分支留有 TODO | 已补齐 fact/clip 统一结果、人物并行召回 | 完整循环自动建立 clips；开启视觉通道时编码实际帧，并接真实文本索引 |
| fact→clip 证据 | 没有本项目新增的发言/视觉证据字段与对齐链 | 已有语音时间、视觉 grounding、证据边及测试 | 自动在完成 episode 后重建证据边；基础无人物 fact 也需引用采样帧 |
| 主循环恢复 | 原 Builder 有 buffer 保存/恢复机制 | 独立模块各自落盘，尚未验证完整链路事务 | 增加窗口级原子 checkpoint、幂等恢复、输入/模型配置校验及坏索引重建 |
| 验证 | 不能仅凭原公开文件声称整条链可运行 | 156 项组件/接入测试，隔离能力可测 | 173 项测试及干净部署复验，包含完整工程端到端路径；真实模型准确率仍待测 |

原版并非没有动态分段、topic 或检索设计。问题是这些真实编排模块依赖了公开提交里不存在的文件，所以此前不能把组件测试当成完整系统运行证明。

## 2. 原来具体缺了哪些核心文件

下表文件在固定上游提交中不存在，调用方却引用它们；本次在 Grove 补丁中新增了可执行实现。

| 文件，相对 `vendor/Grove/` | 现在实现的内容 |
|---|---|
| `hypervideo/extractors/llm_utils.py` | JSON 解析、重试、统计与严格失败上下文；主循环耗尽重试后报错 |
| `hypervideo/prompts/episode_prompts.py` | 与现有 segmenter 格式参数兼容的边界与摘要提示词 |
| `hypervideo/prompts/topic_prompts.py` | 主题匹配、新建、更新及 episode 角色提示词 |
| `hypervideo/extractors/fact_extractor.py` | 基础事实抽取、frame ID 引用校验、角色/权重、可定位视觉点证据 |
| `hypervideo/extractors/fact_deduplicator.py` | 旧非 stream 路径的保守异步去重；拒绝跨事件时间合并，排除身份/证据事实。标准 stream 路径沿用关闭跨 episode 去重的行为 |
| `hypervideo/index/bm25_index.py` | 分层 BM25、Unicode/CJK 分词、候选内排序、保存与重载；历史 `.pkl` 文件名内保存版本化 JSON |
| `hypervideo/index/embedding_index.py` | 调用真实 embedding callable、归一化、超边聚合/配置的传播、候选内 cosine 检索 |
| `hypervideo/retrieval/rrf_fusion.py` | 按排名融合召回结果，避免不同通道原始分数直接相加 |

这些是**根据公开接口新写的兼容实现**。提示词、tokenization、抽取细节与传播实现不能被表述成找回了原作者源码，也没有“与论文数值等价”的结论。基础事实 prompt 要求引用采样帧，是本项目为了证据可追踪增加的约束。

## 3. 保留、修改和新增的边界

| 分类 | 代码 | 本次作用 |
|---|---|---|
| 保留架构并补接口行为 | Grove `VideoEpisodeSegmenter`、`VideoTopicAggregator`、`VideoHypergraphBuilder` | 原有动态活动组织、主题编排与入图关系继续使用；补 source 传递、响应校验和严格失败传播 |
| 保留数据模型 | Grove `VideoHypergraph`、原有 Fact/Episode/Topic、visual clip 与超边 | 延续此前已增加的人物/证据字段，没有另建替代记忆数据库 |
| 保留检索编排 | Grove `DualIndex`、`HierarchicalRetriever`、此前的 multimodal 叶节点检索 | 使用补齐的真实索引，延续已有 fact/clip 联合结果和人物召回 |
| 新增主循环 | `hypervideo/streaming_memory.py` | 将真实 Builder、图、投影和检索接成窗口事务，管理 checkpoint 与索引缓存 |
| 新增视频/API 入口 | `lab/full_loop.py` | 读视频、实际 PTS 抽帧、VLM caption、视觉编码、顺序喂窗口、finish 和查询 |
| 新增人物自动接入 | `lab/stream_identity.py` | 共同 PCM/ASR 准备，边界后调用现有 perception/M3/grounding，管理身份快照和已提交 prior |
| 延用此前独立模块 | `lab/speech.py`、`perception.py`、`association.py`、`m3.py`、`visual_grounding.py` 等 | 模型职责与基本关联策略沿用；此次把手动阶段衔接到动态主循环 |
| 新增部署验证 | `configs/full_loop.json`、`scripts/check_clean_export.py`、`scripts/smoke_full_loop.py` | 可配置的真实入口、仓库独立还原验证及明确标注合成模型的示例 |

详细当前流程和每层数据职责见 [ARCHITECTURE.md](ARCHITECTURE.md)。代码安装后位于 `vendor/Grove/`；这些文件通过 [补丁](../patches/grove-character.patch) 交付，所以 GitHub 主仓库没有直接展示整个 vendor 目录。

## 4. 此前已经完成的改动

| 提交 | 当时补齐的内容 |
|---|---|
| `d85a3d3` | 对齐 M3 联合记忆输入/提示词与脸声等价关系；Grove 在写入阶段使用 character 映射 |
| `135843c` | visual clip 与 fact 联合检索，补目标视觉分支与候选过滤 |
| `f90bd4f` | 完整人物时间线到 clip 引用、人物反向索引、并行召回，以及合并/撤销后的投影重建 |
| `4dd2a8e` | fact 按发言时间和视觉区间/帧关联具体 clips，保留未定位状态与来源校验 |
| `ca7222e` | 补齐缺失核心模块，跑通标准动态主循环，自动接回以上组件并完成端到端契约测试 |

更早的 ASR 对比、InsightFace/HDBSCAN、ERes2NetV2、跨片段匹配与测试视频准备也继续保留；原兄弟目录 `Grove`、`m3-agent`、`person-identity` 不是部署运行依赖。

## 5. 测试变化和仍待完成的工作

本次主循环实现新增 **12 项核心组件测试 + 5 项端到端测试**，总计从 156 项增加到 173 项。原 `test_boundary_hook.py` 已移除给缺失 prompts/utility 注入临时模块的做法。生产入口不导入 `tests`，生产核心文件不返回预写示例结果。

端到端测试中，ffmpeg、HTTP、动态 segmenter、facts/topic 编排、图持久化、BM25/向量计算和 evidence 检索均实际执行；模型服务返回合成预测，人物用例的 GPU 推理边界也使用明确的合成观测。报告禁止用于模型选型。干净目录复验覆盖 173 项测试和 5 条 smoke，证明仓库 bundle/补丁足以重建已测代码。

| 当前状态 | 内容 |
|---|---|
| 已实现并通过工程测试 | 标准 QA stream 主循环，边界后人物接入，事实/视觉证据与联合检索，保存与恢复 |
| 有代码，待真实环境验收 | 配置模型 API、本地 ASR/脸声/M3 权重推理，真实视频上的完整链路 |
| 尚无实测结论 | ASR/脸声/character 准确率、检索质量、延迟、显存、资源成本、论文复现指标 |
| 未纳入支持范围 | 旧 VideoRAG 总入口及部分 benchmark/async/audio_cues/entity/profile 缺失模块；不阻塞新标准入口，也没有用空壳补齐 |
| 后续能力 | 实时采集上传、并行取证、多源统一人物图库、人工身份审核图导入主循环 checkpoint |

GPU 服务器操作见 [FULL_LOOP.md](FULL_LOOP.md)，三阶段模型对比方法见 [DESIGN.md](DESIGN.md)，实际测试记录见 [VALIDATION.md](VALIDATION.md)。
