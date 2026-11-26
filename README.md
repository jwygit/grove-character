# Grove Character

在 **Grove 确定 episode 边界后、生成最终摘要和 facts 前** 解析人物身份，把 character 和可回看的视觉/语音证据接入记忆。当前已提供完整本地视频主循环；独立模块实验继续保留。

本目录独立于原来的 `Grove`、`m3-agent`、`person-identity`。`vendor/Grove` 是固定上游版本的实验副本，已实际修改；可移植补丁在 `patches/grove-character.patch`，来源与版本在 `sources.lock.json`。

## 文档从这里看

| 想了解什么 | 文档 |
|---|---|
| 原版有什么、缺什么；此前已做什么、本次补了什么 | [改动说明](docs/CHANGES.md) |
| 当前架构、实际执行顺序、各层数据分工，以及 fact 如何对应人物和 clips | [架构与数据流程](docs/ARCHITECTURE.md) |
| 内网要准备哪些模型/镜像，先核心再人物的部署与测试命令 | [内网 GPU 部署](docs/OFFLINE_DEPLOYMENT.md) |
| 上传视频、阶段实时输出、人物关联和带引用问答 | [工作台部署与使用](docs/WORKBENCH.md) · [设计与素材](docs/WORKBENCH_DESIGN.md) |
| 在 GPU/API 服务器部署，先跑核心、再启用人物，或升级已有副本 | [完整主循环手册](docs/FULL_LOOP.md) |
| 分开测试 ASR、脸声关联、character 归属，如何控制变量和计分 | [实验设计](docs/DESIGN.md) / [独立模块运行手册](docs/RUNBOOK.md) |
| 人物图、视觉检索和事实证据的具体字段/接口 | [人物图](docs/CHARACTER_GRAPH.md) / [视觉检索](docs/VISUAL_RETRIEVAL.md) / [fact→clip](docs/FACT_CLIP_EVIDENCE.md) |
| 实际测过什么、哪些代码来自上游 | [测试记录](docs/VALIDATION.md) / [来源与差异](docs/PROVENANCE.md) |

## 完整 Grove 主循环

新增 `python -m lab.full_loop`：本地视频 → 实际抽帧 → API 帧描述 → Grove 动态分段 → 摘要 → facts/topic → 原有超图 → BM25/语义/visual clip 联合检索。核心缺失模块已用生产兼容实现补齐，人物模块在边界确定后接回。部署、配置、恢复和验证范围见 [完整主循环手册](docs/FULL_LOOP.md)，配置见 [configs/full_loop.json](configs/full_loop.json)。

保留 Grove 的构建器、分段器、Topic 聚合器、图结构和检索编排；新增实现不冒充上游未公开的源码或论文复现。端到端测试使用真实视频解码、HTTP、生产模块及索引，仅模型输出为明确测试数据。本机仍未执行真实 GPU 模型准确率验收。

```text
本地视频 → 抽帧与 VLM 描述 → Grove 动态 episode
                                  ↓
                 可选：按已定边界做人脸/声音关联与视觉定位
                                  ↓
              episode 摘要 → facts 与 topic → 图、人物投影、clip 证据
                                  ↓
                  checkpoint → 索引 → fact / visual clip 联合检索
```

启用人物时，整段 ASR 在窗口循环前准备；当前人脸/声纹取证在 episode 关闭后顺序执行。核心 API 示例默认开启视觉通道，人物模块另行配置。`--query` 返回检索证据，尚不自动生成自由文本答案。

## 代码与测试入口

| 部分 | 实现 | 验证内容 |
|---|---|---|
| 独立模块 | `lab/speech.py`、`perception.py`、`association.py`、`m3.py`、`metrics.py` | Gemini / 本地语音前端、episode 级音画关联、跨 episode 人物图 |
| 接入 Grove | `vendor/Grove/hypervideo/modules/character_identity.py` 及补丁；`lab/grove.py` | 边界后注入、摘要/事实证据引用、图写入与重载 |
| 完整主循环 | `lab/full_loop.py`、`lab/stream_identity.py`、Grove `streaming_memory.py` | 实际视频/API、动态分段、facts/topic、自动接回人物和证据、恢复与联合检索 |
| 独立模块单测 | `tests/modules/` | 分段污染、画外音、误关联、拒识、跨边界与人物 ID |
| Grove 接入/端到端测试 | `tests/grove/` | 实际 finalize 调用顺序、证据限制、图序列化、完整主循环及失败恢复 |

详细设计见 [实验设计](docs/DESIGN.md)，GPU/API 操作见 [运行手册](docs/RUNBOOK.md)，复用关系见 [代码来源](docs/PROVENANCE.md)，实际验证见 [测试记录](docs/VALIDATION.md)。

已补齐 clip 的 `visible`/`speaker` 人物引用、人物反向索引与并行召回，以及身份合并/撤销后的正文和索引重建。独立命令、Grove 接口与升级方式见 [人物图与检索](docs/CHARACTER_GRAPH.md)。运行 `python scripts/smoke_characters.py runs/characters-smoke-new` 可检查修订前后的图与召回结果。

进一步补齐 **fact→具体 clip 的证据关联**：语音沿用引用发言的时间，视觉通过独立 `ground-visual` 定位到时间段/实际帧；writer 引用后生成可重建的证据边，检索返回具体 clip 与时间。部署与独立测试见 [Fact/clip 证据说明](docs/FACT_CLIP_EVIDENCE.md)。M3 原始联合提示保持不变，无法定位的事件明确保留为未定位。

## Visual clip 联合检索

已补齐 Grove `use_visual_channel` 的 TODO，并接到 `hierarchical_v2_day` 和核心 `HierarchicalRetriever`。visual clip 与 fact 分别检索后共享一个 top-k 列表，保留各自类型、ID、时间、父 episode 和分数；没有 fact 的 episode 也可返回视觉证据。视觉编码可调用远程 API，本机只计算 cosine。

开启配置见 [configs/visual_retrieval.json](configs/visual_retrieval.json)，运行方式与接口见 [视觉检索说明](docs/VISUAL_RETRIEVAL.md)。本机可运行 `.venv/bin/python scripts/smoke_visual.py runs/visual-smoke-new`；这是合成向量与测试用文本索引下的真实检索路径验证，不代表模型准确率。新主循环已使用补齐的真实文本索引；该旧 smoke 仍保留为检索组件回归测试。

## 当前离线流程与模型

1. `speech`：对完整测试片段做语音分段和转录。分别运行 Gemini、faster-whisper large-v3 + pyannote community-1，或中文 FunASR（Paraformer-zh + FSMN-VAD + ct-punc + CAM++），不是依次串联三组模型。
2. 独立组件实验可提供固定 episode 时间范围；完整流程使用 `lab.full_loop` 让 Grove 根据帧描述动态确定边界，不需预先提供 episode。
3. `perceive`：按 episode 裁剪音视频，用 InsightFace buffalo_l 按 5 FPS 提取人脸，执行 M3 原始 HDBSCAN、质量筛选及每簇最多 3 张代表脸选择；用 ERes2NetV2 提取每段合格语音的声纹。
4. `associate` 先建立人物证据节点：人脸按多代表脸平均余弦相似度匹配（默认 0.3，最多 10 个特征）；声音按 M3 的节点平均相似度匹配（默认 0.6，最多 20 条声纹）。相同声音的发言先归为同一个 `voice_id`，再进入模型；用 `--prior` 接入历史快照。
5. 音画关联与记忆联合生成：`configs/av_omni.json` 默认使用 **[ByteDance-Seed/M3-Agent-Memorization](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)** 微调权重和 M3 原始联合提示，输入视频、代表脸、按声音节点汇总的发言，输出事件描述、高层结论和 `Equivalence: <face_N>, <voice_N>`。`--accept-model-links` 才把预测等价关系用于人物合并；图快照保存原始输出与映射版本。
6. `grove-write`：把原始描述、特征标签→character 映射和发言证据注入摘要/fact writer。代码按同一份映射确定性替换正文中的 `<face_N>/<voice_N>`，写入 `character_refs`，保存原文及 `identity_evidence`，再验证 Grove 图重载一致。

M3 原始输出没有逐句露脸判断，本项目不再为它虚构此标签；用新增 `evaluate-equivalence` 评估脸声身份关系，用 `evaluate-characters` 评估写入归属。旧的逐句 AV 协议仅供显式使用旧适配器时调用，不能拿它的可见说话人指标评判 M3 等价关系。

与 M3 的适配差异仍保留：按 Grove episode 取证、稳定人物 ID 和图版本、在**写入阶段**转写 character（M3 在检索阶段转换）。没有复制其完整记忆检索图、语义权重及冲突消解策略。新主循环已验证动态边界后等待取证的集成路径；本机尚未执行真实 GPU 模型，取证当前顺序运行，实时并发调度仍未验证。

所有语音前端统一从同一份 16kHz 单声道 PCM 切片提声纹，不再从重新编码的视频音轨提取。统一采用 >=2 秒时长门限，不根据某个前端独有的 overlap 标签过滤。短句文字保留到 Grove，但不创建声纹身份、不参加音画身份关联。升级后需重新运行 `perceive` 和 `associate`，旧版 bundle 会被明确拒绝。

## 本机立即可跑

在本项目目录执行：

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/smoke.py runs/smoke-new
```

新机器先安装 `requirements-workbench.txt`；运行 `python scripts/setup_grove.py`，会从仓库内的 `upstream/Grove.bundle` 离线还原固定版本、验证校验和并应用补丁，无需再访问 Grove 上游。现有目录不会被覆盖。两段 AMI 测试视频也随仓库提供。

已有服务器仅执行 `git pull` 不会更新被忽略的 `vendor/Grove`。请按 [升级步骤](docs/FULL_LOOP.md#已有部署如何升级) 先保留旧副本，再重建本次补丁对应的 vendor。

从 GitHub 部署：

```bash
git clone https://github.com/jwygit/grove-character.git
cd grove-character
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-workbench.txt
python scripts/setup_grove.py
python -m unittest discover -s tests -t . -v
python scripts/preflight.py
```

仓库为私有，需要具备 GitHub 访问凭据。内网无法访问 PyPI/模型仓库时，依赖 wheel 和模型权重需提前下载或使用内部镜像；见运行手册。以上先验证 CPU 测试，不会自动安装 GPU 模型。

`smoke` 使用人工构造的向量、关联和假 LLM，执行真实 Grove 数据模型及图保存/读取；结果明确标为 synthetic，不能据此选模型。真实推理命令不会退回假数据。

## 已知边界

- 公开 Grove 的核心缺失模块已在实验副本中兼容实现。标准 QA 主循环通过工程端到端测试；旧 VideoRAG 总入口、部分 benchmark 异步/音频/entity/profile 扩展仍缺上游实现，不属于本次支持范围。
- 本机没有 GPU/模型权重，也未提供 Gemini 或 writer API 凭据。模型适配代码已写，**没有真实模型准确率或显存结论**。
- 发言证据产生 `speaker` 引用；正文中的已知人物标签产生 `referenced` 引用，不自动解释为视觉 `actor` 或被谈论的 `mentioned`。语义归属及事实忠实度仍需人工标注测试。
- 两段 AMI 视频已复制到 `data/media/`，是真实英语会议视频，只适合初步联调；标注模板尚未人工复核。FunASR 中文方案需另备中文数据，不与英语 AMI 直接做排名。
