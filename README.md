# Grove Character

目标：并行提取人脸/声音证据，在 **Grove 确定 episode 边界后、生成最终摘要和 facts 前** 解析人物身份。先测组件，再测记忆写入的 character 归属。

本目录独立于原来的 `Grove`、`m3-agent`、`person-identity`。`vendor/Grove` 是固定上游版本的实验副本，已实际修改；可移植补丁在 `patches/grove-character.patch`，来源与版本在 `sources.lock.json`。

## 两部分代码

| 部分 | 实现 | 验证内容 |
|---|---|---|
| 独立模块 | `lab/speech.py`、`perception.py`、`association.py`、`m3.py`、`metrics.py` | Gemini / 本地语音前端、episode 级音画关联、跨 episode 人物图 |
| 接入 Grove | `vendor/Grove/hypervideo/modules/character_identity.py` 及补丁；`lab/grove.py` | 边界后注入、摘要/事实证据引用、图写入与重载 |
| 独立模块单测 | `tests/modules/` | 分段污染、画外音、误关联、拒识、跨边界与人物 ID |
| Grove 单测 | `tests/grove/` | 实际 finalize 调用顺序、证据限制、真实 Grove 图序列化 |

详细设计见 [实验设计](docs/DESIGN.md)，GPU/API 操作见 [运行手册](docs/RUNBOOK.md)，复用关系见 [代码来源](docs/PROVENANCE.md)，实际验证见 [测试记录](docs/VALIDATION.md)。

## 当前离线流程与模型

1. `speech`：对完整测试片段做语音分段和转录。分别运行 Gemini、faster-whisper large-v3 + pyannote community-1，或中文 FunASR（Paraformer-zh + FSMN-VAD + ct-punc + CAM++），不是依次串联三组模型。
2. 提供固定 episode 时间范围：可以来自已有 Grove 结果，联调也可以人工指定；当前独立入口不会自动跑完整 Grove 动态分段。
3. `perceive`：按 episode 裁剪音视频，用 InsightFace buffalo_l 按 5 FPS 提取人脸，执行 M3 原始 HDBSCAN、质量筛选及每簇最多 3 张代表脸选择；用 ERes2NetV2 提取每段合格语音的声纹。
4. `associate` 先建立人物证据节点：人脸按多代表脸平均余弦相似度匹配（默认 0.3，最多 10 个特征）；声音按 M3 的节点平均相似度匹配（默认 0.6，最多 20 条声纹）。相同声音的发言先归为同一个 `voice_id`，再进入模型；用 `--prior` 接入历史快照。
5. 音画关联与记忆联合生成：`configs/av_omni.json` 默认使用 **[ByteDance-Seed/M3-Agent-Memorization](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)** 微调权重和 M3 原始联合提示，输入视频、代表脸、按声音节点汇总的发言，输出事件描述、高层结论和 `Equivalence: <face_N>, <voice_N>`。`--accept-model-links` 才把预测等价关系用于人物合并；图快照保存原始输出与映射版本。
6. `grove-write`：把原始描述、特征标签→character 映射和发言证据注入摘要/fact writer。代码按同一份映射确定性替换正文中的 `<face_N>/<voice_N>`，写入 `character_refs`，保存原文及 `identity_evidence`，再验证 Grove 图重载一致。

M3 原始输出没有逐句露脸判断，本项目不再为它虚构此标签；用新增 `evaluate-equivalence` 评估脸声身份关系，用 `evaluate-characters` 评估写入归属。旧的逐句 AV 协议仅供显式使用旧适配器时调用，不能拿它的可见说话人指标评判 M3 等价关系。

与 M3 的适配差异仍保留：按 Grove episode 取证、稳定人物 ID 和图版本、在**写入阶段**转写 character（M3 在检索阶段转换）。没有复制其完整记忆检索图、语义权重及冲突消解策略。本机尚未执行真实 GPU 模型；并行取证和动态边界后的实时等待仍需完整集成验证。

所有语音前端统一从同一份 16kHz 单声道 PCM 切片提声纹，不再从重新编码的视频音轨提取。统一采用 >=2 秒时长门限，不根据某个前端独有的 overlap 标签过滤。短句文字保留到 Grove，但不创建声纹身份、不参加音画身份关联。升级后需重新运行 `perceive` 和 `associate`，旧版 bundle 会被明确拒绝。

## 本机立即可跑

在本项目目录执行：

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/smoke.py runs/smoke-new
```

新机器先安装 `requirements-test.txt`；运行 `python scripts/setup_grove.py`，会从仓库内的 `upstream/Grove.bundle` 离线还原固定版本、验证校验和并应用补丁，无需再访问 Grove 上游。现有目录不会被覆盖。两段 AMI 测试视频也随仓库提供。

从 GitHub 部署：

```bash
git clone https://github.com/jwygit/grove-character.git
cd grove-character
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
python scripts/setup_grove.py
python -m unittest discover -s tests -t . -v
python scripts/preflight.py
```

仓库为私有，需要具备 GitHub 访问凭据。内网无法访问 PyPI/模型仓库时，依赖 wheel 和模型权重需提前下载或使用内部镜像；见运行手册。以上先验证 CPU 测试，不会自动安装 GPU 模型。

`smoke` 使用人工构造的向量、关联和假 LLM，执行真实 Grove 数据模型及图保存/读取；结果明确标为 synthetic，不能据此选模型。真实推理命令不会退回假数据。

## 已知边界

- 当前 Grove 公开提交缺少 `fact_extractor.py`、`llm_utils.py`、episode/topic prompts 以及 audio_cues 的实现文件。因此完整 Grove 动态边界到检索的端到端运行尚不能验证。这里提供的是可运行的隔离写入路径与真实代码的接入补丁。
- 本机没有 GPU/模型权重，也未提供 Gemini 或 writer API 凭据。模型适配代码已写，**没有真实模型准确率或显存结论**。
- 发言证据产生 `speaker` 引用；正文中的已知人物标签产生 `referenced` 引用，不自动解释为视觉 `actor` 或被谈论的 `mentioned`。语义归属及事实忠实度仍需人工标注测试。
- 两段 AMI 视频已复制到 `data/media/`，是真实英语会议视频，只适合初步联调；标注模板尚未人工复核。FunASR 中文方案需另备中文数据，不与英语 AMI 直接做排名。
