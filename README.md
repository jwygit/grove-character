# Grove Character

目标：并行提取人脸/声音证据，在 **Grove 确定 episode 边界后、生成最终摘要和 facts 前** 解析人物身份。先测组件，再测记忆写入的 character 归属。

本目录独立于原来的 `Grove`、`m3-agent`、`person-identity`。`vendor/Grove` 是固定上游版本的实验副本，已实际修改；可移植补丁在 `patches/grove-character.patch`，来源与版本在 `sources.lock.json`。

## 两部分代码

| 部分 | 实现 | 验证内容 |
|---|---|---|
| 独立模块 | `lab/speech.py`、`perception.py`、`association.py`、`metrics.py` | Gemini / 本地语音前端、episode 级音画关联、跨 episode 人物图 |
| 接入 Grove | `vendor/Grove/hypervideo/modules/character_identity.py` 及补丁；`lab/grove.py` | 边界后注入、摘要/事实证据引用、图写入与重载 |
| 独立模块单测 | `tests/modules/` | 分段污染、画外音、误关联、拒识、跨边界与人物 ID |
| Grove 单测 | `tests/grove/` | 实际 finalize 调用顺序、证据限制、真实 Grove 图序列化 |

详细设计见 [实验设计](docs/DESIGN.md)，GPU/API 操作见 [运行手册](docs/RUNBOOK.md)，复用关系见 [代码来源](docs/PROVENANCE.md)，实际验证见 [测试记录](docs/VALIDATION.md)。

## 当前离线流程与模型

1. `speech`：对完整测试片段做语音分段和转录。分别运行 Gemini、faster-whisper large-v3 + pyannote community-1，或中文 FunASR（Paraformer-zh + FSMN-VAD + ct-punc + CAM++），不是依次串联三组模型。
2. 提供固定 episode 时间范围：可以来自已有 Grove 结果，联调也可以人工指定；当前独立入口不会自动跑完整 Grove 动态分段。
3. `perceive`：按 episode 裁剪音视频，用 InsightFace buffalo_l 检测/提取人脸特征并聚类，用 ERes2NetV2 提取每段合格语音的声纹。
4. `associate`：按 `configs/av_omni.json` 使用 **Qwen2.5-Omni-7B**，输入带音轨视频、人脸参考图、发言时间与转录，预测每段发言对应的 face_id 或 null。这一步才是脸与声音的跨模态关联；也可切换 Gemini 作为对照。
5. 人物图：保存关联候选；使用 `--accept-model-links` 才按实验策略确认 face/voice 边。历史身份重识别还需传入 `--prior` 并显式设置 `reid_thresholds`，在同模态内比较特征，不直接比较脸向量和声音向量。
6. `grove-write`：将身份快照注入固定 episode，用配置的 writer 生成摘要与 facts，按其引用的 utterance_ids 写入 character_refs，并保存/重载 Grove 图。

Omni 默认配置是基础 Qwen2.5-Omni-7B，**不是 M3-Agent 微调权重**。借鉴的是 M3 的处理分工；当前模型适配器尚未做真实 GPU 准确率验证。并行取证、动态边界后的实时等待属于完整接入目标，不能等同于当前离线命令已经完成实时调度。

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
- 本版自动写入的是有发言证据的 `speaker` 引用。视觉 `actor` 和被提到的 `mentioned` 不能由说话人身份推导；指标支持按角色评测，但自动视觉 actor 解析尚未实现。
- 两段 AMI 视频已复制到 `data/media/`，是真实英语会议视频，只适合初步联调；标注模板尚未人工复核。FunASR 中文方案需另备中文数据，不与英语 AMI 直接做排名。
