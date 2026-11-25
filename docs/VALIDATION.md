# 实际验证记录

日期：2026-11-25。环境：macOS，Python 3.13.9；隔离环境 NumPy 2.5.5、Pydantic 2.14.4。未运行任何 GPU 模型。

| 验证 | 结果 | 可以说明什么 |
|---|---|---|
| `python -m unittest discover -s tests -t . -v` | 31 项通过：23 项模块测试、8 项 Grove 测试 | 图操作、指标、适配器协议、边界后接入及序列化契约 |
| `scripts/smoke.py` | 生成 snapshot、真实 Grove hypergraph、synthetic report | 假 LLM / 合成向量下，写入路径可跑，结果明确不用于选型 |
| AMI 真实视频 ffmpeg 解码 | 音频长度 60.000 秒 | 媒体可读取；已处理 AAC 解码尾部 padding，否则为 60.032 秒 |
| 无 Gemini 配置/密钥运行 | 抛错并写 status=failed | 缺依赖时不会伪造预测或偷用标注 |
| compileall | 通过 | 新代码及修改的 Grove 文件语法有效 |
| Grove git diff --check | 通过 | 无 diff 空白错误 |
| 清洁 Grove 副本应用补丁并逐文件比较 | 5 个文件完全一致 | 补丁可重现当前实验改动，包括新增模块 |
| Git 待提交文件的全新导出目录部署 | 离线还原 Grove、31 项测试及 smoke 全部通过，两段视频齐全 | 仓库无需原本机兄弟目录或访问 Grove 上游即可初始化实验代码；GPU 环境未验证 |

Grove 测试使用其真实 VideoEpisode/VideoFact、Pydantic 图节点、VideoHypergraph.save/load，以及新增 writer/provider。boundary hook 测试执行真实修改后的 `_finalize_episode`，**仅用测试替身补上公开仓库缺失的 prompt/llm_utils**；它验证调用顺序，不能证明真实模型的动态边界划分准确。

覆盖的失败场景包括：漏项/重复 AV 输出、虚构 face/utterance ID、跨模态 embedding 比较、候选边不合并、历史画外音复识别、未知短发言、图版本/时间范围不符、错误角色归属、错误关联、全拒识导致召回为零、跨边界句子不重复写入。

尚未验证：Gemini 实际 API 响应、Whisper/pyannote/FunASR 的安装和模型推理、InsightFace/ERes2NetV2 输出、Omni GPU 推理、真实 writer 的事实忠实度、真实 ASR/AV/character 准确率、全 Grove topic/retrieval 链路、实时并行调度与显存/费用。

`scripts/preflight.py` 会列出缺失上游文件，当前 full_grove_pipeline_ready=false。没有把测试替身补成生产源码来冒充完整 Grove 已可运行。
