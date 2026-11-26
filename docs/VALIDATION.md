# 实际验证记录

2026-11-26 兼容主循环更新：当前工作目录 **173 项测试通过（102 组件、71 Grove）**，新增 12 项生产核心组件测试和 5 项端到端测试。新增测试真实执行 ffmpeg、loopback HTTP、原 Grove 分段/聚合/图编排、生产 facts/BM25/embedding/RRF、人物投影和 clip 证据；仅模型服务输出与 GPU 推理边界使用合成数据。验证动态生成 3 个 episode、12 条基础 facts、12 个 clips，以及人物扩展的 6 条 facts、跨 episode 相同 character 和具体语音/视觉证据。另覆盖中途失败回滚、重启幂等、topic 更新、源/cutoff、坏缓存重建和模型版本失配。真实模型/GPU准确率验收尚未执行。部署与调用见 [FULL_LOOP.md](FULL_LOOP.md)。

干净目录部署复验：仅导出仓库文件，从 bundle 离线还原并应用补丁；102 项组件测试、71 项 Grove 测试及 5 条 smoke 全部通过。29 个 Grove 修改文件与当前工作目录逐字节一致，92 个 Python 文件通过 AST 检查，两层 Git diff 空白检查通过。审计脚本为 `scripts/check_clean_export.py`；本机报告为 `runs/full-loop-clean-export.json`（不提交运行产物）。

当前 `test_boundary_hook.py` 直接导入生产 prompts/llm_utils，已移除缺失模块的测试替身。preflight 区分兼容核心可导入、模型配置与可选旧 benchmark 缺口。较早版本的验证记录另存于 [VALIDATION_HISTORY.md](VALIDATION_HISTORY.md)。其中“完整主循环未就绪/缺文本索引”描述当时的状态；真实模型未验证的限制仍然成立。

## 当前可核对的验证范围

| 验证 | 已执行的真实部分 | 模型输出/未验证部分 |
|---|---|---|
| 核心主循环端到端 | 真实视频解码、HTTP、原有动态分段器、兼容摘要/facts/topic 依赖、超图、BM25/向量检索及保存恢复 | HTTP 模型契约服务使用合成返回；不代表真实 VLM 的分段/摘要质量 |
| 人物接入端到端 | 边界后 provider、声音/人脸身份匹配、M3 等价关系解析、grounding API 协议、正文转写、人物投影和 fact→clip 对齐 | GPU 推理边界提供合成脸/声向量和 M3 输出；不代表识别准确率 |
| 失败与恢复 | 摘要/facts 故障不提交半成品，索引故障后重建，恢复不重复写入，坏缓存和模型版本失配处理 | 没有测高并发、多写入者或长期服务 SLA |
| 干净部署 | 仅复制仓库文件，离线还原 Grove bundle、应用补丁、跑组件/接入测试和全部 smoke | 依赖使用本机已安装环境；不等于在目标 Linux/CUDA 服务器完成安装验收 |
| 语音、人脸与证据组件 | 既有单测真实执行分组/匹配、HDBSCAN、共同 PCM 切片、引用校验和指标计算 | Whisper/pyannote/FunASR、InsightFace、ERes2NetV2、M3 权重尚未在本机真实推理 |

## 复验命令

```bash
python -m unittest discover -s tests/modules -t . -v
python -m unittest discover -s tests/grove -t . -v
python scripts/check_clean_export.py --report runs/full-loop-clean-export.json
```

纯文档更新不产生新的模型实验结论。当前实现与前后差异见 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [CHANGES.md](CHANGES.md)；真实 API/GPU 验收步骤见 [FULL_LOOP.md](FULL_LOOP.md)。
