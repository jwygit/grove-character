# Grove 兼容主循环

本版本在固定 Grove 副本内补齐生产模块，不再通过 `sys.modules` 注入缺失模块。新增 `StreamingMemory` 负责事务、缓存和视频源状态，实际调用原有 `VideoHypergraphBuilder`、`VideoEpisodeSegmenter`、`VideoTopicAggregator`、`VideoHypergraph`、`DualIndex` 和 `HierarchicalRetriever`。没有另造一套记忆图。

这些是根据公开调用接口编写的**兼容实现**，不是恢复作者未发布的源码/提示词，也不承诺复现论文结果。标准 QA 主循环已可运行；旧 VideoRAG 总入口、EgoLife 累积事件、部分异步 benchmark、entity/profile 扩展仍有上游缺口，不用空壳模块冒充支持。

## 数据流程

```text
本地视频 → ffmpeg 实际 PTS 抽帧 → VLM 帧描述
                                    ↓
                             Grove stream 动态边界
                                    ↓ episode 已结束
                  可选：脸/声取证 → M3 描述与关联 → 视觉定位
                                    ↓
                             Grove episode 摘要
                                    ↓
                     facts 抽取 ║ topic 匹配/创建/更新
                                    ↓
                        原有节点、超边和双向引用
                                    ↓
                    clip 建立、character 投影、fact 证据边
                                    ↓
                         原子 checkpoint → 双索引
                                    ↓
                topic → episode → fact / visual clip 联合检索
```

不启用人物时，facts 必须引用输入帧 ID。代码保留该帧实际解码时间、文件和 SHA256，映射成点证据；不会把一张静态图自动扩成整段动作时长。启用人物后，沿用现有 M3 writer：语音引用 utterance ID 并用真实发言区间；视觉引用 memory ID，再使用独立 grounding 得到的区间或帧。时间交集只能说明去哪里检查，不能证明事实语义正确。

每个 caption frame 对应一个可回看的 clip 时间窗，clip 与 episode 通过已有 window 超边连接。`use_visual_channel=true` 时使用与查询相同的远程视觉 encoder 编码实际帧，视觉 clip 和 fact 分享检索 top-k；没有把 caption 的文本 embedding 冒充图像 embedding。

## 最小部署：先不启用人物

在项目目录执行现有安装和 `scripts/setup_grove.py`。核心 API 路径只需 `requirements-test.txt`、ffmpeg/ffprobe，以及已部署的模型服务；不会在运行中下载权重。配置参考 `configs/full_loop.json`，填写真实模型 ID、部署/权重版本，禁止保留 `SET_`。

需要的服务：

| 配置 | 实际接口/用途 |
|---|---|
| caption | OpenAI-compatible `/v1/chat/completions`，支持多图 `image_url` 与结构化 JSON |
| writer | 同类文本聊天接口；动态边界、摘要、facts、topic 共用，可与 caption 部署相同模型 |
| embedding | `/v1/embeddings`，逐输入返回 `index` 和向量，构建/查询使用同一模型版本 |
| visual_encoder | Grove 已有 server 的 `/health`、`/encode_image_batch`、`/encode_text`；backbone 和维度须与配置一致 |

`visual_encoder.dim` 的示例值不是自动识别的部署参数，须按实际 `/health` 填写。需要先验证纯文本核心时可显式把 `grove.use_visual_channel` 改为 false，clip 和证据引用仍保存。模型不支持 `response_format` 时可在对应聊天配置设置 `json_mode=false`，输出仍必须通过 JSON/schema 校验。

```bash
cp configs/full_loop.json configs/full_loop.local.json
# 编辑本地配置后：
python scripts/preflight.py --config configs/full_loop.local.json
python -m lab.full_loop \
  --video data/media/IS1009a_01.mp4 \
  --config configs/full_loop.local.json \
  --output runs/ami-full-core --source IS1009a-central-01 \
  --query 'What objects are on the table?'
```

`preflight` 区分核心可导入、模型配置与可选旧 benchmark 缺失文件；不会仅因配置填写完成就声称服务器/GPU已验证。

产物：

- `checkpoint.json`：权威恢复点，包含完整图、待分段 buffer、已提交窗口及帧证据；每个窗口成功才原子替换。
- `hypergraph.json`、`builder_state.json`：Grove 格式导出。
- `frames.json`、`caption-*.json`：实际抽帧与模型描述缓存。
- `indices/<id>/`：真实 BM25、三层语义向量和 visual NPZ；`index_manifest.json` 记录图/配置指纹及文件校验和。
- `run_report.json`：图数量与完整 typed retrieval evidence。`accuracy_measured=false`，不会把运行成功当作准确率。

重复执行同一命令会恢复，已提交窗口不会重新请求模型。`--stop-after-windows 2` 可停在检查点、下次继续。模型版本、输入视频 SHA 或构建配置改变应使用新输出目录；不同人物实验也使用不同目录。不要手改 checkpoint 或把一次运行的中间文件与另一目录混用。

摘要/facts/topic/角色分配失败时当前窗口回滚；索引失败保留已经成功的图，下次只重建索引。缓存损坏会重建，不会加载不完整的双索引。一个目录只允许一个写入进程。此驱动是本地视频顺序处理，不是摄像头网络传输服务；当前只管理一个 source 的连续输入。大型视频的帧数上限须显式配置，不能静默丢帧。

## 再开启人物

在同一配置中增加 `characters`，内容使用已有独立模块配置，以下脚本只组装 JSON；各配置文件中的路径、版本和服务地址须先填好。

```bash
python - <<'PY'
import json
from pathlib import Path
def read(name):
    return json.loads(Path('configs', name).read_text())
config = read('full_loop.local.json')
config['characters'] = {
    'speech': read('whisper_pyannote.json'),
    'perception': read('perception.json'),
    'association': read('av_omni.json'),
    'grounding': read('visual_grounding.json'),
    'accept_links': False,
}
Path('configs/full_loop.characters.local.json').write_text(json.dumps(config, indent=2))
PY
python -m lab.full_loop \
  --video data/media/IS1009a_01.mp4 \
  --config configs/full_loop.characters.local.json \
  --output runs/ami-full-characters --source IS1009a-central-01 \
  --query 'Who spoke, and what visual evidence supports the observations?'
```

`speech` 可换为 Gemini 或 FunASR 配置。ASR 先处理同一份 16kHz PCM；边界确定后才裁取属于该 episode 的发言，保留已有跨边界/短句规则。也可用 `characters.speech_input` 指向独立语音实验的 JSON，必须带与该 PCM 一致的 `audio_sha256`，不会把不明来源的转录当作同一视频。

`perception` 默认实际执行 InsightFace/HDBSCAN/ERes2NetV2，`association` 默认实际运行本地 M3-Agent-Memorization，`grounding` 使用已有多图 API。它们仍需要 GPU 环境/相应依赖；没有新增伪造输出的运行分支。`accept_links=false` 保留候选等价边；只有显式设为 true 才按实验策略采纳模型关联，不等于人工确认准确。

新增 `lab/stream_identity.py` 自动衔接这些已有模块。它用**已提交 episode** 的人物图作为 prior；失败窗口产生的临时推理文件不能成为历史身份状态。`identity-*.json` 保存快照、来源、模型配置及 prior 哈希。摘要与 facts 使用验证后的同一份人物映射，随后调用现有 projection 和 fact→clip 代码。跨视频共享人物图库和多源并发接入可在此 provider 接口上继续扩展，本驱动不自动合并不同运行目录。

## 测试与验收边界

```bash
python -m unittest tests.modules.test_compatible_core -v
python -m unittest tests.grove.test_full_loop tests.grove.test_full_loop_characters -v
python -m unittest discover -s tests -t . -v
python scripts/smoke_full_loop.py runs/full-loop-synthetic
# 重新验证仓库可独立部署（需允许本机 loopback 测试服务）：
python scripts/check_clean_export.py --report runs/full-loop-clean-export.json
```

端到端测试真实解码带音轨视频，启动仅绑定 `127.0.0.1` 的 HTTP 模型契约服务；生产 builder、prompts、fact extractor、BM25、语义索引、视觉 cosine、持久化和检索均执行真实实现。人物用例只在 GPU 推理边界提供合成脸/声向量和 M3 输出，真正执行声音/人脸分组、跨 episode 匹配、等价映射、API grounding、character writer 与证据对齐。原 boundary hook 测试不再注入缺失 prompts/llm_utils。

测试覆盖无人物主循环、动态 episode（不是固定手写边界）、topic 新建与更新、fact/clip 联合召回、来源/cutoff、人物跨 episode、原文转写、发言区间/视觉采样帧、失败回滚、恢复幂等、坏索引重建及模型版本变化拒绝复用。

以上证明工程链路，**不证明模型准确率，也不是使用真实 GPU 模型的端到端验收**。内网真实验收使用前述 `lab.full_loop` 命令和 AMI/自有视频；人工核对动态边界、转录、M3关联、fact语义及具体clip，再使用已有评估模块计分。生产入口不导入 `tests`，缺模型、API或证据均明确报错。
