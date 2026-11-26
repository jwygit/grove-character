# Fact → evidence → clip

本次新增明确的证据关联，不从“同一个人物/同一个 episode”推断 fact 属于某个 clip。

```text
ASR 发言 + 原始时间 ────────────────┐
                                  ├→ writer 引用证据 ID → fact.evidence_refs
M3 描述 → 视觉定位 → 时间段/实际帧 ──┘                           │
                                                           按来源与时间求交
                                                                ↓
                                                       fact.clip_evidence
```

## 语音与视觉分别处理

语音：writer 引用 `utterance_ids` 后，代码直接取该发言的原始时间、来源和文字；不接受 writer 自填的 clip ID 或时间。短句/未知 speaker 也有语音证据，可以关联 clip。语音证明“这句话被说过”，不能据此证明谈论的事件实际发生。

视觉：保留原始 M3 联合 prompt，新增独立 `ground-visual` 第二遍定位。它抽取 episode 视频的完整画面，携带真实解码 PTS、帧 ID，并将代表脸、已确认的身份分组和 M3 description 发给支持多图输入的视觉模型。模型为每条 description 返回：

- 一个或多个时间段，并引用时间段内实际提供的帧；或直接引用帧 ID，作为点证据。
- 无法验证、画外事件、纯语音内容：返回空定位及原因。

不会把多次分离出现连成一个大区间，不把高层 conclusion 自动当成直接观察。conclusion 默认保持未定位。采样率不足、事件在两帧之间、遮挡或人物不确定都可能导致未定位；本版不实现主动说话检测或单独的语义蕴含验证器。

定位结果校验 memory ID、原文/标签哈希、来源、episode 时间、帧 ID、区间范围和区间内的帧支持。API 出错写 `failure.json` 并退出，不把失败伪装成空成功结果。模型定位仍是预测，`provenance.reviewed=false`；合法时间和证据 ID 不代表模型一定看对了动作。

## 落盘字段

| 字段 | 内容 |
|---|---|
| `fact.evidence_refs` | 被引用的发言/视觉记忆、原始文字、来源、episode、绝对时间或点帧、定位方法与模型版本；视觉帧含文件位置和 SHA |
| `fact.clip_evidence` | `evidence_id → clip_id`、实际相交时间、证据类型、对应帧 ID；关系类型为 `temporal_overlap` |
| `fact.evidence_alignment` | 覆盖状态、每条证据的时间覆盖率、输入指纹 |

例如动作证据为绝对时间 `[86404,86406)`，clips 为 `[86400,86405)`、`[86405,86410)`：保存两条关联，分别指向前一 clip 的 `[86404,86405)`、后一 clip 的 `[86405,86406)`。恰好位于 86405 的帧点只属于后一 clip。同时间的另一个 source 不匹配。

状态：`complete` 表示选定证据的时间/帧被 clips 全部覆盖；`partial` 表示只有部分覆盖或混有未定位证据；`uncovered` 表示已知位置没有可用 clip；`unlocalized` 表示证据位置未知；`no_evidence` 表示没有引用可用证据。**complete 是覆盖完整，不是事实正确率或人工确认。** 重叠 clips 的覆盖时间先求并集，不重复计数。

人物合并/撤销只重算身份引用和正文，事件原始时间及 clip 证据不应移动。窗口重建后会重新计算关联、清除过期 clip ID。新图全部计算成功才返回；保存/加载与 VideoFact/VideoFactNode 转换均保留这些字段。

视觉预测记录其使用的 identity state 和 feature registry 版本。保留原始定位不会自动证明修订后的语义仍正确：如果错误脸声关系影响了模型对动作人物的判断，应重新定位或人工复核；代码不会把身份重建当作重新运行视觉模型。

## GPU/API 服务器流程

先按 RUNBOOK 运行 speech → perceive → associate。新版 perceive 会记录 `episode_video_sha256`；定位阶段校验它和 association 的 bundle 哈希，防止同一路径的视频已被替换。旧 bundle 需要重新 perceive/associate。

配置 `configs/visual_grounding.json`：填写能处理多张 `image_url` 的 OpenAI-compatible `/v1/chat/completions` 地址、视觉模型 ID 和模型/部署 revision。默认指向 localhost；可在 GPU 服务器部署视觉模型服务。此配置不固定新的模型选型，也没有宣称某个 checkpoint 已测通。

```bash
python -m lab ground-visual \
  --bundle runs/perception/bundle.json \
  --snapshot runs/association/snapshot.json \
  --config configs/visual_grounding.json \
  --out runs/grounding

python -m lab grove-write \
  --episode data/episode.json \
  --snapshot runs/grounding/snapshot.json \
  --config configs/writer.json \
  --out runs/grove-grounded
```

上述路径均为示例，替换为同一 episode 的实际文件。`ground-visual` 不修改原 snapshot，输出新 snapshot 和抽取帧。默认 1 FPS、最多 120 帧；超过预算会报错，需明确调低采样率或使用更短 episode，不会悄悄截断后半段。实际解码使用 ffmpeg showinfo 的 PTS，不用请求 seek 的时间冒充帧时间。

`grove-write` 是现有隔离 writer，本身不生成视觉窗口。若还没有 clips，先保存时间证据，状态为 uncovered。用已有视觉构建流程加入 clips 后：

```bash
python -m lab grove-link-clips \
  --graph runs/grove-with-clips/hypergraph.json \
  --out runs/grove-with-clips/hypergraph-linked.json
```

输出要求新路径，原图不覆盖。完整 Grove 接入沿用 SnapshotProvider，将 enriched snapshot 提供给边界后的 writer；builder 保存前和核心索引检查前自动重建关联。身份修订的 `grove-project` 也会重建这些边。

旧 graph 没有 `evidence_refs` 时不会从人物标签猜关联，应使用已定位的 snapshot 重跑 writer。旧 snapshot 未做视觉定位仍能写入，但视觉 memory 会明确标为 unlocalized。

## 独立模块与查询

不依赖 Grove 的 `lab/fact_evidence.py` 提供 `select_evidence()`、`align_evidence()`；同一实现逐字节放入 Grove 补丁并由测试校验。纯 JSON CPU 对齐：

```bash
python -m lab align-facts --facts facts.json --clips clips.json --out aligned-facts.json
```

`facts.json` 是带 `evidence_refs` 的 fact 数组；`clips.json` 是 `[{"id":"clip-1","source":"recording","start":86400,"end":86405}]`。字段来源以 writer/grounding 输出为准，不让另一个模型重新编造时间。

核心 HierarchicalRetriever 与 week 的 `hierarchical_v2_day` 返回 fact 的具体 clip 关联，回答上下文显示 clip ID、交集时间和 speech/visual 类型。关闭视觉语义通道时仍保留已存的证据引用，不调用视觉 encoder。查询 scope/cutoff 会过滤引用，完整 clip 超过 cutoff 时不会经 fact 引用泄露；响应会重新显示当前范围内的覆盖状态，不修改存储的全量关联。修改窗口但未重建会明确报 stale，不能继续使用旧关联。旧 week 检索 modes 不作为本次证据链验证入口。

引用 clips 不占据额外语义检索排名；返回一条 fact 的证据关联，不等于把它关联的所有 clips 都加入最终 top-k 节点列表。

## 验证范围

```bash
python -m unittest tests.modules.test_fact_evidence tests.modules.test_visual_grounding -v
python -m unittest tests.grove.test_fact_clips -v
python scripts/smoke_fact_clips.py runs/fact-clips-smoke-new
```

独立测试执行真实 ffmpeg 生成/解码合成测试视频、检查 PTS，并用明确的 HTTP 调用替身验证多图/身份分组/时间协议。Grove 测试运行真实 writer 解析、图持久化、修订重建和两条查询入口。smoke 生成 4 条 facts、2 个 clips、4 条证据边，其中一条 fact 保持未定位；模型输出是合成的，报告禁止用于模型选型。

真实服务器评测需人工标注事件时间/帧和 fact 的支持证据，分别测语音对齐、视觉定位 IoU/帧命中率、clip precision/recall、未定位比例，以及“引用是否真的支持该句 fact”。本机尚未运行真实视觉定位模型或准确率评测；完整 Grove 上游缺失模块的限制仍然存在。
