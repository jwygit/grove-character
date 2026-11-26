# Visual clip 与 fact 联合检索

当前默认部署入口是 [完整主循环](FULL_LOOP.md)，整体检索架构见 [ARCHITECTURE.md](ARCHITECTURE.md)。它会自动建立 clips 和索引，再调用这里的联合检索模块。下方 week CLI 属于原 benchmark 环境的入口说明，不代表新标准主循环已补齐所有旧入口依赖。

2026-11-26 增加 clip 人物引用和独立人物召回通道；`--character` 可绕过语义 topic/episode 召回漏失，但仍受日期/来源/cutoff 限制。角色与修订流程见 [CHARACTER_GRAPH.md](CHARACTER_GRAPH.md)。以下原视觉通道仍保留；人物标签不会变成额外 fact。

上游 `mmlifelong_week_retrieval_v2.py` 的 `hierarchical_v2_day` 中，`use_visual_channel=True` 原先只收集 clip ID，后续打分为 TODO。`visual_hybrid_day` 则用视觉命中扩展 parent episode 的 facts；另一个 visual rerank 是检索后的附加描述。它们都没有让 clip 独立进入最终的 fact 候选列表。本次补丁实现统一的叶节点检索，并接到 Grove 核心检索器。

## 当前路径

```text
query → topic → episode（日期 / 历史时间范围过滤）
                   ├─ fact → 原文本索引的 BM25 / embedding 检索
                   └─ visual_clip → 匹配的视觉文本 encoder → visual_index.npz cosine
                              ↓
                 两类节点按各通道排名对齐分数
                 可选 topic / episode 得分传递
                              ↓
                     一个共享 top-k evidence 列表
                              ↓
           带类型、ID、时间、parent episode 的回答上下文
```

visual_clip 不依赖相关 fact 是否存在。episode 有视觉片段但没有 fact 时，可以直接返回 clip。节点通过 `episode_window_hyperedges` 与父 episode 相连；导入图缺少超边时支持 `parent_episode_id` 反向链接。有超边时校验其 clip 与父节点的一致性，不把旧的游离节点当作当前窗口。

fact 文本检索得分通常是 RRF，clip 是 cosine，二者不直接按数值相加。默认使用各通道的 `1/(60 + rank)`，等权合并后共同取 top-k；相同排名的并列项固定按类型/ID 排序。`mult`、`additive`、`rrf` 跨层得分传递同时作用于两类节点。这是对不同通道排名的对齐，不是概率校准，也不是同一节点被两路命中的佐证；相关性效果需要真实查询验证。

## 开关与输出

`configs/visual_retrieval.json` 已设置 `use_visual_channel=true`，`fact_top_k=12` 表示 **fact + visual_clip 合计最多 12 个叶节点**。先修改索引路径和 encoder 服务地址。通用 `HyperVideoConfig` 默认仍关闭该通道，以兼容没有视觉索引的调用方；按以下方式加载配置即可开启。

```python
import json
from hypervideo.config import HyperVideoConfig
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever

config = HyperVideoConfig(**json.load(open("configs/visual_retrieval.json")))
retriever = HierarchicalRetriever(hypergraph, dual_index, config)
result = await retriever.retrieve("谁拿着红色杯子？", current_time_span="DAY1-12:00:00")
context = retriever.format_for_llm(result)

# 新消费方使用同一排序列表；旧接口保留各类型分组。
result.evidence       # List[dict]，含 node_type、id、score、raw_score、channel_rank
result.facts          # 仅 fact 的 (id, score)
result.visual_clips   # 仅 visual_clip 的 (id, score)
```

这里的 `hypergraph` 是真实 `VideoHypergraph`，`dual_index` 是已有文本索引实例，需要实现 Grove 的异步 `search` / `search_within`。`HyperVideoGraph` 仍通过传入的 `hyper_config` 初始化核心检索器，其 `format_retrieval_for_llm` 会使用上述混合结果。仍然只读取 `result.facts` 的旧业务代码不会自动消费 clips，需要改读 `result.evidence` 或调用统一 formatter。

视觉条目示例：

```json
{
  "node_type": "visual_clip",
  "id": "vc_123",
  "clip_id": "vc_123",
  "parent_episode_id": "ep_12",
  "episode_ids": ["ep_12"],
  "day": 1,
  "start_seconds": 126000,
  "end_seconds": 126010,
  "frame_time_span": "1-11:00:00-11:00:10",
  "caption_text": "...",
  "raw_score": 0.63,
  "channel_rank": 0,
  "score": 0.016666666666666666
}
```

这里 cosine 只用于选片段；默认回答上下文读取该片段的已有 caption，并不是把真实帧直接传给文本 LLM。需要看原画面补充细节时，可另外开启 VLM caption rewrite。它只处理最终选中的 clips，不重新从所有父 episode 扩展出一批片段。默认关闭 rewrite，避免无配置时额外调用 VLM。

## week retrieval 入口

新的 CLI 默认 mode 为 `hierarchical_v2_day`，默认启用 visual channel，提供关闭开关。完整 Grove 运行环境中：

```bash
PYTHONPATH=vendor/Grove python -m hypervideo.mmlifelong_week_retrieval_v2 \
  --working_dir /data/grove/run \
  --test_json /data/questions.json \
  --retrieval_mode hierarchical_v2_day \
  --use_visual_channel \
  --visual_index_path /data/grove/run/visual_index.npz \
  --visual_server_url http://ENCODER_HOST:9090 \
  --stage2_top_k 12 --score_prop none \
  --out_path /data/grove/predictions.json
```

用 `--no-use_visual_channel` 关闭通道；`--enable_visual_rewrite` 另外启用已存在的 VLM 重描述步骤。`--visual_cosine_threshold` 控制视觉召回门限，默认 0.2，需要用独立开发集选择。上游 counting 类问题仍可能扩大本次查询的 top-k；这个有效预算由两类节点共享。

视觉模式会执行 Stage 2，Stage 1 的摘要直答或 skill 的事实答案不会把视觉检索跳过。多 mode 对比时，其他文本模式仍保留自己的早答逻辑。结果 `retrieved` 中保留类型、分数、时间和父节点，visual 条目使用 `clip_id`，不会伪装成 `fact_id`。服务/索引故障返回显式 failed，避免拿空上下文继续生成答案。

## 索引和服务要求

- 图必须包含有 parent episode 的 visual clips；NPZ 至少包含 `embeddings`、`meta`、`backbone`、`dim`。
- 兼容各构建入口的 `meta.id` / `meta.visual_clip_id` / `meta.clip_id`，ID 必须唯一并对应图中的节点。metadata 的 episode、day、时间必须与图一致；不能混用不同运行生成的 graph 和 NPZ。
- 先按候选 clip ID 和 day 过滤，再计算/截取 top-k。不会先全库取 top-k 后过滤，导致真正候选被挤掉。
- 远程编码使用现有协议：`GET /health` 返回 `backbone`、`dim`；`POST /encode_text` 接收 `{"text":"..."}`，返回 `{"embedding":[...]}`。本机只执行 NumPy cosine，不加载模型。需要安装服务客户端的 `requests`。
- 查询 encoder 必须与构建索引时的模型、权重、预处理一致。代码检查 backbone、维度、有限数和非零向量；同名 backbone 的不同权重仍须由部署方锁定版本。无远程地址时使用已有本地 VLM2Vec/SigLIP 适配，模型回退后若空间不匹配会报错。
- 修复三个视觉构建入口的节点时间，统一为 `day*86400 + 秒数`；metadata 的 `*_sec_in_day` 仍是当天秒数。旧图节点可在查询时转换，不修改原文件。允许跨 episode 边界但与父 episode 重叠的完整窗口；历史查询会额外排除 clip.end 超过 cutoff 的窗口。
- visual 索引按文件版本和 endpoint 缓存。索引缺失、缺少候选 ID、图/索引不一致、模型空间不匹配或 encoder 故障会明确失败，不静默当作零命中。合法的零命中（门限过滤、该范围没有 clips）仍返回空视觉列表，可保留文本结果。

## 本机验证与部署范围

```bash
.venv/bin/python -m unittest tests.modules.test_visual_index tests.grove.test_visual_retrieval -v
.venv/bin/python scripts/smoke_visual.py runs/visual-smoke-new
```

smoke 生成真实 Grove graph、NPZ、混合 `evidence.json` 和 `context.txt`，其中使用合成向量、固定 query embedding 和测试用文本索引，报告明确 `synthetic=true`，不能用于准确率选型。单测实际运行 NumPy cosine、图加载、核心检索、week retrieval、Stage 2 格式与异常路径；没有执行 GPU 或真实 API。

最初补视觉分支时，公开 Grove 仍缺少文本索引、RRF 和部分 prompts/核心模块；后续兼容主循环版本已补齐这些依赖，见 [FULL_LOOP.md](FULL_LOOP.md)。上游未提供 `visual_encoder_server.py`，现在由项目的 `lab.visual_server` 提供本地 SigLIP 图像/文本编码服务。真实部署和依赖见 [OFFLINE_DEPLOYMENT.md](OFFLINE_DEPLOYMENT.md)；工程测试不代表 GPU 模型质量验收。

内网升级不能只 `git pull` 后继续使用旧 `vendor/Grove`。保留旧目录中需要的改动，移到备份目录，再运行 `python scripts/setup_grove.py` 应用新版补丁。原人物关联和 character 转写仍包含在同一补丁中。

真实检索评测应固定 graph、索引模型、query 集、有效 top-k 和 writer，标注相关 fact/clip 与时间范围；分别测各类型的 recall、整体 precision、视觉独有问题的覆盖及答案是否有证据。API 失败单独记失败率，不能排除失败样本后只报告命中率。CPU smoke 不提供这些真实准确率结论。
