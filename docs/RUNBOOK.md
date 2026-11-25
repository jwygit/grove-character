# 运行手册

以下命令从本项目根目录运行。每次 `--out` 使用新目录，避免覆盖旧证据。所有模型路径需要替换；配置只放环境变量名，不放密钥。

## 1. CPU 单测与依赖

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/smoke.py runs/smoke-example
```

本地已建好 `.venv`。GitHub 拉取后先运行 `python scripts/setup_grove.py`；该脚本从随仓库提供的 `upstream/Grove.bundle` 离线还原 `sources.lock.json` 的固定提交，校验 bundle/补丁哈希，再应用补丁。不访问 Grove 上游，原来的兄弟目录不作为运行依赖；两段测试视频已随仓库提供。模型权重和 Python 依赖未打包，内网部署需事先准备或使用内部镜像。

## 2. GPU 环境准备

推荐单独 Python 3.11 环境分别安装语音与 Omni 依赖，防止依赖互相约束。先安装适合服务器 CUDA/驱动的 PyTorch，再安装所选后端。下面是安装方向，**不是已在 GPU 验证的锁定环境**：

```bash
pip install faster-whisper 'pyannote.audio>=4,<5'
# 或独立环境：
pip install funasr
# face / voice 环境：
pip install insightface onnxruntime-gpu opencv-python-headless torchaudio 'hdbscan>=0.8.40,<0.9'
# Omni 环境：
pip install 'transformers>=4.51,<5' accelerate qwen-omni-utils
```

ERes2NetV2 还需要 [3D-Speaker 的 speakerlab](https://github.com/modelscope/3D-Speaker) 源码加入 PYTHONPATH（依照其依赖安装），以及与 M3 相同结构的 checkpoint。代码从该源码导入类，没有复制 M3 的 import-time CUDA 全局初始化。

提前下载 faster-whisper large-v3、pyannote community-1 完整离线模型、Paraformer-zh/FSMN-VAD/ct-punc/CAM++、InsightFace buffalo_l、ERes2NetV2、Qwen2.5-Omni。pyannote 的 gated 权重须先接受模型条件。`configs/perception.json` 中两个 `*_model_id` 必须替换为实际权重指纹，防止不同权重混用同一 embedding 空间。InsightFace 的 root 下按其 models/buffalo_l 布局放置模型。

内网推理时将配置指向绝对本地路径，并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。确认所有 tokenizer/feature-extractor/ONNX 子文件齐全；未完整安装时不要把模型的自动下载失败解释为识别失败。保存 `pip freeze`、模型 revision/hash、GPU/驱动信息到每轮运行记录。代码记录 Python/关键依赖版本，但未自动采集显存峰值（Omni 适配器除外）或 API 费用。

## 3. 语音前端对比

编辑模型配置后执行：

```bash
python -m lab speech --video data/media/IS1009a_01.mp4 --config configs/gemini.json --out runs/speech-gemini-01
python -m lab speech --video data/media/IS1009a_01.mp4 --config configs/whisper_pyannote.json --out runs/speech-local-01
```

Gemini 密钥通过 `GEMINI_API_KEY` 注入，不需要写入文件。中文视频另运行 `configs/funasr_zh.json`；AMI 是英语，不能使用中文 Paraformer 作为英语公平对照。

每组写出 `audio.wav`、`result.json`（原始响应/词时间戳、分段、输入哈希、模型配置、代码哈希、耗时）。失败写出 `status=failed`，不返回伪造结果。耗时包含 ffmpeg 与模型加载，不等于稳定流式延迟。

复制 `data/IS1009a_01.speech.truth.template.json`，人工填写所有轮次的 `id/start/end/text/speaker`，时间相对视频开头；短发言和重叠发言也标注，speaker 使用人工人物标签，不使用待测模型标签。确认音画同步后设 `reviewed=true`，保存到 `data/IS1009a_01.speech.truth.json`。模板默认不能被计分。

```bash
python scripts/compare_speech.py --truth data/IS1009a_01.speech.truth.json --predictions runs/speech-gemini-01/result.json runs/speech-local-01/result.json --out runs/speech-comparison-01.json
```

比较器校验原视频与解码音频哈希；失败模型仍有单独行。不要只报告成功 clip 的平均分。

## 4. 固定 episode 后的人脸声音关联

第一轮将 Grove 导出的 episode 边界、frame captions 固定。Grove 使用 `day * 86400 + 当天秒数`；本实验视频内部时间从 0 开始，`--time-origin` 显式提供视频第 0 秒对应的 Grove 绝对秒数。不要把 DAY1 的 00:00:00 当作 0。

例如取视频内 0–30 秒作为一个**人为 smoke 边界**，并非声称 Grove 算法产生了此边界：

```bash
python -m lab perceive --video data/media/IS1009a_01.mp4 --speech runs/speech-local-01/result.json --start 0 --end 30 --time-origin 86400 --source ami-clip01 --episode-id ami-clip01-ep1 --config configs/perception.json --out runs/perception-ep1
python -m lab associate --bundle runs/perception-ep1/bundle.json --config configs/av_omni.json --accept-model-links --out runs/av-ep1
```

`perceive` 产生裁剪视频、每簇最多 3 张脸图、face/voice observations 和统一时间戳，另存 `face_detections.json` 记录所有检测的分数、特征、时间和聚类标签。使用 M3 HDBSCAN 和两阶段质量筛选，默认 min_cluster_size=20；短 episode/短暂人物可能没有合格簇，不会自动退回阈值去重。声音从 speech 结果旁的 `audio.wav` 无损裁剪，校验与 ASR 一致的音频哈希；文件不在时从原视频直接解码，绝不从 AAC 重编码片段提声纹。统一只做 >=2 秒时长筛选，不按前端独有 overlap 标记筛除。短句文字保留，但没有声纹身份。首次 GPU 调通仍需检查人脸质量与声纹输出。

`associate` 先完成声音节点匹配，再以 `voice_groups` 形式调用 AV 适配器；Gemini AV 对照可换 `configs/gemini.json`。后一 episode 用 `--prior runs/av-ep1/snapshot.json` 传入图快照。人脸默认均值阈值 0.3/特征池 10，声音默认均值阈值 0.6/特征池 20；配置分别为 `img_matching_threshold` / `max_img_embeddings` 与 `audio_matching_threshold` / `max_audio_embeddings`。AV 配置的 `reid_thresholds.face` / `.voice` 可覆盖阈值，`reid_margin` 已不适用且会报错。以上默认参数沿用 M3，仍需真实数据验证。

新 snapshot 增加 `voice_groups`、`voice_links`、`excluded_voice_turns`，每条 character_ref 增加 `voice_id` 和 `visible_face_id`。音画模型需返回完整的声音身份关系与逐条可见性；未知/画外音为 null。旧 bundle 的语音来自不同切片路径，需重跑 `perceive` 生成 `voice_pipeline_version=2`，再运行 `associate`。不要拿旧结果缓存冒充新实验。`specialists` 的旧逐句协议暂不支持新的声音分组入口；默认 Omni 和 Gemini 均已接入新协议。

需要先隔离 AV 错误时，将人工校正轮次制成相同 speech result schema，注明其 oracle 来源，再提取 bundle。更换前端时应复制冻结的人脸 gallery/observations，只重做声音证据，避免人脸模型随机差异混入对比。当前没有批量冻脸 orchestration，需人工检查 bundle 的 face 部分一致。

人工填写 `data/association.truth.template.json` 的 `bundle_sha256` 与全部真实语音区间 `start/end/face_id`；face_id 使用此 bundle 的 gallery ID，画外音为 null。然后：

```bash
python -m lab evaluate-association --truth data/association.truth.json --bundle runs/perception-ep1/bundle.json --snapshot runs/av-ep1/snapshot.json --out runs/association-score.json
```

## 5. 隔离验证 Grove 写入

准备 `data/episode.json`：从真实 Grove 导出 `VideoEpisode.to_dict()`，包含 episode_id/time_span/start_seconds/end_seconds/day/frame_captions；时间范围须与 snapshot 完全一致。`tests/fixtures.py` 的 episode() 展示格式，但其中 caption 是合成数据。

配置 `configs/writer.json` 的 OpenAI-compatible endpoint/model 与可选环境变量。可以接本地模型服务或 API。

```bash
python -m lab grove-write --episode data/episode.json --snapshot runs/av-ep1/snapshot.json --config configs/writer.json --identity-mode unknown --out runs/grove-g0
python -m lab grove-write --episode data/episode.json --snapshot runs/av-ep1/snapshot.json --config configs/writer.json --identity-mode predicted --out runs/grove-g1
```

G0 与 G1 使用同一 writer、同一边界、同一发言文字；G0 仅隐藏 character。G2 使用独立人工校正的 snapshot。输出真实 `hypergraph.json` 并验证保存/重载一致，另有 provenance。此入口不运行 topic aggregation / retrieval / 动态分段。

按 `characters.truth.template.json` 人工对齐事实的共同 unit_id、role、canonical person，同时把预测事实的 character ID 映射到相同注册人物体系，形成 `characters.prediction.json`；两文件使用相同 evaluation_set_id。

```bash
python -m lab evaluate-characters --truth data/characters.truth.json --prediction data/characters.prediction.json --out runs/character-score.json
```

这是人工标注后的可执行准确率计分，不是用 LLM 自己的输出生成真值。当前没有自动跨文本事实对齐器；不应使用生成 fact 的随机 UUID 直接对齐不同模型输出。

## 6. 完整 Grove 接入位置

获得上游缺失模块后，在喂 caption 前调用：

```python
from hypervideo.modules.character_identity import SnapshotProvider
builder.set_character_identity_provider(SnapshotProvider(snapshots, source_id))
```

测试默认更换为专用 character writer。`replace_fact_writer=False` 仅注入 episode 身份证据；原始 fact extractor 还需补充 prompt 和解析代码才会生成结构化 character_refs，不能认为加字段就完成归属。

在线系统应替换为异步 provider，在真实动态边界确定后取对应原始证据并等待推理；SnapshotProvider 只用于严格匹配既有 episode 的离线实验。整个系统的并行队列、实时 backpressure、动态边界准确率仍属于后续完整集成验证。
