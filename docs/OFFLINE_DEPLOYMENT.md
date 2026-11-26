# 内网 GPU 部署：核心 → 视觉检索 → 人物

适用于当前 `grove-character` 的 `python -m lab.full_loop`。目标是先验证动态 episode → 摘要 → facts/topic → 检索，再验证人物归属和 fact→clip 证据。不是启动上游仍有缺失依赖的旧 benchmark 总入口。

**状态：下面是按实际代码及上游依赖编写的候选部署配方。已验证本项目 CPU/HTTP 工程链路；本机没有 NVIDIA GPU，尚未构建/验收这些 Linux 镜像或跑真实权重。** 首次部署必须执行本页的服务和视频验收。安装成功、HTTP 200、`status=complete` 与识别准确率分别记录。

## 1. 需要提前带进内网的材料

| 阶段 | 模型/材料 | 具体用途 |
|---|---|---|
| 核心必需 | `Qwen/Qwen2.5-VL-7B-Instruct` | 一套服务同时承担帧描述、动态边界判断、摘要、facts、topic；人物阶段复用为视觉 grounding |
| 核心必需 | `Qwen/Qwen3-Embedding-0.6B` | topic、episode、fact 的语义索引和查询 |
| 开启视觉通道 | `google/siglip-base-patch16-224` | 实际帧与文本查询编码到同一空间；本例 768 维，以服务 `/health` 为准 |
| 人物：语音 | `Systran/faster-whisper-large-v3` + `pyannote/speaker-diarization-community-1` | 单词时间戳、转录、说话人分段；先单独运行，结果交给主循环 |
| 人物：人脸 | InsightFace **完整 buffalo_l 模型包** | 检测、脸特征；HDBSCAN 是算法/依赖，不另下模型 |
| 人物：声纹 | `pretrained_eres2netv2.ckpt` + 固定版本 `modelscope/3D-Speaker` 源码 | ERes2NetV2 声纹；`speakerlab` 从该源码导入 |
| 人物：音画联合理解 | `ByteDance-Seed/M3-Agent-Memorization` | episode 视频、代表脸、发言 → 描述及 face/voice 等价关系 |
| 数据与代码 | 完整仓库、`upstream/Grove.bundle`、补丁、两段 AMI 视频、下面的 Docker 镜像 | 不需要在内网重新访问 Grove 上游或 PyPI |

这是一组用于先跑通的模型选择，还不是准确率排名。核心不需要 Gemini、Whisper、InsightFace 或 M3；完整本地人物路线也不需要 Gemini。已有内网 API 时可复用满足协议的服务，不必在主循环机器重复部署所有模型。当前 M3/人脸/声纹分支在主进程本地推理，还没有改成远程 API。

权重必须连同 tokenizer、processor、配置和全部 shards 一起复制；只有 `.safetensors` 或 Git LFS 指针不够。pyannote 模型先在联网环境用自己的账号接受访问条件、确认可下载。自有中文数据可另测 FunASR；本页用仓库的英语 AMI 视频，选择 Whisper + pyannote，不混装另一条 ASR 路线。

宿主机准备：Linux x86_64、适配 GPU/CUDA runtime 的 NVIDIA 驱动、Docker Engine、NVIDIA Container Toolkit、足够存储。下面镜像内含 Python 3.11、ffmpeg/ffprobe、音视频动态库与 Python 依赖。**驱动、Docker 和 Container Toolkit 的离线安装包也要按目标操作系统准备**；镜像不能代替宿主机驱动。不要把 Mac 的 venv/wheel 当作 Linux GPU 安装包。

显存分配取决于 GPU 型号、张数、VLM 上下文、视频长度及并发；本项目尚无实测最低显存。先把 ASR 跑完并退出，再启动人物。SigLIP 可以先放 CPU。VLM/embedding 必须在人物主循环运行期间可用，不能简单关掉它们给 M3 腾显存；显存不够时将这些 API 放到其他内网 GPU，或重新安排模型/服务配置。

## 2. 联网 Linux 机器：构建并打包依赖

在**与内网目标相同架构的 Linux** 上，从最新仓库根目录执行。`SPEAKERLAB_REVISION` 会解析一次并固定到镜像；保存后不会在内网跟随 main 更新。Docker 镜像保存已经解析安装的依赖与 `/opt/python-packages.txt`。配方没有锁死所有传递依赖，因此同一实验使用同一份导出镜像，不重新在线构建后混用结果。

```bash
git clone https://github.com/jwygit/grove-character.git
cd grove-character
export GROVE_ASSETS="$(cd .. && pwd)/grove-assets"
mkdir -p "$GROVE_ASSETS/models"

docker build -f deploy/Dockerfile --target core -t grove-core:offline-v1 .
docker build -f deploy/Dockerfile --target visual -t grove-visual:offline-v1 .
docker build -f deploy/Dockerfile --target asr -t grove-asr:offline-v1 .
export SPEAKERLAB_REVISION="$(git ls-remote https://github.com/modelscope/3D-Speaker.git HEAD | cut -f1)"
docker build -f deploy/Dockerfile --target character \
  --build-arg SPEAKERLAB_REVISION="$SPEAKERLAB_REVISION" -t grove-character:offline-v1 .
docker pull vllm/vllm-openai:v0.10.2

docker save -o "$GROVE_ASSETS/grove-images.tar" \
  grove-core:offline-v1 grove-visual:offline-v1 grove-asr:offline-v1 \
  grove-character:offline-v1 vllm/vllm-openai:v0.10.2
docker image inspect --format '{{.Id}} {{.RepoTags}}' \
  grove-core:offline-v1 grove-visual:offline-v1 grove-asr:offline-v1 \
  grove-character:offline-v1 vllm/vllm-openai:v0.10.2 \
  > "$GROVE_ASSETS/image-ids.txt"
```

依赖隔离的原因：核心安装 `requirements-workbench.txt`（含原 CPU 核心依赖、Web/API 库）和下载工具；SigLIP 用 Transformers 4.51.3 / Torch 2.6；ASR 用 pyannote 4.0.3（其依赖指定 Torch/Torchaudio 2.8、TorchCodec 0.7）；M3 使用其上游指定的 Transformers commit 和 `qwen-omni-utils==0.0.4`，另装人脸/声纹依赖。vLLM 是独立服务镜像，不装进 M3 环境。

M3 首次使用 `sdpa`，避免先编译 FlashAttention；这不是原上游默认的 `flash_attention_2`，须作为实验配置记录。需要严格比较时，另外制作带匹配 FlashAttention 的镜像、改配置并用新输出目录跑同一组数据。

## 3. 联网下载模型

先在终端安全设置自己的 `HF_TOKEN` 环境变量（需要 pyannote 权限；不把 token 写进配置、镜像或仓库），再执行：

```bash
docker run --rm -i --user "$(id -u):$(id -g)" \
  -e HF_TOKEN -v "$GROVE_ASSETS/models:/models" grove-core:offline-v1 python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root = Path('/models')
path = root / 'manifest.json'
manifest = json.loads(path.read_text()) if path.exists() else {}
repos = {
    'Qwen2.5-VL-7B-Instruct': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'Qwen3-Embedding-0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    'siglip-base-patch16-224': 'google/siglip-base-patch16-224',
    'faster-whisper-large-v3': 'Systran/faster-whisper-large-v3',
    'pyannote-community-1': 'pyannote/speaker-diarization-community-1',
    'M3-Agent-Memorization': 'ByteDance-Seed/M3-Agent-Memorization',
}
for name, repo in repos.items():
    revision = manifest.get(name, {}).get('revision') or HfApi().model_info(repo).sha
    snapshot_download(repo, revision=revision, local_dir=str(root / name))
    manifest[name] = {'repo_id': repo, 'revision': revision}
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, revision)
PY

mkdir -p "$GROVE_ASSETS/models/insightface/models/buffalo_l"
curl --fail --location --retry 3 \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
  -o "$GROVE_ASSETS/buffalo_l.zip"
unzip -n "$GROVE_ASSETS/buffalo_l.zip" \
  -d "$GROVE_ASSETS/models/insightface/models/buffalo_l"
curl --fail --location --retry 3 \
  https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt \
  -o "$GROVE_ASSETS/models/pretrained_eres2netv2.ckpt"
```

`manifest.json` 固定 Hugging Face 下载 commit。配置生成器还会计算完整人脸包及声纹文件的 SHA256。buffalo_l 目录应直接含 `det_10g.onnx`、`w600k_r50.onnx`、`1k3d68.onnx`、`2d106det.onnx`、`genderage.onnx`，不要多套一层目录。3D-Speaker 源码已在 character 镜像中，不需再次 pip 安装同名 `speakerlab` 包。

打包模型并记录传输校验和；将这三个文件、`image-ids.txt` 和最新项目代码带进内网。若只能先准备核心，可先只下载表中前三个 HF 模型、构建 core/visual/vLLM；人物材料后补。

```bash
tar -C "$GROVE_ASSETS" -cf "$GROVE_ASSETS/grove-models.tar" models
(cd "$GROVE_ASSETS" && sha256sum grove-images.tar grove-models.tar > SHA256SUMS)
# 无法在内网 git clone 时，可在项目目录导出代码（含视频、Grove bundle 和补丁）：
git archive --format=tar.gz -o "$GROVE_ASSETS/grove-character-source.tar.gz" HEAD
```

## 4. 内网：加载环境，先跑工程测试

项目和资产放到任意可写位置。下文所有操作从项目根目录执行；把 `GROVE_ASSETS` 改成实际绝对路径。`deploy/run.sh` 统一挂载 `/work`、`/models`，保持证据/缓存的绝对路径稳定。导出源码没有 `.git` 也可以 `setup_grove.py` 和跑主循环；`check_clean_export.py` 则需要正常 Git checkout。

```bash
export GROVE_ASSETS=/srv/grove-assets
(cd "$GROVE_ASSETS" && sha256sum -c SHA256SUMS)
docker load -i "$GROVE_ASSETS/grove-images.tar"
tar -C "$GROVE_ASSETS" -xf "$GROVE_ASSETS/grove-models.tar"
nvidia-smi

bash deploy/run.sh core python scripts/setup_grove.py
bash deploy/run.sh core python -m unittest discover -s tests -t . -v
bash deploy/run.sh core python scripts/smoke_full_loop.py runs/deploy-engineering-smoke
```

已有 `vendor/Grove` 时按 [升级说明](FULL_LOOP.md#已有部署如何升级) 先备份、再 setup；setup 不会覆盖旧目录。以上测试的模型输出是明确的测试数据，不下载/加载真实模型，不能代替下面的验收。

## 5. 核心：启动 VLM 与文本 embedding

示例把两个服务分别放在物理 GPU 0 和 1。**这里的编号是服务部署示例，不代表必须两张或三张卡；必须按实际卡数/显存修改。** 同一卡共享时分别降低 `--gpu-memory-utilization`，并留出实际权重和 KV cache 空间；该参数不是显存足够的保证。其他内网机器已有服务时直接使用其 endpoint，跳过相应启动命令。

```bash
export VLM_GPU=0
export EMBEDDING_GPU=1

docker run -d --name grove-vlm --gpus "device=$VLM_GPU" --ipc=host \
  -p 127.0.0.1:8000:8000 -v "$GROVE_ASSETS/models:/models:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai:v0.10.2 \
  --model /models/Qwen2.5-VL-7B-Instruct --served-model-name grove-vlm \
  --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 \
  --max-num-seqs 1 --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":64}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":200704}'

docker run -d --name grove-embedding --gpus "device=$EMBEDDING_GPU" --ipc=host \
  -p 127.0.0.1:8001:8000 -v "$GROVE_ASSETS/models:/models:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai:v0.10.2 \
  --model /models/Qwen3-Embedding-0.6B --served-model-name grove-embedding \
  --runner pooling --host 0.0.0.0 --port 8000 --dtype auto \
  --max-model-len 8192 --gpu-memory-utilization 0.30

docker logs --tail 50 grove-vlm
docker logs --tail 50 grove-embedding
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8001/v1/models
curl --fail http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"grove-embedding","input":["a person sitting at a meeting table"]}'
```

两个服务分配的是不同卡时，0.85 和 0.30 不需要相加；如果设成同一卡，必须调整。人物服务仍需另分可用显存，不能忽略已有服务的预留。老 GPU 不支持相应 dtype/内核时须调整部署，不把失败归结为 Grove。

`caption` 接口必须接受多图和 JSON 输出；文字模型不能代替它。64 图限制覆盖本页起步配置的至多 40 张 grounding 帧 + 24 个代表脸；只支持 1 张图的默认服务会在人物阶段失败。图像像素、上下文及 episode 上限都是影响质量和显存的实验参数，不可调整后继续复用旧实验目录。

生成并运行核心配置：

```bash
bash deploy/run.sh core python scripts/configure_offline.py \
  --stage core --output configs/deploy-core.local.json
bash deploy/run.sh core python scripts/preflight.py --config configs/deploy-core.local.json
bash deploy/run.sh core python -m lab.full_loop \
  --video data/media/IS1009a_01.mp4 --config configs/deploy-core.local.json \
  --output runs/deploy-core --source IS1009a-central-01 \
  --query 'What objects are on the table?'
```

配置生成器读取模型 commit，使用服务别名 `grove-vlm`、`grove-embedding`，核心阶段 `use_visual_channel=false`。它采用统一的 **30 秒 buffer 上限**，仍由 Grove 动态判断更早边界，不是固定每 30 秒分段；上游按窗口追加，实际 episode 上限可能越过一个窗口。与正式实验的 120 秒配置分开记录。

远程 API 可在生成时加 `--vlm-url`、`--embedding-url`、`--visual-url`；如果服务名称不同，生成后同步改 `model`，并按实际部署记录 revision。鉴权键由 shell 环境传入，不写进 JSON。已填配置只证明格式，`preflight` 不探测真实模型服务；最终以真实视频运行和人工检查为准。

## 6. 核心第二步：开启 visual clip 检索

此前只有客户端，没有可直接启动的视觉服务端；现已提供 `lab.visual_server`，加载本地 SigLIP 权重，实际编码图像和查询。不是将 caption 向量冒充图像特征。单独终端运行以下前台服务；先放 CPU，无需占用 GPU：

```bash
SIGLIP_REVISION=$(bash deploy/run.sh core python -c \
  'import json; print(json.load(open("/models/manifest.json"))["siglip-base-patch16-224"]["revision"])')
bash deploy/run.sh visual python -m lab.visual_server \
  --model /models/siglip-base-patch16-224 \
  --revision "$SIGLIP_REVISION" --device cpu --port 9090
```

要放 GPU：在命令前设置 `GROVE_GPU=实际卡号`，并把 `--device cpu` 改为 `--device cuda:0`。服务启动时做真实图像/文本编码探针，`/health` 返回实际维度和记录的 revision。另一个终端执行：

```bash
curl --fail http://127.0.0.1:9090/health
curl --fail http://127.0.0.1:9090/encode_text \
  -H 'Content-Type: application/json' -d '{"text":"a meeting table"}'
bash deploy/run.sh core python scripts/configure_offline.py \
  --stage visual --output configs/deploy-visual.local.json
bash deploy/run.sh core python -m lab.full_loop \
  --video data/media/IS1009a_01.mp4 --config configs/deploy-visual.local.json \
  --output runs/deploy-visual --source IS1009a-central-01 \
  --query 'people sitting around a table'
```

此时 `use_visual_channel=true`，fact 和 visual clip 联合召回。生成器从 `/health` 填维度并核对模型 revision，不照抄原示例里的 1152。base SigLIP 用于先跑通英语 AMI；中文视觉查询质量要单独测，不能因为文本 embedding 支持中文就假定视觉 encoder 同样合适。

## 7. 人物第一步：单独运行 ASR

`GROVE_GPU` 是此进程使用的物理卡号；容器内都是 `cuda:0`。可使用暂时空闲的卡，或先停服务完成 ASR、再恢复服务。必须对随后主循环的**同一份视频**生成转录；换视频/剪辑后重新生成，系统会校验 PCM。

```bash
GROVE_GPU=2 bash deploy/run.sh asr python -c \
  'import torch, torchaudio, torchcodec, ctranslate2; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

GROVE_GPU=2 bash deploy/run.sh asr python -m lab speech \
  --video data/media/IS1009a_01.mp4 --config configs/whisper_pyannote.json \
  --out runs/deploy-speech
```

检查 `runs/deploy-speech/result.json` 的 `status=completed`、非空 `segments`、`raw.words`、`raw.diarization`、`audio_sha256`。人工听几个发言，检查原文与起止时间。`audio.wav` 保留作核对。命令结束容器退出，ASR 模型不继续占显存；后续人物配置只引用该 JSON，不在 M3 环境再安装/调用 pyannote。

如果出现 cuBLAS/cuDNN 动态库错误，先检查镜像 `/opt/python-packages.txt`、入口中的 `LD_LIBRARY_PATH` 及驱动。entrypoint 已在 Python 启动前暴露 Torch wheel 携带的 NVIDIA 动态库；faster-whisper 当前 CUDA 路线需要 CUDA 12 + cuDNN 9。

## 8. 人物第二步：模型依赖检查 → 接入主循环

先检查导入、CUDA provider 和权重路径。示例 `GROVE_GPU=2` 同样须换成实际可用卡，或把核心服务放到其他内网机器；不要在只有一张卡的机器照抄 2。

```bash
GROVE_GPU=2 bash deploy/run.sh character python - <<'PY'
import torch, onnxruntime as ort
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
from insightface.app import FaceAnalysis
from speakerlab.process.processor import FBank
from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
assert torch.cuda.is_available()
assert 'CUDAExecutionProvider' in ort.get_available_providers()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(ort.get_available_providers())
app = FaceAnalysis(name='buffalo_l', root='/models/insightface',
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))
for model in app.models.values():
    assert model.session.get_providers()[0] == 'CUDAExecutionProvider', 'ONNX silently fell back to CPU'
voice = ERes2NetV2(feat_dim=80, embedding_size=192)
voice.load_state_dict(torch.load('/models/pretrained_eres2netv2.ckpt', map_location='cpu', weights_only=True))
print('face sessions and voice checkpoint ready; M3 inference still needs video acceptance')
PY

bash deploy/run.sh core python scripts/configure_offline.py \
  --stage characters --speech-input runs/deploy-speech/result.json \
  --accept-model-links --output configs/deploy-characters.local.json

GROVE_GPU=2 bash deploy/run.sh character python -m unittest \
  tests.modules.test_faces tests.modules.test_voices \
  tests.grove.test_full_loop_characters -v

GROVE_GPU=2 bash deploy/run.sh character python -m lab.full_loop \
  --video data/media/IS1009a_01.mp4 --config configs/deploy-characters.local.json \
  --output runs/deploy-characters --source IS1009a-central-01 \
  --query 'Who spoke, and what visual evidence supports the observations?'
```

`--accept-model-links` 是本次实验明确采纳 M3 候选等价关系的开关，目的是测合并后的 character 归属，**不代表关系已被人工确认**。不加则保留候选，脸与声音可以各自形成身份节点，但不会因该预测自动合并成一个人物。配置生成器默认不采纳。

此时实际流程：复用整段 ASR → 抽帧描述 → 动态 episode 关闭 → buffalo_l/HDBSCAN + ERes2NetV2 → M3 音画等价关系/描述 → API 视觉定位 → Grove 摘要/facts/topic → character 投影及语音/视觉 clip 证据 → 联合检索。M3 默认音视频采样仍沿用原处理器设置；本页通过短 episode 控制首次联调规模。显存不足时先查看 `nvidia-smi`、实际 episode 时长、代表脸数及 M3 峰值，再另设 `association.fps/max_pixels` 等参数进行新实验，不静默截断证据。

若要先排查单模块，可在主循环前用现有 `perceive` → `associate` 命令测试一个固定时间片，命令见 [RUNBOOK.md](RUNBOOK.md)。固定时间片只用于组件诊断；完整主循环仍让 Grove 动态产生边界。

## 9. 看什么才算这一阶段通过

```bash
bash deploy/run.sh core python - <<'PY'
import json
from pathlib import Path
for name in ('deploy-core', 'deploy-visual', 'deploy-characters'):
    path = Path('runs') / name / 'run_report.json'
    if not path.exists():
        print(name, 'not completed'); continue
    report = json.loads(path.read_text())
    types = sorted({e['node_type'] for q in report['queries'] for e in q['result']['evidence']})
    print(name, report['status'], report['stats'], 'retrieved types:', types)
PY
```

| 阶段 | 需要实际检查 |
|---|---|
| core | `checkpoint.json`、`hypergraph.json`、`index_manifest.json`、`run_report.json` 生成；动态边界、摘要/facts/topic 内容合理；查询有可追溯 evidence |
| visual | 图和索引内有真实 clip 向量；检查合适查询是否召回 `node_type=visual_clip`，与 fact 同属一个 top-k；不要求每个查询必然同时出现两种类型 |
| characters | `identity-*.json`、episode 取证及 M3 原始输出存在；检查聚类/声纹/候选与采纳状态、跨 episode 稳定人物、fact 的 `character_refs` 和具体 clip 证据 |
| 质量测试 | 人工复核的转录/发言、脸声等价关系、character 归属和事实证据作为真值，分别跑已有评估命令；连通成功不等于这些指标通过 |

短发言（少于 2 秒）、脸数不足或质量不够时保留 unknown 是合法结果，不能为了“有 character”放松规则后与原实验混比。M3 的等价关系也不等于逐帧 active-speaker 标签。

同一视频、模型、配置、输出目录重复执行主循环会恢复/复用缓存。切换视觉开关、人物策略、模型 revision、视频或采样参数时使用新的配置名及运行目录。ASR 独立命令要求新输出目录；已经成功的 ASR 不必重跑。`--query` 返回检索证据，尚不自动生成最终问答文本。

上游依据：[Qwen2.5-VL 模型](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[Qwen3 embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、[vLLM 0.10.2 参数源码](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/engine/arg_utils.py)、[SigLIP API](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/siglip)、[faster-whisper CUDA 依赖](https://github.com/SYSTRAN/faster-whisper#requirements)、[pyannote 离线模型](https://huggingface.co/pyannote/speaker-diarization-community-1)、[pyannote 4.0.3 依赖](https://github.com/pyannote/pyannote-audio/blob/4.0.3/pyproject.toml)、[M3 安装说明](https://github.com/ByteDance-Seed/m3-agent)、[ONNX Runtime CUDA 依赖](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

## 10. 上传视频的在线工作台

核心 API 与 SigLIP 启动完成后，可通过 `python -m lab.workbench` 启动时间线工作台；支持上传、实时阶段输出、历史快照、人物和带引用问答。人物版新增 `workbench` 镜像，将 ASR 放进独立 venv，由任务按上传视频自动执行。完整构建/启动/测试命令见 [工作台部署与使用](WORKBENCH.md)。
