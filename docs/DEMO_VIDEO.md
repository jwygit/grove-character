# 中文多人访谈演示素材

使用《圆桌派》第二季第 11 期“AI：圆桌新嘉宾，空前不绝后”的优酷纪实频道发布版本。来源为 [YOUKU DOCUMENTARY 视频](https://www.youtube.com/watch?v=vlEbfmQXC7I)，平台标题为“当 AI 越来越懂人类，我们会不会被机器控制？”。这是现成节目，不是生成视频。

## 取文件

| 版本 | 文件 | 用途 |
|---|---|---|
| 连续 8 分钟短片（80.02 MiB） | `data/media/yuanzhuopai-s2e11-ai-short.mp4` | 随 Git 仓库提供，先做上传、ASR、人物和问答联调 |
| 43:59 整期发布版（371.03 MiB） | `data/media/yuanzhuopai-s2e11-ai-full.mp4` | [GitHub Release 附件](https://github.com/jwygit/grove-character/releases/tag/demo-yuanzhuopai-s2e11-v1)，检验长视频、跨 episode 人物与观点检索 |

两份均为 **1280×720 / 30 FPS / H.264 / AAC / MP4**，保留原节目音轨、画面和烧录字幕，并设置 faststart。整期不重新编码画面；短片连续截取整期的 **00:30–08:30**，不拼接、不变速。节目有多位嘉宾、桌面机器人及反复切换的近景。

在项目根目录执行（Python 3.11+，脚本只用标准库）：

```bash
git pull --ff-only

# 短片已随仓库提供；只验证，不联网。
python scripts/fetch_demo_video.py --variant short --verify-only

# 联网机取整期；下载完自动校验大小和 SHA256。
python scripts/fetch_demo_video.py --variant full

# 复制整个项目和整期 MP4 到内网之后，再离线校验。
python scripts/fetch_demo_video.py --variant full --verify-only
```

也可从 Release 页面手动下载到 `data/media/`。自定义存放位置用 `--output-dir /path/to/media`。断网或校验失败不会留下可误用的成品；已有文件不被覆盖，损坏副本应先移走再下载。当前下载器不支持断点续传。完整大小、SHA256、编码信息、剪辑范围和授权来源见 [素材清单](../data/yuanzhuopai-source-manifest.json)。无需在内网安装 yt-dlp 或访问原视频平台。

## 在工作台演示

按 [工作台部署手册](WORKBENCH.md) 启动真实服务，在浏览器上传短片。用于上传的文件须在运行浏览器的电脑上；仅放在服务器磁盘不会自动出现在浏览器文件选择框中。

先跑核心配置，观察抽帧描述、动态 episode、摘要、facts、topic、visual clip 检索和带引用问答；再为同一短片新建开启 ASR/人物的任务，观察 face/voice → character、发言时间及 fact→clip 证据。模型和依赖仍按部署文档准备，素材下载脚本不会安装推理模型。

- 上传短片后，时间轴从 **00:00** 开始；对应整期时间 = 短片时间 + **30 秒**。不要把整期时间直接当作短片标注。
- 选择不同人物，检查跨镜头是否仍映射到同一 character；名字不能仅凭人脸自动当成真值。
- 检查机器人、画外音和无法归属的发言，是否保持独立/unknown，而非强行归给画面中的人脸。
- 提问可从“刚才大家怎样和机器人互动”“谁提出了这个问题”“同一个人后来补充了什么”开始，再展开引用回看。具体答案、人物和时间要人工核对，以上不是已标注问答集。
- 历史提问受已提交记忆版本限制，播放进度领先时可能暂无可用证据；整段 ASR 仍属于离线处理。

先用 8 分钟验证流程，再上传整期。原片含烧录字幕，视觉模型可能读到台词；不能据此单独断言 ASR 或音画关联准确率。当前只抽查画面、验证媒体及上传播放链路，未完整人工标注，也未在本机跑该节目真实 GPU 模型准确率测试。

本次媒体验收：两份 MP4 均通过 `ffmpeg -v error -xerror` 全程音视频解码，确认 `moov` 位于 `mdat` 前。8 分钟真实素材通过工作台上传及上传后 SHA256 核对，HTTP Range 在文件开头和末尾均返回正确字节；未启动模型 worker。下载工具的 3 项独立测试验证成功下载/离线复验、错误内容不发布且清理临时文件、已有损坏文件不被覆盖。可运行 `python -m unittest tests.modules.test_demo_video -v` 复验工具行为。

## 来源与公开分发记录

2026-11-26，项目所有者明确确认已有本视频公开再分发授权；此次仓库短片和 Release 整期发布依据该确认。该记录是项目所有者的授权声明，不表示已独立审核授权文件，也不把节目改为开源或 Creative Commons 许可。节目版权保留于原权利人，项目代码许可不自动覆盖视频素材。

旧推荐 YouTube 链接 `UaqMBHZUZuU` 已不可用，优酷旧页面下载受限。本次实际获取的官方频道发布版 ID 为 `vlEbfmQXC7I`，时长约 **43:59**；旧优酷播单标注的 **45:53** 属于不同发布版本。此次文件和全部时间映射以素材清单的实际媒体为准，不声称二者逐帧相同。

下载工具为 yt-dlp 2026.11.23，选取视频格式 136 和音频格式 140；原平台签名媒体地址、下载中间文件和原始 info JSON 不进入 Git。整期超过 GitHub 单文件 100 MiB 限制，故存于固定 Release 标签下；短片直接随仓库提供。
