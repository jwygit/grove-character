# 实验设计

## 假设与变量

联合方案借鉴 M3 的“人脸特征 + 声纹 + 音画证据 + 跨片段身份图”，保留 Grove 的动态 episode 组织。人物图独立于 episode：边界划分不重置身份，身份关联也不要求每个 episode 只有一个人。

主要假设：

1. 专用本地 ASR/diarization 能替代 Gemini 语音前端，但是否保持说话轮次纯度和下游人物归属，需要测试，不能只看文字准确率。
2. 在 episode 边界确定后汇总人物证据，能保留完整事件上下文；相比逐窗口关联可能改善准确率，但增加等待时间。先固定边界测试，不同时改变 Grove 分段模型。
3. 将结构化人物证据交给写入器并保存到 fact，可减少姓名漂移、speaker/actor 混淆；错误身份也可能被固化，因此必须测错误归属和 unknown 误绑定。

运行依赖：原视频/音频 → 并行 caption 与人物证据 → 固定本次 episode 时间范围 → AV 关联及历史人物检索 → 摘要 → facts → Grove 图。
第一版实验按已有边界离线重放，计算顺序不等同于实时调度已完成。真实流式 provider 将来可等待 evidence watermark；不能用未来 episode 的人物图反向改善过去预测后仍声称在线准确率。

## 第一阶段：语音分段与转录

| 方案 | 实际适配器 | 定位 |
|---|---|---|
| Gemini | 音频 generateContent，输出浮点秒数 start/end/text，不要求人物 ID | 闭源对照 |
| faster-whisper + pyannote community-1 | 词级 ASR 时间戳与 diarization 区间对齐，按 speaker 变化合并成轮次 | 多语言本地候选 |
| FunASR Paraformer-zh + FSMN-VAD + 标点 + CAM++ | 读取 sentence_info 的毫秒时间戳与局部 speaker 标签 | 中文候选 |

这不是严格复现 M3：M3 将视频送 Gemini、使用 MM:SS 并丢弃短声音片段。本轮统一同一个 ffmpeg 解码 WAV，保留短转录，不因声纹不可用而删文字，避免视听输入差异混入 ASR 对比。M3 原始视频提示路径可另设补充实验，不能与音频输入对照混称同条件。

本地方案借鉴 WhisperX 的 ASR/diarization 分工，但本实现使用 faster-whisper 原生词时间戳，**没有声称实现 WhisperX 强制对齐**。词与多个 speaker 区间相交时保留 unknown/overlap，不硬选最大重叠者。单个词跨轮次边界也可能被保守标记为 ambiguous。FunASR 的句级 speaker 输出不保证每处轮次变化均被切开，这正是待测项。

固定：音频哈希、解码方式、语言子集、标注及下游声纹模型。主指标：中文 CER / 英文 WER、轮次起止边界 F1（分别一对一匹配，容差 250ms）、语音覆盖 F1、漏语音秒数、分段说话人纯度、混入其他说话人的秒数。纯度按每个预测段覆盖的单人真值时长计算，不包含重叠真值；重叠时长单独报告。长段/静音不能只靠纯度获益，必须同时看边界与覆盖。

WER/CER 按时间排序后连接全文，使用 NFKC、小写、去标点/符号；CER 去空格，WER 用空白分词。中文 WER 和多人重叠全文串行顺序不是可靠主指标。无参考文字的错误率返回 null，保留插入数量。这里没有把这些指标误称为 DER；Gemini 未生成 speaker 标签，标准 DER 不是主对照项。

## 第二阶段：episode 后音画关联

先复用同一份 episode 划分与同一套 face gallery，使用人工校正的语音轮次测试关联能力；再换成各语音前端的预测轮次，测误差传递。必须冻结视觉证据，不能给某组额外人工修正人脸。

- `perception.py` 实际适配 InsightFace buffalo_l 与 M3 使用的 ERes2NetV2/FBank；采用逐帧抽样及余弦聚类，不是连续 tracker，不是主动说话检测器，也不等同于 M3 的 HDBSCAN。第一轮所有组固定此实现。
- 复用已有 AV 适配器：原视频片段 + 人脸参考图 + 带时间的 utterances → face_id 或 null。支持 Gemini 和本地 Qwen2.5-Omni；换 M3 微调权重时须单独标明，当前 association-only prompt 不是 M3 原始记忆任务 prompt，适配收益未知。
- 关联预测输出必须覆盖全部 utterance，重复、漏项或虚构 face ID 均报错。无脸语音仍可通过历史声纹关联人物，不等于必须 unknown character。
- `--accept-model-links` 是实验策略开关：允许模型预测成为图内 confirmed 边，不代表人工确认或真实正确。默认只记录 candidate，不合并 face/voice；两种策略可做精度/覆盖率对照。
- 历史检索阈值默认不启用；需要显式配置 `reid_thresholds`，同模态、同模型空间比较，阈值与 margin 在开发集校准。脸与声音的 embedding 不直接计算相似度。

指标使用共同 face gallery 下的时间积分 precision/recall/F1、画外音误绑定率；这些指标可比较不同预测轮次。关联人工真值须包含漏识别语音；不能只标预测正确输出部分。人物图另看跨 episode 重现、错误合并、错误拆分及拒识，用带人工 canonical person 的单元表评测；当前 CLI 没有自动人物聚类全套指标，需追加标注与分析，不能把 AV F1 当作人物图准确率。

## 第三阶段：Grove character 归属

固定 caption、episode 边界、writer 模型、prompt 模板和解码参数。比较：

- G0：相同新 writer，无人物身份（speaker 标签保留，character 为 null）。
- G1：同样的发言 evidence，使用预测人物图。
- G2：同样的 evidence，使用人工校正人物图（归属上限，用于定位前端错误）。

这一轮不会把 Grove 原始 writer 与新的简化 writer 直接比较后把收益全归于人物模块。新 writer 仅覆盖本实验 speaker 归属，不支持 Grove 所有 dataset 专用字段。后续获得缺失上游文件后，再测原始 fact extractor 的增量 prompt。

模型只选择事实对应的 utterance_ids；代码解析其中的 character_id、图版本和证据，拒绝虚构 utterance_id。它能防止凭空生成 ID，但**不能保证模型引用了语义正确的发言**，因此仍需人工标注事实归属。

人工将不同组产生的事实对齐到共同语义 unit_id，确定 role 与 canonical person；预测 ID 到 canonical person 的映射须来自预设注册/独立身份标注，不能直接读取本条目标答案。漏事实计漏项，多余事实计额外输出；同一 unit/role 不允许重复掩盖错误。报告归属 precision/recall/F1、unit accuracy、覆盖率和 unknown 误绑定率。actor 与 speaker 是不同评价键。文本事实本身的忠实度、完整性需另做人工审查，不能由 character 指标替代。

## 数据与决策

现有 AMI 两段分别取同一会议 60–120s、180–240s；低分辨率可能影响 face 质量，不能从这两段推断家庭摄像头或中文会议效果。
建议先人工标注 10–20 分钟开发集，包含短发言、打断/重叠、画外音、噪声与多人同框，再收集独立 session 留出测试集。身份注册历史可以提供给后续时间段，但不能用测试标签调阈值。按 session 划分；顺序重放每个 recording 的 episode，记录 graph revision。

第一轮先做可行性筛选，不预设胜者。开发集确定可接受的 character precision 下限及延迟/费用约束，再在留出集比较召回、CER/WER 与资源消耗。最终需按 session 做配对统计/置信区间；本版尚未实现 bootstrap 汇总，单条视频 report 不是统计显著性结论。

跨边界 utterance 保留稳定 ID，观测切片另建 ID；有词级对齐时只保留当前 episode 的词。没有词对齐时部分切片保留音频及 ID，文字暂留空，原全文保留在语音结果中，避免把整句复制进两段记忆。
