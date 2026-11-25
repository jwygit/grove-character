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

- `perception.py` 使用 InsightFace buffalo_l 与 M3 的 ERes2NetV2/FBank。人脸已替换为 M3 原始 HDBSCAN（函数原样保存在 lab/reused/m3_face_clustering.py）：5 FPS，聚类前检测分数 >=0.8、原始 embedding 范数 >=20，min_cluster_size=20；聚类后严格筛选检测分数 >0.85、范数 >22，丢弃 -1 噪声，按检测分数再质量分数排序，每簇最多 3 张。给 Omni 的是排序第一张，与 M3 的 face_only 路径相同。没有新增 tracker 或主动说话检测器。
- 默认 `lab/m3.py` 使用 ByteDance-Seed/M3-Agent-Memorization 微调 checkpoint；`lab/reused/m3_prompt.py` 保留原始联合 prompt 字符串。输入顺序为提示、视频、Face features 与每节点一张代表脸、Voice features 分组 JSON。输出原始事件描述、高层结论和 face/voice Equivalence，按原任务做联合生成。基础 Qwen 的自定义 association-only prompt 不再是默认路径。
- 所有前端先使用共同 PCM 进行切片、>=2 秒筛选、ERes2NetV2/FBank 提取与归一化，再逐条按声音节点平均余弦相似度匹配（默认 >=0.6，最多 20 条匹配声纹），然后才把按 voice_id 汇总的发言交给 Omni/Gemini。ASR 的局部 speaker 标签不参与全局身份分组。与 M3 一样不单独筛除 overlap；各前端的标记保留供审计，不能只对 pyannote 一组额外过滤。重叠语音仍可能污染声纹，需要用指标衡量。
- 解析 `Equivalence: <face_N>, <voice_N>` 形成候选图边，其余描述/结论保持来源类型。兼容上游 prompt 的单数 `video_description` 和读取端复数 `video_descriptions`，不一致时拒绝；虚构标签、畸形等价句、本次输出一声多脸冲突报错。同一人物多个声音节点可指向同一脸。模型没有声称逐句是否露脸，本路径不生成对应标签；无合格声音时仍调用模型生成视觉记忆。短句不创建声纹身份，文字及 unknown 保留。
- `--accept-model-links` 是实验策略开关：允许模型预测成为图内 confirmed 边，不代表人工确认或真实正确。默认只记录 candidate，不合并 face/voice；两种策略可做精度/覆盖率对照。
- 人脸/声音节点匹配均使用当前特征与节点保留特征的所有两两余弦相似度的平均，分别取 >=0.3 / >=0.6 的最高者，不使用 margin。人脸/声音的匹配特征池上限分别为 10 / 20，超过时按 M3 的随机采样方式更新；本实现加固定 seed 便于复现，完整观测仍保留在审计图。均限定相同模态、模型和维度，不直接比较脸与声音的 embedding。

Grove 适配差异：聚类范围是给定 episode 而非 M3 固定 30 秒；局部 face_id 仍作为 AV 可审计引用，随后映射为全局人物，不复用 M3 的全局节点编号；裁剪框增加越界保护。少于 20 个合格检测时不会绕过 HDBSCAN 创建人脸身份，单簇/短暂人物可能被拒识。这些行为需要在真实数据中评测，不能声称整个 M3 流水线已等价复现。旧阈值去重版本只保留在 Git 历史，不维护第二套运行分支。

声音适配差异保留：浮点秒数、episode 边界裁剪、短句文字保留、不从身份等价关系反推逐句画内可见性、可复现采样 seed，以及统一纯音频 ASR 输入。这些规则对所有待测前端相同。没有声源分离模型；声音节点是逐条阈值归并，不是人脸的 HDBSCAN。与 M3 对齐的是声纹后处理及先分组后关联的顺序，不声称替换后的所有模型输出完全等价。

指标分清输出语义：新增 `evaluate-equivalence` 在 registry 的全部 face × voice 组合上评估身份是否等价，分别报告本次模型预测与最终图映射的 precision/recall/F1 和 pair accuracy；真值绑定 bundle 与 registry 哈希。没有脸或声音时为零对，不能把未检出样本当作成功。前端覆盖、短句 unknown 和最终写入归属一起报告。M3 不输出 active-speaker/逐句画内可见性，原时间积分 AV 指标仅供旧协议适配器使用，不能混入 M3 结果。

人物图采用本项目有证据/版本的 confirmed-edge 连通分量及稳定 person anchor，而非 M3 重建 character_N 的编号方式。未复制 M3 的语义记忆边权、最高权冲突选择和完整检索图；跨 episode 矛盾预测可能产生错误合并，需通过人物标签审计、撤销错误边并重建快照。当前严格解析只拒绝同一次输出的一声多脸，不等于已解决跨片段冲突。新增身份评分也不替代跨 episode 错误合并/错误拆分评测。

## 第三阶段：Grove character 归属

固定 caption、episode 边界、writer 模型、prompt 模板和解码参数。比较：

- G0：相同新 writer、同一 M3 原始描述，无人物身份；speaker 分组保留，发言和 feature→character 映射均设 null，转写文本重新生成。
- G1：同样的发言 evidence，使用预测人物图。
- G2：同样的 evidence，使用人工校正人物图（归属上限，用于定位前端错误）。

这一轮不会把 Grove 原始 writer 与新的简化 writer 直接比较后把收益全归于人物模块。新 writer 覆盖本实验 speaker 和正文 referenced 引用，不支持 Grove 所有 dataset 专用字段。后续获得缺失上游文件后，再测原始 fact extractor 的增量 prompt。

模型选择事实对应的 `utterance_ids` / `memory_ids`，正文引用提供的 feature 标签；代码用同一版本的 feature→character 映射替换完整标签，避免 face_1/face_10 的子串误替换。摘要、事实和 M3 原始描述转写均使用此映射。speaker 引用来自发言证据，正文标签另记 referenced，不推导 actor/mentioned。事实中每个标签必须有被引用的发言/记忆支持；虚构 ID 和无证据标签报错，但这**不能保证引用在语义上正确**。

这是写入时适配：M3 原本存储 feature 标签并在检索时替换为 character；Grove 存储转写后的正文，同时保存原始输出、原文、feature_mapping 和 graph_revision。人物图未来合并或撤销不会悄悄改写旧事实，需要显式重放/迁移；此版不自动批量更新历史文字。

人工将不同组产生的事实对齐到共同语义 unit_id，确定 role 与 canonical person；预测 ID 到 canonical person 的映射须来自预设注册/独立身份标注，不能直接读取本条目标答案。漏事实计漏项，多余事实计额外输出；同一 unit/role 不允许重复掩盖错误。报告归属 precision/recall/F1、unit accuracy、覆盖率和 unknown 误绑定率。actor 与 speaker 是不同评价键。文本事实本身的忠实度、完整性需另做人工审查，不能由 character 指标替代。

## 数据与决策

现有 AMI 两段分别取同一会议 60–120s、180–240s；低分辨率可能影响 face 质量，不能从这两段推断家庭摄像头或中文会议效果。
建议先人工标注 10–20 分钟开发集，包含短发言、打断/重叠、画外音、噪声与多人同框，再收集独立 session 留出测试集。身份注册历史可以提供给后续时间段，但不能用测试标签调阈值。按 session 划分；顺序重放每个 recording 的 episode，记录 graph revision。

第一轮先做可行性筛选，不预设胜者。开发集确定可接受的 character precision 下限及延迟/费用约束，再在留出集比较召回、CER/WER 与资源消耗。最终需按 session 做配对统计/置信区间；本版尚未实现 bootstrap 汇总，单条视频 report 不是统计显著性结论。

跨边界 utterance 保留稳定 ID，观测切片另建 ID；有词级对齐时只保留当前 episode 的词。没有词对齐时部分切片保留音频及 ID，文字暂留空，原全文保留在语音结果中，避免把整句复制进两段记忆。
