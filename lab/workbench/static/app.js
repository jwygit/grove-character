"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  run: null,
  stream: null,
  seq: 0,
  events: [],
  snapshot: null,
  version: undefined,
  processed: 0,
  poll: null,
  snapshotBusy: false,
  asking: false,
};
const terminal = new Set(["complete", "failed", "stopped", "interrupted"]);
const labels = {
  queued: "排队中",
  running: "处理中",
  complete: "已完成",
  failed: "失败",
  stopped: "已停止",
  interrupted: "已中断",
  started: "开始",
  completed: "完成",
  response: "模型响应",
  artifact: "结果文件",
  pending: "待关闭",
  committed: "已提交",
  ready: "可查询",
  run: "任务",
  media: "视频校验",
  speech: "语音分段 / 转录",
  frames: "抽帧",
  caption_frames: "帧描述",
  boundary_stream: "动态边界判断",
  episode_summary: "Episode 摘要",
  fact_extract: "事实提取",
  character_facts: "人物事实转写",
  topic_create: "创建话题",
  topic_match: "匹配话题",
  topic_update: "更新话题",
  episode_roles: "话题角色",
  text_embedding: "文本向量",
  visual_embedding: "图像向量",
  face_voice: "人脸 / 声纹取证",
  association: "脸声关联",
  grounding: "视觉证据定位",
  identity: "人物快照",
  episode: "Episode",
  window_transaction: "窗口事务",
  memory: "记忆",
  index: "检索索引",
  finalize: "末段写入",
  retrieval: "Grove 证据检索",
  evidence_answer: "带引用回答",
  question: "问答",
};
const label = (text) => labels[text] || text;
function time(value = 0) {
  value = Math.max(0, Math.floor(value));
  return `${Math.floor(value / 60)
    .toString()
    .padStart(2, "0")}:${(value % 60).toString().padStart(2, "0")}`;
}
function element(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function button(text, action, className) {
  const el = element("button", text, className);
  el.type = "button";
  el.addEventListener("click", action);
  return el;
}
function showError(error) {
  $("error").hidden = !error;
  $("error").textContent = error?.message || String(error || "");
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const value = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : JSON.stringify(value.detail),
    );
  }
  return response.json();
}
const post = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
function guard(fn) {
  return (...args) =>
    Promise.resolve()
      .then(() => fn(...args))
      .catch(showError);
}
function seek(seconds) {
  $("video").currentTime = Math.max(0, seconds);
  $("video").scrollIntoView({ behavior: "smooth", block: "center" });
  updateSnapshot();
}
function scope() {
  return $("mode").value === "latest" ? null : $("video").currentTime;
}
function detail(title, value) {
  $("detail-title").textContent = title;
  $("detail-media").replaceChildren();
  $("detail-text").textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
  if (!$("detail").open) $("detail").showModal();
}
async function artifact(item) {
  const run = state.run;
  const response = await fetch(item.url);
  if (!response.ok) throw new Error("证据文件不可用或已改变");
  if (run !== state.run) return;
  const type = response.headers.get("content-type") || "";
  detail(item.name, "");
  if (type.includes("json"))
    $("detail-text").textContent = JSON.stringify(
      await response.json(),
      null,
      2,
    );
  else if (
    type.startsWith("image/") ||
    type.startsWith("audio/") ||
    type.startsWith("video/")
  ) {
    const media = element(
      type.startsWith("image/")
        ? "img"
        : type.startsWith("audio/")
          ? "audio"
          : "video",
    );
    media.src = item.url;
    media.alt = item.name;
    if (media.tagName !== "IMG") media.controls = true;
    $("detail-media").append(media);
  } else
    $("detail-text").textContent = (await response.text()).slice(0, 300000);
}
function controls(run) {
  $("stop").disabled = terminal.has(run.status) || !!run.stop;
  $("stop").textContent = run.stop ? "等待安全停止" : "请求停止";
  $("resume").disabled = !["failed", "stopped", "interrupted"].includes(
    run.status,
  );
  $("ask").disabled = state.asking;
}
async function listRuns() {
  const current = state.run;
  const runs = await api("/api/runs");
  $("runs").replaceChildren(new Option("选择任务", ""));
  for (const run of runs)
    $("runs").add(
      new Option(
        `${label(run.status)} · ${run.profile} · ${run.id.slice(0, 8)}`,
        run.id,
      ),
    );
  $("runs").value = current || "";
  return runs;
}
async function refreshRun() {
  const id = state.run;
  if (!id) return;
  const run = await api(`/api/runs/${id}`);
  if (id !== state.run) return;
  state.current = run;
  $("video-name").textContent = run.video.name;
  $("run-state").textContent = label(run.status);
  const option = [...$("runs").options].find((o) => o.value === id);
  if (option)
    option.textContent = `${label(run.status)} · ${run.profile} · ${id.slice(0, 8)}`;
  $("audio-note").hidden = !run.offline_audio_context;
  controls(run);
  const ready = run.snapshots.filter((s) => s.ready);
  const coverage = ready.at(-1)?.coverage || 0;
  const committed = run.snapshots.at(-1);
  for (const kind of ["play", "process", "memory"])
    $(`${kind}-progress`).max = run.video.duration;
  $("memory-progress").value = coverage;
  $("memory-time").textContent = time(coverage);
  $("pending").textContent =
    run.error ||
    (committed && !committed.ready
      ? "图已提交，正在准备索引。"
      : terminal.has(run.status)
        ? `任务${label(run.status)}；可选择播放位置回看记忆。`
        : "未关闭的 episode 仍在等待边界；停止会等待当前窗口完成。");
  await updateSnapshot();
}
async function selectRun(id) {
  if (state.stream) state.stream.close();
  clearInterval(state.poll);
  state.run = id;
  state.current = null;
  state.seq = 0;
  state.events = [];
  state.version = undefined;
  state.snapshot = null;
  state.processed = 0;
  state.snapshotBusy = false;
  $("events").replaceChildren();
  $("answers").replaceChildren();
  $("episodes").replaceChildren();
  $("people-list").replaceChildren();
  $("memory-list").replaceChildren();
  $("process-progress").value = 0;
  $("process-time").textContent = "00:00";
  $("event-count").textContent = "0 条";
  showError(null);
  if (!id) {
    $("video").removeAttribute("src");
    $("video").load();
    $("video-empty").hidden = false;
    $("ask").disabled = true;
    $("stop").disabled = true;
    $("resume").disabled = true;
    return;
  }
  localStorage.setItem("grove-run", id);
  $("runs").value = id;
  const run = await api(`/api/runs/${id}`);
  if (id !== state.run) return;
  $("video-empty").hidden = true;
  $("video").src = `/api/videos/${run.video_id}/media`;
  $("video").load();
  await refreshRun();
  const answers = await api(`/api/runs/${id}/questions`);
  if (id !== state.run) return;
  for (const answer of answers) renderAnswer(answer);
  connect(id);
  state.poll = setInterval(guard(refreshRun), 1500);
}
function connect(id) {
  if (state.stream) state.stream.close();
  const source = new EventSource(`/api/runs/${id}/events?after=${state.seq}`);
  state.stream = source;
  source.onopen = () => {
    $("connection").textContent = "阶段事件已连接";
  };
  source.onmessage = (message) => {
    if (id !== state.run) return;
    const event = JSON.parse(message.data);
    if (event.seq <= state.seq) return;
    state.seq = event.seq;
    state.events.push(event);
    if (state.events.length > 300) state.events.shift();
    if (event.stage === "window_transaction" && event.state === "completed") {
      state.processed = Math.max(state.processed, event.input_frontier || 0);
      $("process-progress").value = state.processed;
      $("process-time").textContent = time(state.processed);
    }
    renderEvent(event);
    if (event.stage === "run" || event.state === "ready") guard(refreshRun)();
  };
  source.onerror = () => {
    if (id !== state.run) return;
    if (terminal.has(state.current?.status)) {
      source.close();
      if (state.stream === source) state.stream = null;
      $("connection").textContent = "历史事件已保存";
    } else $("connection").textContent = "连接中断，自动续传事件…";
  };
}
function renderEvent(event) {
  const box = element("div", undefined, `event ${event.state}`);
  const head = element("div", undefined, "event-head");
  head.append(
    element("strong", label(event.stage)),
    element("span", label(event.state)),
  );
  box.append(head);
  const info = [
    `#${event.seq}`,
    event.media_end !== undefined
      ? `${time(event.media_start)}–${time(event.media_end)}`
      : "",
    event.elapsed !== undefined ? `${event.elapsed.toFixed(2)}s` : "",
    event.version ? `v${event.version}` : "",
    event.model || "",
  ].filter(Boolean);
  box.append(element("p", info.join(" · "), "event-meta"));
  if (event.error) box.append(element("p", event.error, "event-meta"));
  for (const item of event.artifacts || [])
    box.append(
      button(
        item.name,
        guard(() => artifact(item)),
        "artifact",
      ),
    );
  box.append(
    button("事件详情", () => detail(label(event.stage), event), "artifact"),
  );
  $("events").classList.remove("empty");
  $("events").prepend(box);
  while ($("events").children.length > 300) $("events").lastChild.remove();
  $("event-count").textContent =
    `最近 ${state.events.length} 条 · #${state.seq}`;
}
async function updateSnapshot() {
  if (!state.run || !state.current || state.snapshotBusy) return;
  const playback = scope();
  const snapshots = state.current.snapshots.filter(
    (s) => s.ready && (playback === null || s.frontier <= playback),
  );
  const version = snapshots.at(-1)?.version || null;
  if (state.version === version) return;
  const id = state.run;
  state.snapshotBusy = true;
  try {
    const value = await api(
      `/api/runs/${id}/snapshot${version ? `?version=${version}` : `?playback_time=${playback ?? 0}`}`,
    );
    if (id !== state.run) return;
    state.version = value.version;
    state.snapshot = value;
    renderSnapshot(value);
  } finally {
    if (id === state.run) state.snapshotBusy = false;
  }
}
function renderSnapshot(value) {
  const graph = value.graph;
  for (const key of ["episodes", "people-list", "memory-list"]) {
    $(key).replaceChildren();
    $(key).classList.remove("empty");
  }
  $("person-filter").replaceChildren(new Option("全部人物", ""));
  if (!graph) {
    $("snapshot-label").textContent =
      "当前播放位置尚无已提交且可查询的记忆。可继续播放或切换到最新记忆。";
    $("episodes").append(
      element("p", "等待动态 episode 关闭与索引就绪。", "empty"),
    );
    $("people-list").append(element("p", "当前版本暂无人物关联", "empty"));
    $("memory-list").append(element("p", "当前版本暂无记忆", "empty"));
    return;
  }
  $("snapshot-label").textContent =
    `版本 v${value.version} · 记忆覆盖至 ${time(value.coverage)} · 观察到 ${time(value.input_frontier)} 时形成`;
  const origin = value.origin;
  for (const episode of Object.values(graph.episodes).sort(
    (a, b) => a.start_seconds - b.start_seconds,
  )) {
    const el = button(
      "",
      () => {
        seek(episode.start_seconds - origin);
        detail("Episode · " + episode.id, episode);
      },
      "episode",
    );
    el.append(
      element(
        "span",
        `${time(episode.start_seconds - origin)}–${time(episode.end_seconds - origin)}`,
      ),
      element("strong", episode.summary || episode.description || episode.id),
    );
    $("episodes").append(el);
  }
  const people = graph.character_catalog || {};
  for (const id of Object.keys(people))
    $("person-filter").add(new Option(id, id));
  for (const person of value.people) {
    const card = element("article", undefined, "card");
    card.append(element("h3", person.episode_id));
    const faces = element("div", undefined, "face-row");
    for (const face of person.faces) {
      const fig = element("figure", undefined, "face");
      const img = element("img");
      img.src = face.media.url;
      img.alt = face.label;
      img.loading = "lazy";
      fig.append(img, element("figcaption", face.label));
      faces.append(fig);
    }
    card.append(faces);
    for (const [tag, mapping] of Object.entries(person.mapping)) {
      card.append(
        element(
          "p",
          `${tag} (${mapping.modality}) → ${mapping.character_id || "unknown"}`,
          "mapping",
        ),
      );
    }
    card.append(
      element(
        "p",
        `人物图修订 ${person.graph_revision} · ${person.equivalences_accepted ? "按策略采纳模型关联，未经人工核验" : "模型候选关联，未采纳"}`,
        "muted",
      ),
    );
    for (const relation of person.equivalences)
      card.append(element("p", `${relation.face} ↔ ${relation.voice}`));
    for (const ref of person.refs) {
      card.append(
        button(
          `${time(ref.start - origin)}–${time(ref.end - origin)} · ${ref.feature_id || "unknown voice"} · ${ref.text || "无可对齐文字"}`,
          () => {
            seek(ref.start - origin);
            detail("发言证据 · " + (ref.character_id || "unknown"), ref);
          },
          "utterance",
        ),
      );
    }
    card.append(
      button(
        "关联证据与人物图",
        guard(() => artifact(person.artifact)),
        "artifact",
      ),
    );
    $("people-list").append(card);
  }
  if (!value.people.length)
    $("people-list").append(
      element("p", "该版本未启用人物模块，或尚无可确认关联。", "empty"),
    );
  for (const [key, title] of [
    ["facts", "FACT"],
    ["visual_clips", "VISUAL CLIP"],
    ["topics", "TOPIC"],
  ])
    for (const row of Object.values(graph[key] || {})) {
      const card = element("article", undefined, "card");
      card.append(
        element("h3", `${title} · ${row.id}`),
        element(
          "p",
          row.content || row.summary || row.caption || row.title || "",
        ),
      );
      if (row.character_refs?.length)
        card.append(
          element(
            "p",
            row.character_refs
              .map((r) => `${r.role}: ${r.character_id}`)
              .join(" · "),
            "muted",
          ),
        );
      card.append(
        button("查看结构化证据", () => detail(title, row), "artifact"),
      );
      if (key === "visual_clips") {
        const frame = value.frames[row.frame_time_span];
        if (frame)
          card.append(
            button(
              "查看帧",
              guard(() => artifact(frame)),
              "artifact",
            ),
          );
        card.append(
          button(
            `回看 ${time(row.start_seconds - origin)}`,
            () => seek(row.start_seconds - origin),
            "artifact",
          ),
        );
      }
      for (const ref of row.evidence_refs || [])
        if (Number.isFinite(ref.start))
          card.append(
            button(
              `${ref.modality || ref.kind || "证据"} ${time(ref.start - origin)}–${time(ref.end - origin)}`,
              () => seek(ref.start - origin),
              "artifact",
            ),
          );
      $("memory-list").append(card);
    }
}
function renderAnswer(answer) {
  const card = element("article", undefined, "answer");
  card.dataset.version = answer.memory_version || "";
  card.append(
    element("h3", answer.question),
    element("p", answer.answer, "answer-body"),
  );
  const meta = `${answer.memory_version ? `记忆 v${answer.memory_version}` : "暂无记忆"} · ${answer.mode === "latest" ? "最新版本" : `提问位置 ${time(answer.playback_time)}`} · 覆盖至 ${time(answer.coverage)}${answer.insufficient_evidence ? " · 证据不足" : ""}`;
  card.append(element("p", meta, "answer-meta"));
  const citations = element("div", undefined, "citations");
  for (const citation of answer.citations) {
    const span = citation.spans[0];
    citations.append(
      button(
        `[${citation.citation_id}] ${citation.node_type} ${span ? `${time(span.start)}–${time(span.end)}` : "结构化证据"}`,
        () => {
          if (span) seek(span.start);
          detail(
            `引用 ${citation.citation_id} · 原回答 v${answer.memory_version}`,
            citation,
          );
        },
        "citation",
      ),
    );
  }
  card.append(
    citations,
    button("完整检索结果", () => detail("回答与检索证据", answer), "artifact"),
  );
  $("answers").prepend(card);
}
async function upload() {
  const file = $("file").files[0];
  if (!file) return;
  showError(null);
  $("upload").disabled = true;
  try {
    const uploaded = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/videos?filename=${encodeURIComponent(file.name)}`);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = (e) => {
        $("upload-status").textContent = e.lengthComputable
          ? `上传 ${Math.round((e.loaded / e.total) * 100)}%`
          : "上传中";
      };
      xhr.onload = () => {
        let value;
        try {
          value = JSON.parse(xhr.responseText);
        } catch {
          return reject(new Error("服务器未返回有效 JSON"));
        }
        if (xhr.status < 300) resolve(value);
        else reject(new Error(value.detail || "上传失败"));
      };
      xhr.onerror = () => reject(new Error("上传连接失败"));
      xhr.send(file);
    });
    $("upload-status").textContent = "校验完成，创建任务…";
    const run = await post("/api/runs", {
      video_id: uploaded.id,
      profile: $("profile").value,
    });
    await listRuns();
    await selectRun(run.id);
    $("upload-status").textContent = "已加入处理队列";
  } finally {
    $("upload").disabled = false;
  }
}
$("file").addEventListener("change", () => {
  $("upload").disabled = !$("file").files.length || !$("profile").value;
});
$("upload").addEventListener("click", guard(upload));
$("runs").addEventListener(
  "change",
  guard((e) => selectRun(e.target.value)),
);
$("refresh").addEventListener(
  "click",
  guard(async () => {
    await listRuns();
    await refreshRun();
  }),
);
$("stop").addEventListener(
  "click",
  guard(async () => {
    await post(`/api/runs/${state.run}/stop`);
    await refreshRun();
  }),
);
$("resume").addEventListener(
  "click",
  guard(async () => {
    await post(`/api/runs/${state.run}/resume`);
    await refreshRun();
    connect(state.run);
  }),
);
$("close-detail").addEventListener("click", () => $("detail").close());
$("detail").addEventListener("close", () => {
  $("detail-media").querySelectorAll("audio,video").forEach((media) => media.pause());
});
$("mode").addEventListener("change", guard(updateSnapshot));
$("video").addEventListener("timeupdate", () => {
  $("play-progress").value = $("video").currentTime;
  $("play-time").textContent = time($("video").currentTime);
  guard(updateSnapshot)();
});
document.querySelectorAll("[data-tab]").forEach((tab) =>
  tab.addEventListener("click", () => {
    document
      .querySelectorAll("[data-tab]")
      .forEach((b) => b.setAttribute("aria-selected", String(b === tab)));
    document
      .querySelectorAll(".tab-pane")
      .forEach((p) => (p.hidden = p.id !== tab.dataset.tab));
  }),
);
$("question-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guard(async () => {
    if (!state.run || state.asking) return;
    const question = $("question").value.trim();
    if (!question) return;
    const run = state.run;
    state.asking = true;
    $("ask").disabled = true;
    $("question-state").textContent = "正在从所选快照检索并生成带引用回答…";
    showError(null);
    try {
      const answer = await post(`/api/runs/${run}/questions`, {
        question,
        playback_time: scope(),
        characters: $("person-filter").value ? [$("person-filter").value] : [],
      });
      if (run === state.run) {
        renderAnswer(answer);
        $("question").value = "";
        $("question-state").textContent =
          "回答已保存。引用回看不会修改原回答的记忆版本。";
      }
    } catch (error) {
      $("question-state").textContent = "问答失败；没有生成替代答案。";
      throw error;
    } finally {
      state.asking = false;
      $("ask").disabled = !state.run;
    }
  })();
});
guard(async () => {
  const profiles = await api("/api/profiles");
  for (const p of profiles)
    $("profile").add(
      new Option(`${p.description}${p.characters ? " · 含人物" : ""}`, p.id),
    );
  const runs = await listRuns();
  $("connection").textContent = "服务已连接";
  const saved = localStorage.getItem("grove-run");
  if (runs.some((r) => r.id === saved)) await selectRun(saved);
})(); // No seeded captions, characters, answers or synthetic production data.
