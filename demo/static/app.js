const $ = (id) => document.getElementById(id);
const PLAYBACK_FPS = 10;
const ENCODED_FPS = 16;
let ws = null,
  session = null,
  generation = 0,
  uploadId = null,
  allPresets = [],
  currentState = "idle";
const held = new Set();
const controlKeys = new Set([
  "w",
  "a",
  "s",
  "d",
  "q",
  "e",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
]);
const labels = {
  waiting_connection: "等待浏览器连接",
  loading_model: "正在加载常驻模型",
  initializing: "正在编码首帧与描述",
  queued: "等待下一段",
  waiting_action: "等待动作输入 · WASD / QE / 方向键",
  generating: "正在生成",
  pausing: "本段完成后暂停",
  paused: "已暂停",
  stopped: "已停止",
  complete: "探索完成",
  disconnected: "连接已断开",
  error: "生成出错",
};
function error(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}
async function api(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw Error(e.detail);
  }
  return r.json();
}
function send(value) {
  if (ws?.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ ...value, generation }));
}
function actionText(a) {
  if (!a) return "等待下一段";
  const parts = [];
  if (a.forward)
    parts.push(
      `${a.forward > 0 ? "前进" : "后退"} ${(Math.abs(a.forward) * a.speed).toFixed(1)} m`,
    );
  if (a.right)
    parts.push(
      `${a.right > 0 ? "右移" : "左移"} ${(Math.abs(a.right) * a.speed).toFixed(1)} m`,
    );
  if (a.up)
    parts.push(
      `${a.up > 0 ? "上浮" : "下降"} ${(Math.abs(a.up) * a.speed).toFixed(1)} m`,
    );
  if (a.yaw)
    parts.push(`${a.yaw > 0 ? "右转" : "左转"} ${Math.abs(a.yaw).toFixed(0)}°`);
  if (a.pitch)
    parts.push(
      `${a.pitch > 0 ? "抬头" : "低头"} ${Math.abs(a.pitch).toFixed(0)}°`,
    );
  if (a.orbit) parts.push(`Orbit · 半径 ${a.orbit_radius.toFixed(1)} m`);
  return parts.join(" · ") || "静止";
}
class Player {
  constructor() {
    this.videos = [$("video0"), $("video1")];
    this.clear();
    this.videos.forEach((v) =>
      v.addEventListener("ended", () => {
        if (v === this.videos[this.front]) {
          this.waiting = true;
          this.advance();
        }
      }),
    );
  }
  clear() {
    this.token = (this.token || 0) + 1;
    this.queue = [];
    this.front = 0;
    this.waiting = true;
    this.loaded = null;
    this.loading = false;
    this.seen = new Set();
    this.videos.forEach((v) => {
      v.pause();
      v.removeAttribute("src");
      v.load();
      v.style.visibility = "hidden";
      v.style.zIndex = "1";
    });
    $("waitLabel").style.display = "none";
    $("playbackLabel").textContent = "首帧预览";
  }
  add(chunk) {
    if (this.seen.has(chunk.chunk_index)) return;
    this.seen.add(chunk.chunk_index);
    this.queue.push(chunk);
    this.queue.sort((a, b) => a.chunk_index - b.chunk_index);
    this.preload();
  }
  preload() {
    if (this.loading || this.loaded || !this.queue.length) return;
    this.loading = true;
    const token = this.token;
    const chunk = this.queue.shift();
    const v = this.videos[1 - this.front];
    v.src = chunk.url;
    v.defaultPlaybackRate = PLAYBACK_FPS / ENCODED_FPS;
    v.playbackRate = PLAYBACK_FPS / ENCODED_FPS;
    const ready = () => {
      v.removeEventListener("canplay", ready);
      if (token !== this.token) return;
      this.loading = false;
      this.loaded = { v, chunk };
      this.advance();
    };
    v.addEventListener("canplay", ready);
    v.onerror = () => {
      if (token === this.token) {
        this.loading = false;
        error("视频加载失败，请检查连接。");
      }
    };
    v.load();
  }
  async advance() {
    if (!this.waiting) return;
    if (!this.loaded) {
      $("waitLabel").style.display = session ? "block" : "none";
      this.preload();
      return;
    }
    const { v, chunk } = this.loaded;
    v.playbackRate = PLAYBACK_FPS / ENCODED_FPS;
    this.loaded = null;
    this.front = this.videos.indexOf(v);
    this.waiting = false;
    v.style.visibility = "visible";
    v.style.zIndex = "3";
    this.videos[1 - this.front].style.zIndex = "2";
    $("waitLabel").style.display = "none";
    $("playbackLabel").textContent =
      `播放第 ${chunk.chunk_index + 1} 段 · ${PLAYBACK_FPS} fps`;
    try {
      await v.play();
      send({
        type: "playback_started",
        chunk_index: chunk.chunk_index,
        client_time: Date.now() / 1000,
      });
    } catch (e) {
      error("浏览器暂停了播放，请点击画面继续。");
      this.waiting = true;
    }
    this.preload();
  }
}
const player = new Player();
function state(s) {
  currentState = s.state;
  $("state").textContent = labels[s.state] || s.state;
  $("progressText").textContent = `${s.chunks} / ${s.max_chunks} 段`;
  $("progressBar").style.width = `${(100 * s.chunks) / s.max_chunks}%`;
  $("currentAction").textContent = actionText(s.active);
  const completed = Math.max(0, Math.min(6, Number(s.completed_steps) || 0));
  $("stepCount").textContent = `${completed} / 6 步`;
  $("stepProgress").setAttribute("aria-valuenow", String(completed));
  [...$("stepProgress").children].forEach((segment, i) =>
    segment.classList.toggle("complete", i < completed),
  );
  $("nextAction").textContent = actionText(s.pending) === "静止"
    ? "等待动作输入" : actionText(s.pending);
  if (s.generation_s != null) {
    $("seconds").innerHTML = `${s.generation_s.toFixed(2)}<small> s</small>`;
    $("fps").innerHTML = `${s.generation_fps.toFixed(2)}<small> FPS</small>`;
  }
  const terminal = ["stopped", "complete", "disconnected", "error"].includes(
    s.state,
  );
  $("start").disabled = !terminal;
  $("pause").disabled = terminal || ["paused", "pausing"].includes(s.state);
  $("resume").disabled = !["paused", "pausing"].includes(s.state);
  $("reset").disabled = ["stopped", "disconnected", "error"].includes(s.state);
  $("stop").disabled = terminal;
  $("download").setAttribute("aria-disabled", String(s.chunks === 0));
  $("download").href =
    `/api/sessions/${session}/recording?generation=${generation}`;
  for (const id of ["preset", "upload", "prompt", "seed", "maxChunks"])
    $(id).disabled = !terminal;
  if (s.error) error(s.error);
}
function changePreset() {
  const p = allPresets.find((p) => p.id === $("preset").value);
  uploadId = null;
  $("prompt").value = p.prompt;
  $("sceneName").textContent = p.name;
  $("preview").src = p.image;
  $("preview").style.visibility = "visible";
  $("emptyState").style.display = "none";
}
$("preset").addEventListener("change", changePreset);
$("upload").addEventListener("change", async () => {
  const f = $("upload").files[0];
  if (!f) return;
  try {
    const r = await api("/api/upload", { method: "POST", body: f });
    uploadId = r.upload_id;
    $("preview").src = r.preview;
    $("preview").style.visibility = "visible";
    $("sceneName").textContent = f.name;
    error("");
  } catch (e) {
    error(e.message);
  }
});
$("start").addEventListener("click", async () => {
  try {
    error("");
    $("start").disabled = true;
    if (ws) ws.close();
    const s = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset: $("preset").value,
        upload_id: uploadId,
        prompt: $("prompt").value,
        seed: Number($("seed").value),
        max_chunks: Number($("maxChunks").value),
      }),
    });
    session = s.session_id;
    generation = s.generation;
    player.clear();
    state(s);
    $("seedFooter").textContent = $("seed").value;
    const connection = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${session}`,
    );
    ws = connection;
    ws.onopen = () => {
      if (ws !== connection) return;
      $("connection").textContent = "已连接";
      $("connectionDot").classList.add("online");
      send({ type: "speed", value: Number($("speed").value) });
      send({ type: "vertical_speed", value: Number($("verticalSpeed").value) });
      send({ type: "rotation_angle", value: Number($("rotationAngle").value) });
      send({ type: "rotation_mode", value: $("rotationMode").value });
      send({ type: "orbit_radius", value: Number($("orbitRadius").value) });
    };
    ws.onmessage = (e) => {
      if (ws !== connection) return;
      const m = JSON.parse(e.data);
      if (m.type === "error") {
        error(m.message);
        return;
      }
      if (m.generation < generation) return;
      if (m.generation > generation) {
        generation = m.generation;
        player.clear();
        clearKeys(false);
      }
      if (m.type === "status") state(m);
      if (m.type === "chunk_ready") player.add(m);
    };
    ws.onclose = () => {
      if (ws !== connection) return;
      clearKeys(false);
      $("connection").textContent = "连接断开 · 生成已停止";
      $("connectionDot").classList.remove("online");
      $("start").disabled = false;
      for (const id of ["preset", "upload", "prompt", "seed", "maxChunks"])
        $(id).disabled = false;
    };
  } catch (e) {
    error(e.message);
    $("start").disabled = false;
  }
});
for (const type of ["pause", "resume", "reset", "stop"])
  $(type).addEventListener("click", () => {
    if (type === "reset" || type === "stop") clearKeys();
    send({ type });
  });
$("rotationMode").addEventListener("change", () => {
  send({ type: "rotation_mode", value: $("rotationMode").value });
});
$("orbitRadius").addEventListener("input", () => {
  const value = Number($("orbitRadius").value);
  $("orbitRadiusLabel").textContent = value.toFixed(1);
  send({ type: "orbit_radius", value });
});
for (const [id, type, label, angle] of [
  ["speed", "speed", "speedLabel", false],
  ["verticalSpeed", "vertical_speed", "verticalSpeedLabel", false],
  ["rotationAngle", "rotation_angle", "rotationAngleLabel", true],
]) {
  $(id).addEventListener("input", () => {
    const value = Number($(id).value);
    $(label).textContent = angle
      ? `${value.toFixed(0)}° / 段`
      : `${value.toFixed(1)} 米 / 段`;
    send({ type, value });
  });
}
function clearKeys(notify = true) {
  held.clear();
  if (notify) send({ type: "blur" });
}
document.addEventListener("keydown", (e) => {
  if (
    ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) ||
    e.target.isContentEditable
  )
    return;
  if (
    !session ||
    ["stopped", "complete", "disconnected", "error"].includes(currentState)
  )
    return;
  if (e.key === "Escape") {
    clearKeys();
    return;
  }
  if (e.code === "Space") {
    e.preventDefault();
    if (!e.repeat)
      send({
        type: ["paused", "pausing"].includes(currentState) ? "resume" : "pause",
      });
    return;
  }
  const key = e.key.toLowerCase();
  if (!controlKeys.has(key)) return;
  e.preventDefault();
  // OS auto-repeat must never make an older held key override a newer key.
  if (!e.repeat && !held.has(key)) {
    held.add(key);
    send({ type: "key", key, down: true });
  }
});
document.addEventListener("keyup", (e) => {
  const key = e.key.toLowerCase();
  if (held.delete(key)) send({ type: "key", key, down: false });
});
window.addEventListener("blur", () => clearKeys());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearKeys();
});
$("viewport").addEventListener("click", () => {
  $("viewport").focus();
  if (session && player.waiting) player.advance();
});
window.addEventListener("beforeunload", () => ws?.close());
(async () => {
  try {
    const r = await api("/api/presets");
    allPresets = r.presets;
    $("presetCount").textContent = `01 — ${String(allPresets.length).padStart(2, "0")}`;
    for (const p of allPresets) {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = p.name;
      $("preset").append(o);
    }
    $("preset").value = r.default;
    changePreset();
    const h = await api("/api/health");
    $("connection").textContent = h.error
      ? "模型加载失败"
      : h.mock
        ? "模拟模式"
        : h.ready
          ? "模型已就绪"
          : "模型加载中";
    $("connectionDot").classList.add("online");
    if (h.error) error(h.error);
  } catch (e) {
    error(e.message);
  }
})();
