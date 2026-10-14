"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ==========================================
// TRANG THAI CHUNG
// ==========================================

const state = {
  mode: "image",
  conf: 0.35,
  iou: 0.45,
  imgsz: 640,
  info: null,
  targets: [], // 4 nhom linh kien (lay tu server: key, vi, color, supported, sources)
  results: { image: null, video: null, camera: null },
};

const targetOf = (key) => state.targets.find((t) => t.key === key);
const colorOf = (key) => targetOf(key)?.color || "#64748b";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (v) => `${(v * 100).toFixed(1)}%`;

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

function toast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), type === "error" ? 6000 : 3500);
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Lỗi máy chủ (${res.status})`);
  return data;
}

function postJSON(url, body) {
  return api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function downloadCanvas(canvas, name) {
  canvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/png");
}

const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");

// ==========================================
// GIAO DIEN SANG / TOI
// ==========================================

const THEMES = {
  auto: ["Tự động", '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/>'],
  light: ["Sáng", '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'],
  dark: ["Tối", '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'],
};

function applyTheme(theme) {
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $("#themeText").textContent = THEMES[theme][0];
  $("#themeIcon").innerHTML = THEMES[theme][1];
}

let theme = store.get("theme");
if (!THEMES[theme]) theme = "auto";
applyTheme(theme);
$("#themeBtn").addEventListener("click", () => {
  const order = ["auto", "light", "dark"];
  theme = order[(order.indexOf(theme) + 1) % order.length];
  if (theme === "auto") store.del("theme"); else store.set("theme", theme);
  applyTheme(theme);
});

// ==========================================
// VE BOUNDING BOX
// ==========================================

function drawBoxes(ctx, dets, w, h) {
  const lw = Math.max(2, Math.round(Math.max(w, h) / 400));
  const fs = Math.max(13, Math.round(Math.max(w, h) / 55));
  ctx.lineWidth = lw;
  ctx.font = `600 ${fs}px "Be Vietnam Pro", system-ui, sans-serif`;
  ctx.textBaseline = "top";

  for (const d of dets) {
    const [x1, y1, x2, y2] = [d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h];
    const color = colorOf(d.key);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${d.label} ${(d.conf * 100).toFixed(0)}%`;
    const pad = Math.round(fs * 0.35);
    const tw = ctx.measureText(label).width + pad * 2;
    const th = fs + pad * 2;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = color;
    ctx.fillRect(x1 - lw / 2, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 - lw / 2 + pad, ty + pad);
  }
}

// ==========================================
// HIEN THI KET QUA
// ==========================================

const SCOPE = {
  image: "Trong ảnh hiện tại",
  video: "Trong khung hình hiện tại · không cộng dồn giữa các khung",
  camera: "Trong khung hình hiện tại",
};

function renderStats(r) {
  $("#statScope").textContent = SCOPE[state.mode] || "";
  $("#statTotal").textContent = r ? r.total : 0;
  $("#statConf").textContent = r && r.avg_conf != null ? pct(r.avg_conf) : "—";
  $("#statMs").textContent = r && r.ms ? Math.round(r.ms) : "—";
  $("#statFps").textContent = r && r.fps ? r.fps.toFixed(1) : "—";

  const counts = (r && r.counts) || {};
  const max = Math.max(1, ...state.targets.map((t) => counts[t.key] || 0));
  $("#countList").innerHTML = state.targets.length
    ? state.targets.map((t) => {
        const n = counts[t.key] || 0;
        return `<li class="${t.supported ? "" : "unsupported"}">
          <div class="count-row"><span class="swatch" style="background:${t.color}"></span>
            <span class="count-name">${esc(t.vi)} <span class="count-sub">(${esc(t.key)})</span></span>
            <span class="count-num">${t.supported ? n : "—"}</span></div>
          ${t.supported
            ? `<div class="bar"><div style="width:${(n / max) * 100}%;background:${t.color}"></div></div>`
            : `<div class="count-sub">Model hiện tại không có lớp này</div>`}
        </li>`;
      }).join("")
    : `<li class="empty muted">Chưa nạp model</li>`;

  const showDet = state.mode === "image" && r && r.detections && r.detections.length;
  $("#detCard").hidden = !showDet;
  if (showDet) {
    $("#detList").innerHTML = r.detections
      .map((d, i) => `<tr><td class="muted">${i + 1}</td>
        <td><span class="cell-type"><span class="swatch" style="background:${colorOf(d.key)}"></span>${esc(d.label)}</span></td>
        <td class="muted">${esc(d.orig)}</td><td class="num">${pct(d.conf)}</td></tr>`)
      .join("");
  }
}

function setResult(mode, r) {
  state.results[mode] = r;
  if (state.mode === mode) renderStats(r);
}

// ==========================================
// MODEL & CAI DAT
// ==========================================

const ALERT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.5"/></svg>';

function renderAlert(info) {
  const box = $("#modelAlert");
  const vi = (k) => targetOf(k)?.vi || k;
  let html = "";
  let kind = "info";

  if (!info.loaded) {
    kind = "warn";
    html = `<strong>Chưa nạp model.</strong><p>Vào <b>Cài đặt model</b> để chọn hoặc tải lên file .pt.</p>`;
  } else if (info.missing.length) {
    kind = "warn";
    html = `<strong>Model “${esc(info.name)}” không hỗ trợ đủ 4 nhóm linh kiện.</strong>
      <p>Thiếu: ${info.missing.map((k) => `${esc(vi(k))} (${esc(k)})`).join(", ")}. Các nhóm này sẽ luôn hiển thị “—” và không được nhận diện.</p>`;
  } else if (!info.trained_on_targets) {
    const merged = Object.entries(info.merged);
    html = `<strong>Model có ${info.nc} lớp, đang được lọc còn 4 nhóm linh kiện khi suy luận.</strong>
      <p>Bỏ qua ${info.ignored.length} lớp: ${info.ignored.map(esc).join(", ")}.
      ${merged.length ? `Gộp: ${merged.map(([o, k]) => `${esc(o)} → ${esc(vi(k))}`).join(", ")}.` : ""}
      Lưu ý: lọc khi suy luận không biến model thành model được huấn luyện với đúng 4 lớp.</p>`;
  }

  const dismissKey = `alertHidden:${info.path || ""}`;
  let dismissed = false;
  try { dismissed = !!sessionStorage.getItem(dismissKey); } catch {}
  if (!html || (kind === "info" && dismissed)) {
    box.hidden = true;
    return;
  }
  box.className = `alert ${kind}`;
  box.innerHTML = `${ALERT_ICON}<div style="flex:1">${html}</div>${kind === "info" ? '<button class="alert-close" aria-label="Ẩn">×</button>' : ""}`;
  box.hidden = false;
  box.querySelector(".alert-close")?.addEventListener("click", () => {
    try { sessionStorage.setItem(dismissKey, "1"); } catch {}
    box.hidden = true;
  });
}

function renderInfo(info) {
  state.info = info;
  state.targets = info.targets;
  state.conf = info.conf;
  state.iou = info.iou;
  state.imgsz = info.imgsz;

  $("#modelDot").className = "dot " + (info.loaded ? (info.missing.length ? "warn" : "ok") : "err");
  $("#modelPillText").textContent = info.loaded ? `${info.name} · ${info.arch}` : "Chưa nạp model";
  $("#devicePill").textContent = info.device;

  syncConf(info.conf);
  $("#iouSetting").value = info.iou;
  $("#iouVal").textContent = info.iou.toFixed(2);
  $("#imgszSetting").value = String(info.imgsz);

  renderAlert(info);

  const m = info.metrics;
  const kv = (k, v) => (v == null || v === "" ? "" : `<div class="kv"><span>${k}</span><span>${esc(v)}</span></div>`);
  $("#modelCurrent").innerHTML = info.loaded
    ? kv("Tên file", info.name) + kv("Kiến trúc", info.arch) + kv("Tác vụ", info.task)
      + kv("Số lớp gốc", info.nc) + kv("Tham số", `${info.params_m} triệu`) + kv("Thiết bị suy luận", info.device)
      + kv("Ngày huấn luyện", info.train_date) + kv("Số epoch", info.train_epochs) + kv("imgsz lúc huấn luyện", info.train_imgsz)
      + (m ? kv("mAP50", m.map50 != null ? pct(m.map50) : null) + kv("mAP50-95", m.map50_95 != null ? pct(m.map50_95) : null)
        + kv("Precision", m.precision != null ? pct(m.precision) : null) + kv("Recall", m.recall != null ? pct(m.recall) : null) : "")
    : `<span class="muted">Chưa có model nào được nạp. Hãy tải lên hoặc chọn một file .pt bên dưới.</span>`;
  $("#metricsNote").hidden = !(info.loaded && m);

  // Bang anh xa lop
  const tbody = $("#mapTable tbody");
  if (info.loaded) {
    const groupOf = {};
    info.targets.forEach((t) => t.sources.forEach((s) => { groupOf[s] = t; }));
    tbody.innerHTML = Object.entries(info.names).map(([id, name]) => {
      const t = groupOf[name];
      return `<tr><td class="muted">${id}</td><td>${esc(name)}</td><td>${t
        ? `<span class="cell-type"><span class="swatch" style="background:${t.color}"></span>${esc(t.vi)}</span>`
        : `<span class="ignored">Bỏ qua</span>`}</td></tr>`;
    }).join("");
    const kept = info.nc - info.ignored.length;
    $("#mapSummary").textContent = info.trained_on_targets
      ? `Model được huấn luyện đúng ${info.nc} lớp mục tiêu`
      : `Giữ ${kept}/${info.nc} lớp${info.missing.length ? ` · thiếu ${info.missing.join(", ")}` : ""}`;
  } else {
    tbody.innerHTML = `<tr><td colspan="3" class="muted">Chưa nạp model</td></tr>`;
    $("#mapSummary").textContent = "";
  }

  // The tom tat o cot phai
  const row = (k, v) => `<div class="row"><span>${k}</span><span>${esc(v)}</span></div>`;
  $("#modelMini").innerHTML = info.loaded
    ? row("File", info.name) + row("Kiến trúc", info.arch) + row("Thiết bị", info.device)
      + row("Lớp", info.trained_on_targets ? `${info.nc} lớp (đúng 4 nhóm)` : `${info.nc} lớp → lọc ${4 - info.missing.length}/4 nhóm`)
      + (m && m.map50 != null ? row("mAP50 (checkpoint)", pct(m.map50)) : "")
    : `<span class="muted">Chưa nạp model</span>`;

  $("#modelList").innerHTML = info.models.length
    ? info.models.map((md) => {
        const cur = md.path === info.path;
        return `<li class="${cur ? "current" : ""}">
          <div class="m-info"><div class="m-name">${esc(md.name)}</div><div class="m-meta">${esc(md.path)} · ${md.size_mb} MB</div></div>
          ${cur ? `<span class="tag">Đang dùng</span>` : `<button class="btn btn-sm" data-load="${esc(md.path)}">Nạp</button>`}
        </li>`;
      }).join("")
    : `<li class="muted">Không tìm thấy file .pt nào.</li>`;

  if (state.mode !== "settings") renderStats(state.results[state.mode]);
}

async function loadStatus() {
  try {
    renderInfo(await api("/api/status"));
  } catch (e) {
    $("#modelDot").className = "dot err";
    $("#modelPillText").textContent = "Mất kết nối máy chủ";
    toast("Không kết nối được máy chủ Flask. Hãy chạy: python app.py", "error");
  }
}

function onModelChanged(info) {
  // Ket qua cu thuoc model cu -> xoa
  state.results = { image: null, video: null, camera: null };
  renderInfo(info);
  toast(`Đã nạp model ${info.name} (${info.arch}, ${info.device})`, "success");
  if (image.file) detectImage();
}

$("#modelList").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-load]");
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = "Đang nạp…";
  try {
    onModelChanged(await postJSON("/api/model/load", { path: btn.dataset.load }));
  } catch (err) {
    toast(err.message, "error");
    btn.disabled = false;
    btn.textContent = "Nạp";
  }
});

const PT_HINT = "File được lưu vào thư mục models/ và nạp ngay · tối đa 300 MB";

function uploadModel(file) {
  if (!file || !file.name.toLowerCase().endsWith(".pt")) {
    toast("Hãy chọn file có đuôi .pt", "error");
    return;
  }
  if (file.size > 300 * 1024 * 1024) {
    toast("File .pt vượt quá 300 MB", "error");
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  const xhr = new XMLHttpRequest();
  const status = $("#ptStatus");
  $("#ptProgress").hidden = false;
  $("#ptBar").style.width = "0";
  status.textContent = `Đang tải lên ${file.name}…`;

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("#ptBar").style.width = `${(e.loaded / e.total) * 100}%`;
    if (e.loaded === e.total) status.textContent = "Đang kiểm tra và nạp model…";
  };
  xhr.onload = () => {
    $("#ptProgress").hidden = true;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status === 200) {
      status.textContent = PT_HINT;
      onModelChanged(data);
    } else {
      status.textContent = data.error || "Tải lên thất bại";
      toast(status.textContent, "error");
    }
  };
  xhr.onerror = () => {
    $("#ptProgress").hidden = true;
    status.textContent = "Tải lên thất bại";
    toast("Tải lên thất bại", "error");
  };
  xhr.open("POST", "/api/model/upload");
  xhr.send(fd);
}

$("#ptInput").addEventListener("change", (e) => { uploadModel(e.target.files[0]); e.target.value = ""; });
setupDrop($("#ptDrop"), uploadModel);

// ---------- Nguong tin cay (2 thanh truot dong bo) ----------

function syncConf(v) {
  $$(".conf-input").forEach((el) => { el.value = v; });
  $$('[data-bind="conf"]').forEach((el) => { el.textContent = Number(v).toFixed(2); });
}

const saveSettings = debounce((body) => postJSON("/api/settings", body).catch(() => {}), 200);
const redetectImage = debounce(() => { if (image.file) detectImage(); }, 250);

$$(".conf-input").forEach((el) => el.addEventListener("input", () => {
  state.conf = Number(el.value);
  syncConf(state.conf);
  saveSettings({ conf: state.conf });
  redetectImage();
}));

$("#iouSetting").addEventListener("input", (e) => {
  state.iou = Number(e.target.value);
  $("#iouVal").textContent = state.iou.toFixed(2);
  saveSettings({ iou: state.iou });
  redetectImage();
});

$("#imgszSetting").addEventListener("change", (e) => {
  state.imgsz = Number(e.target.value);
  postJSON("/api/settings", { imgsz: state.imgsz })
    .then(() => { toast(`Kích thước ảnh đầu vào: ${state.imgsz}px`); redetectImage(); })
    .catch((err) => toast(err.message, "error"));
});

$("#resetSettings").addEventListener("click", async () => {
  try {
    const info = await postJSON("/api/settings", state.info.defaults);
    renderInfo(info);
    redetectImage();
    toast("Đã khôi phục thông số mặc định");
  } catch (err) {
    toast(err.message, "error");
  }
});

// ==========================================
// CHUYEN CHE DO
// ==========================================

const MODES = {
  image: ["Nhận diện qua ảnh", "Tải ảnh lên để phát hiện và đếm điện trở, tụ điện, LED, vi mạch."],
  video: ["Nhận diện qua video", "Phát video và nhận diện linh kiện theo từng khung hình."],
  camera: ["Camera trực tiếp", "Nhận diện thời gian thực từ webcam, camera USB hoặc camera điện thoại."],
  settings: ["Cài đặt model", "Nạp file .pt, xem ánh xạ lớp và điều chỉnh thông số nhận diện."],
};

function setMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === "camera") camera.stop();
  if (state.mode === "video") video.pause();

  state.mode = mode;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $$("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== mode; });
  $("#pageTitle").textContent = MODES[mode][0];
  $("#pageSub").textContent = MODES[mode][1];

  const side = mode !== "settings";
  $("#sideCol").hidden = !side;
  $("#workspace").classList.toggle("no-side", !side);
  if (side) renderStats(state.results[mode]);
  if (mode === "camera") camera.listDevices(false);
}

$$(".nav-item").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("#modelPill").addEventListener("click", () => setMode("settings"));

function setupDrop(zone, onFile) {
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

// ==========================================
// 1. NHAN DIEN QUA ANH
// ==========================================

const IMAGE_EXT = /\.(jpe?g|png|webp|bmp)$/i;
const image = { file: null, bitmap: null, busy: false, pending: false };

async function openImage(file) {
  if (!file || !(IMAGE_EXT.test(file.name) || /^image\/(jpeg|png|webp|bmp)$/.test(file.type))) {
    toast("Chỉ hỗ trợ ảnh JPG, JPEG, PNG, WEBP, BMP", "error");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    toast("Ảnh vượt quá 25 MB", "error");
    return;
  }
  try {
    image.bitmap = await createImageBitmap(file);
  } catch {
    toast("Không đọc được ảnh này", "error");
    return;
  }
  image.file = file;
  $("#imageName").textContent = file.name;
  $("#imageName").title = file.name;
  $("#imageDrop").hidden = true;
  const canvas = $("#imageCanvas");
  canvas.hidden = false;
  canvas.width = image.bitmap.width;
  canvas.height = image.bitmap.height;
  canvas.getContext("2d").drawImage(image.bitmap, 0, 0);
  setResult("image", null);
  detectImage();
}

async function detectImage() {
  if (image.busy) { image.pending = true; return; }
  image.busy = true;
  $("#imageBusy").hidden = false;
  try {
    const fd = new FormData();
    fd.append("image", image.file);
    fd.append("conf", state.conf);
    fd.append("iou", state.iou);
    const r = await api("/api/detect", { method: "POST", body: fd });
    const canvas = $("#imageCanvas");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image.bitmap, 0, 0);
    drawBoxes(ctx, r.detections, canvas.width, canvas.height);
    $("#imageDownload").disabled = false;
    setResult("image", r);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    image.busy = false;
    $("#imageBusy").hidden = true;
    if (image.pending) { image.pending = false; detectImage(); }
  }
}

$("#imageInput").addEventListener("change", (e) => { openImage(e.target.files[0]); e.target.value = ""; });
$("#imageDrop").addEventListener("click", () => $("#imageInput").click());
setupDrop($("#imageStage"), openImage);
$("#imageDownload").addEventListener("click", () => {
  const base = (image.file?.name || "anh").replace(/\.[^.]+$/, "");
  downloadCanvas($("#imageCanvas"), `ketqua_${base}.png`);
});

// ==========================================
// NHAN DIEN LIEN TUC (dung chung cho video & webcam)
// Moi luc chi co 1 request dang cho -> khong suy luan chong cheo, khong don frame cu.
// ==========================================

class LiveDetector {
  constructor(source, canvas, mode, onDetected) {
    this.source = source;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.mode = mode;
    this.onDetected = onDetected;
    this.grab = document.createElement("canvas");
    this.dets = [];
    this.run = 0;
  }

  start() {
    const id = ++this.run;
    this.dets = [];
    this.lastTime = -1;
    this.lastAt = performance.now();
    this.errorShown = false;
    const draw = () => {
      if (id !== this.run) return;
      this.paint();
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
    this.loop(id);
  }

  stop() { this.run++; }

  paint() {
    const v = this.source;
    if (!v.videoWidth) return;
    if (this.canvas.width !== v.videoWidth || this.canvas.height !== v.videoHeight) {
      this.canvas.width = v.videoWidth;
      this.canvas.height = v.videoHeight;
    }
    this.ctx.drawImage(v, 0, 0);
    drawBoxes(this.ctx, this.dets, this.canvas.width, this.canvas.height);
  }

  async loop(id) {
    while (id === this.run) {
      const v = this.source;
      // Chua san sang, dang tua, hoac video dang dung va khung nay da nhan dien -> cho
      if (v.readyState < 2 || v.seeking || (this.mode === "video" && v.paused && v.currentTime === this.lastTime)) {
        await sleep(30);
        continue;
      }
      const t = v.currentTime;
      const blob = await this.capture();
      try {
        const fd = new FormData();
        fd.append("image", blob, "frame.jpg");
        fd.append("conf", state.conf);
        fd.append("iou", state.iou);
        const r = await api("/api/detect", { method: "POST", body: fd });
        if (id !== this.run) return;
        const now = performance.now();
        r.fps = 1000 / (now - this.lastAt);
        this.lastAt = now;
        this.lastTime = t;
        this.dets = r.detections;
        this.errorShown = false;
        setResult(this.mode, r);
        this.onDetected?.();
      } catch (e) {
        if (!this.errorShown) toast(e.message, "error");
        this.errorShown = true;
        await sleep(1000);
      }
    }
  }

  capture() {
    const v = this.source;
    const scale = Math.min(1, Math.max(state.imgsz, 640) / Math.max(v.videoWidth, v.videoHeight));
    this.grab.width = Math.round(v.videoWidth * scale);
    this.grab.height = Math.round(v.videoHeight * scale);
    this.grab.getContext("2d").drawImage(v, 0, 0, this.grab.width, this.grab.height);
    return new Promise((r) => this.grab.toBlob(r, "image/jpeg", 0.85));
  }
}

// ==========================================
// 2. NHAN DIEN QUA VIDEO
// ==========================================

const fmtTime = (s) => {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

const STEP_SECONDS = 1 / 30; // buoc nhay khi xu ly tung khung hinh

const video = {
  el: $("#videoEl"),
  url: null,
  stepping: false, // dang chay che do "tung khung hinh"
  detector: null,

  stepMode() { return $("#videoProcMode").value === "step"; },

  setIcon(playing) {
    $("#iconPlay").hidden = playing;
    $("#iconPause").hidden = !playing;
  },

  play() {
    if (this.stepMode()) {
      this.stepping = true;
      this.setIcon(true);
      if (this.el.ended || this.el.currentTime >= this.el.duration - STEP_SECONDS) this.el.currentTime = 0;
      else if (this.detector.lastTime === this.el.currentTime) this.next(); // khung nay da xong
    } else {
      this.el.play();
    }
  },

  pause() {
    this.stepping = false;
    this.el.pause();
    this.setIcon(false);
  },

  // Goi sau moi lan nhan dien xong: o che do tung khung thi nhay sang khung tiep theo
  next() {
    if (!this.stepping) return;
    const t = this.el.currentTime + STEP_SECONDS;
    if (t >= this.el.duration) {
      this.pause();
      toast("Đã xử lý hết video");
      return;
    }
    this.el.currentTime = t;
  },
};
video.detector = new LiveDetector(video.el, $("#videoCanvas"), "video", () => video.next());

function openVideo(file) {
  if (!file || !file.type.startsWith("video/")) {
    toast("File không phải video", "error");
    return;
  }
  video.pause();
  if (video.url) URL.revokeObjectURL(video.url);
  video.url = URL.createObjectURL(file);
  video.el.src = video.url;
  $("#videoName").textContent = file.name;
  $("#videoName").title = file.name;
  setResult("video", null);
}

video.el.addEventListener("loadedmetadata", () => {
  $("#videoDrop").hidden = true;
  $("#videoCanvas").hidden = false;
  $("#videoControls").hidden = false;
  $("#videoSnap").disabled = false;
  video.el.playbackRate = Number($("#videoSpeed").value);
  video.detector.start();
  video.play();
});
video.el.addEventListener("error", () => {
  if (video.el.getAttribute("src")) toast("Trình duyệt không phát được định dạng video này", "error");
});
video.el.addEventListener("play", () => video.setIcon(true));
video.el.addEventListener("pause", () => { if (!video.stepping) video.setIcon(false); });
video.el.addEventListener("timeupdate", () => {
  const d = video.el.duration || 0;
  $("#videoTime").textContent = `${fmtTime(video.el.currentTime)} / ${fmtTime(d)}`;
  if (d && !video.seeking) $("#videoSeek").value = (video.el.currentTime / d) * 1000;
});

$("#videoPlay").addEventListener("click", () => {
  const playing = video.stepping || !video.el.paused;
  if (playing) video.pause(); else video.play();
});
$("#videoSeek").addEventListener("input", (e) => {
  video.seeking = true;
  if (video.el.duration) video.el.currentTime = (e.target.value / 1000) * video.el.duration;
});
$("#videoSeek").addEventListener("change", () => { video.seeking = false; });
$("#videoSpeed").addEventListener("change", (e) => { video.el.playbackRate = Number(e.target.value); });
$("#videoProcMode").addEventListener("change", () => {
  video.pause();
  $("#videoSpeed").disabled = video.stepMode();
  toast(video.stepMode()
    ? "Chế độ từng khung hình: nhận diện xong khung này mới chuyển sang khung kế tiếp (không bỏ khung)."
    : "Chế độ thời gian thực: video phát đúng tốc độ, bỏ qua khung khi nhận diện không kịp.");
});
$("#videoInput").addEventListener("change", (e) => { openVideo(e.target.files[0]); e.target.value = ""; });
$("#videoDrop").addEventListener("click", () => $("#videoInput").click());
setupDrop($("#videoStage"), openVideo);
$("#videoSnap").addEventListener("click", () => downloadCanvas($("#videoCanvas"), `video_${stamp()}.png`));

// ==========================================
// 3. CAMERA TRUC TIEP
// ==========================================

const IPCAM = "__ipcam__";

const camera = {
  select: $("#cameraSelect"),
  videoEl: $("#camVideo"),
  stream: null,
  poll: null,
  running: false,
  permission: false,
  detector: null,
  ip: null,

  async listDevices(askPermission) {
    const prev = this.select.value;
    const options = [];
    if (navigator.mediaDevices?.enumerateDevices) {
      try {
        if (askPermission && !this.permission) {
          const s = await navigator.mediaDevices.getUserMedia({ video: true });
          s.getTracks().forEach((t) => t.stop());
          this.permission = true;
        }
        const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
        devices.forEach((d, i) => {
          if (d.label) this.permission = true;
          options.push([d.deviceId, d.label || `Camera ${i + 1}`]);
        });
      } catch (e) {
        toast("Trình duyệt chưa được cấp quyền dùng camera", "error");
      }
    }
    options.push([IPCAM, "📱 Camera điện thoại (IP Webcam)"]);
    this.select.innerHTML = options.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
    if ([...this.select.options].some((o) => o.value === prev)) this.select.value = prev;
    this.onSelect();
  },

  onSelect() {
    $("#ipcamUrl").hidden = this.select.value !== IPCAM;
  },

  async start() {
    this.stop();
    const id = this.select.value;
    try {
      if (id === IPCAM) await this.startIp();
      else await this.startLocal(id);
    } catch (e) {
      toast(e.message || "Không mở được camera", "error");
      this.stop();
      return;
    }
    this.running = true;
    $("#camIdle").hidden = true;
    $("#camStart").hidden = true;
    $("#camStop").hidden = false;
    $("#camSnap").disabled = false;
    $("#liveBadge").hidden = false;
    $("#navLive").hidden = false;
  },

  async startLocal(deviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Trình duyệt không cho dùng camera. Hãy mở bằng http://127.0.0.1:5000");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    this.videoEl.srcObject = this.stream;
    await this.videoEl.play();
    $("#camCanvas").hidden = false;
    this.setLinkStatus("live");
    this.detector.start();
    if (!this.permission) {
      this.permission = true;
      this.listDevices(false); // lay ten camera sau khi da cap quyen
    }
  },

  async startIp() {
    const url = $("#ipcamUrl").value.trim();
    if (!url) throw new Error("Hãy nhập địa chỉ camera điện thoại, ví dụ http://192.168.1.10:8080/video");
    if (!/^(https?|rtsp):\/\//.test(url)) throw new Error("Địa chỉ phải bắt đầu bằng http://, https:// hoặc rtsp://");
    store.set("ipcamUrl", url);
    await this.stopping; // doi lenh dung luong cu xong, tranh dung nham luong moi
    await postJSON("/api/settings", { conf: state.conf, iou: state.iou });
    const img = $("#ipcamImg");
    const ip = { url, lastLoad: Date.now(), wasLive: false, warned: false, startedAt: Date.now() };
    this.ip = ip;
    this.setLinkStatus("connecting");

    // Mo (lai) luong MJPEG tu server
    const connect = () => {
      if (this.ip !== ip) return;
      ip.lastLoad = Date.now();
      img.src = `/api/ipcam/stream?url=${encodeURIComponent(url)}&t=${Date.now()}`;
    };
    // Luong trinh duyet <-> server bi dut -> tu noi lai sau 1s
    img.onerror = () => { if (this.ip === ip) setTimeout(connect, 1000); };
    img.onload = () => { ip.lastLoad = Date.now(); };
    connect();
    img.hidden = false;

    this.poll = setInterval(async () => {
      if (this.ip !== ip) return;
      let s;
      try {
        s = await api("/api/ipcam/stats");
      } catch {
        this.setLinkStatus("server");
        return;
      }
      if (!s.running) {
        // Server khong con doc camera (vd: luong bi dong) -> mo lai
        if (Date.now() - ip.lastLoad > 3000) connect();
        this.setLinkStatus("connecting");
        return;
      }
      this.setLinkStatus(s.status, s.reconnects);
      if (s.status === "live") {
        if (!ip.wasLive && s.reconnects > 0) toast("Đã kết nối lại camera điện thoại", "success");
        ip.wasLive = true;
        ip.warned = false;
        setResult("camera", s);
      } else {
        if (ip.wasLive) toast("Mất tín hiệu camera điện thoại, đang tự kết nối lại…", "error");
        else if (!ip.warned && Date.now() - ip.startedAt > 10000) {
          toast("Chưa kết nối được camera điện thoại. Kiểm tra IP, Wi-Fi và app IP Webcam (vẫn đang thử lại).", "error");
          ip.warned = true;
        }
        ip.wasLive = false;
      }
    }, 700);
  },

  setLinkStatus(status, reconnects = 0) {
    const badge = $("#liveBadge");
    const live = status === "live";
    badge.classList.toggle("warn", !live);
    badge.lastChild.textContent = live
      ? "TRỰC TIẾP"
      : status === "server"
        ? "MẤT KẾT NỐI SERVER"
        : status === "reconnecting"
          ? `ĐANG KẾT NỐI LẠI${reconnects ? ` (${reconnects})` : ""}`
          : "ĐANG KẾT NỐI…";
  },

  stop() {
    this.detector.stop();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.videoEl.srcObject = null;
    clearInterval(this.poll);
    const img = $("#ipcamImg");
    if (this.ip) this.stopping = postJSON("/api/ipcam/stop", {}).catch(() => {});
    this.ip = null;
    img.onerror = img.onload = null;
    img.removeAttribute("src");
    img.hidden = true;
    $("#camCanvas").hidden = true;
    $("#camIdle").hidden = false;
    $("#camStart").hidden = false;
    $("#camStop").hidden = true;
    $("#camSnap").disabled = true;
    $("#liveBadge").hidden = true;
    $("#navLive").hidden = true;
    this.running = false;
  },

  snap() {
    if (this.select.value === IPCAM && !$("#ipcamImg").hidden) {
      const img = $("#ipcamImg");
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      c.getContext("2d").drawImage(img, 0, 0);
      downloadCanvas(c, `camera_${stamp()}.png`);
    } else {
      downloadCanvas($("#camCanvas"), `camera_${stamp()}.png`);
    }
  },
};
camera.detector = new LiveDetector(camera.videoEl, $("#camCanvas"), "camera");

$("#ipcamUrl").value = store.get("ipcamUrl") || "";
camera.select.addEventListener("change", () => {
  camera.onSelect();
  if (camera.running) camera.start();
});
$("#camStart").addEventListener("click", () => camera.start());
$("#camStop").addEventListener("click", () => camera.stop());
$("#camSnap").addEventListener("click", () => camera.snap());
$("#camRefresh").addEventListener("click", () => camera.listDevices(true));
navigator.mediaDevices?.addEventListener?.("devicechange", () => camera.listDevices(false));

// Dong trang -> giai phong camera IP tren server
window.addEventListener("pagehide", () => {
  if (camera.ip) navigator.sendBeacon("/api/ipcam/stop");
});

// ==========================================
// TAC GIA
// ==========================================

const modal = $("#authorModal");
$("#authorBtn").addEventListener("click", () => { modal.hidden = false; });
$("#authorClose").addEventListener("click", () => { modal.hidden = true; });
modal.addEventListener("click", (e) => { if (e.target === modal) modal.hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.hidden = true; });

// ==========================================
// KHOI DONG
// ==========================================

loadStatus();
