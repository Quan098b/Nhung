"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ==========================================
// TRANG THAI CHUNG
// ==========================================

const state = {
  mode: "image",
  conf: 0.5,
  iou: 0.45,
  imgsz: 640,
  info: null,
  results: { image: null, video: null, camera: null },
};

const PALETTE = [
  "#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4",
  "#ec4899", "#84cc16", "#f97316", "#14b8a6", "#a855f7", "#eab308",
];
const colorFor = (cls) => PALETTE[cls % PALETTE.length];

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

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
    const color = colorFor(d.cls);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${d.name} ${(d.conf * 100).toFixed(0)}%`;
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

function renderStats(r) {
  $("#statTotal").textContent = r ? r.total : 0;
  $("#statMs").textContent = r && r.ms ? Math.round(r.ms) : "—";
  $("#statFps").textContent = r && r.fps ? r.fps.toFixed(1) : "—";

  const list = $("#countList");
  const counts = r ? Object.entries(r.counts) : [];
  if (!counts.length) {
    list.innerHTML = `<li class="empty muted">${r ? "Không phát hiện linh kiện nào" : "Chưa có kết quả"}</li>`;
  } else {
    const nameToCls = {};
    for (const [k, v] of Object.entries(state.info?.classes || {})) nameToCls[v] = Number(k);
    const max = Math.max(...counts.map(([, n]) => n));
    counts.sort((a, b) => b[1] - a[1]);
    list.innerHTML = counts.map(([name, n]) => {
      const c = colorFor(nameToCls[name] ?? 0);
      return `<li>
        <div class="count-row"><span class="swatch" style="background:${c}"></span>
        <span class="count-name">${esc(name)}</span><span class="count-num">${n}</span></div>
        <div class="bar"><div style="width:${(n / max) * 100}%;background:${c}"></div></div></li>`;
    }).join("");
  }

  const showDet = state.mode === "image" && r && r.detections && r.detections.length;
  $("#detCard").hidden = !showDet;
  if (showDet) {
    $("#detList").innerHTML = r.detections
      .slice().sort((a, b) => b.conf - a.conf)
      .map((d, i) => `<li><span class="idx">${i + 1}</span><span class="swatch" style="background:${colorFor(d.cls)}"></span>
        <span>${esc(d.name)}</span><span class="conf">${(d.conf * 100).toFixed(1)}%</span></li>`)
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

function renderInfo(info) {
  state.info = info;
  state.conf = info.conf;
  state.iou = info.iou;
  state.imgsz = info.imgsz;

  $("#modelDot").className = "dot " + (info.loaded ? "ok" : "err");
  $("#modelPillText").textContent = info.loaded ? info.name : "Chưa nạp model";
  $("#devicePill").textContent = info.device;

  syncConf(info.conf);
  $("#iouSetting").value = info.iou;
  $("#iouVal").textContent = info.iou.toFixed(2);
  $("#imgszSetting").value = String(info.imgsz);

  const classes = Object.entries(info.classes);
  $("#modelCurrent").innerHTML = info.loaded
    ? `<div class="kv"><span>Tên file</span><span>${esc(info.name)}</span></div>
       <div class="kv"><span>Loại tác vụ</span><span>${esc(info.task || "detect")}</span></div>
       <div class="kv"><span>Số lớp</span><span>${classes.length}</span></div>
       <div class="kv"><span>Thiết bị</span><span>${esc(info.device)}</span></div>`
    : `<span class="muted">Chưa có model nào được nạp. Hãy tải lên hoặc chọn một file .pt bên dưới.</span>`;

  $("#classChips").innerHTML = classes
    .map(([id, name]) => `<span class="chip"><span class="swatch" style="background:${colorFor(Number(id))}"></span>${esc(name)}</span>`)
    .join("");

  $("#modelList").innerHTML = info.models.length
    ? info.models.map((m) => {
        const cur = m.path === info.path;
        return `<li class="${cur ? "current" : ""}">
          <div class="m-info"><div class="m-name">${esc(m.name)}</div><div class="m-meta">${esc(m.path)} · ${m.size_mb} MB</div></div>
          ${cur ? `<span class="tag">Đang dùng</span>` : `<button class="btn btn-sm" data-load="${esc(m.path)}">Nạp</button>`}
        </li>`;
      }).join("")
    : `<li class="muted">Không tìm thấy file .pt nào.</li>`;
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
  renderInfo(info);
  toast(`Đã nạp model ${info.name}`, "success");
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

function uploadModel(file) {
  if (!file || !file.name.toLowerCase().endsWith(".pt")) {
    toast("Hãy chọn file có đuôi .pt", "error");
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
    if (e.loaded === e.total) status.textContent = "Đang nạp model…";
  };
  xhr.onload = () => {
    $("#ptProgress").hidden = true;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status === 200) {
      status.textContent = "File sẽ được lưu vào thư mục models/ và nạp ngay";
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
  postJSON("/api/settings", { imgsz: state.imgsz }).then(() => {
    toast(`Kích thước ảnh đầu vào: ${state.imgsz}px`);
    redetectImage();
  });
});

// ==========================================
// CHUYEN CHE DO
// ==========================================

const MODES = {
  image: ["Nhận diện qua ảnh", "Tải ảnh lên để phát hiện và đếm linh kiện."],
  video: ["Nhận diện qua video", "Phát video và nhận diện linh kiện theo từng khung hình."],
  camera: ["Camera trực tiếp", "Nhận diện thời gian thực từ webcam hoặc camera điện thoại."],
  settings: ["Cài đặt model", "Nạp file .pt và điều chỉnh thông số nhận diện."],
};

function setMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === "camera") camera.stop();
  if (state.mode === "video") video.el.pause();

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

const image = { file: null, bitmap: null, busy: false, pending: false };

async function openImage(file) {
  if (!file || !file.type.startsWith("image/")) {
    toast("File không phải ảnh", "error");
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
  $("#imageDrop").hidden = true;
  const canvas = $("#imageCanvas");
  canvas.hidden = false;
  canvas.width = image.bitmap.width;
  canvas.height = image.bitmap.height;
  canvas.getContext("2d").drawImage(image.bitmap, 0, 0);
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
// ==========================================

class LiveDetector {
  constructor(source, canvas, mode) {
    this.source = source;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.mode = mode;
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
      // Video dang dung va da nhan dien khung nay roi -> cho
      if (v.readyState < 2 || (v.paused && v.currentTime === this.lastTime && this.mode === "video")) {
        await sleep(80);
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
        setResult(this.mode, r);
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

const video = {
  el: $("#videoEl"),
  url: null,
  detector: null,
};
video.detector = new LiveDetector(video.el, $("#videoCanvas"), "video");

function openVideo(file) {
  if (!file || !file.type.startsWith("video/")) {
    toast("File không phải video", "error");
    return;
  }
  if (video.url) URL.revokeObjectURL(video.url);
  video.url = URL.createObjectURL(file);
  video.el.src = video.url;
  video.el.playbackRate = Number($("#videoSpeed").value);
  $("#videoName").textContent = file.name;
  setResult("video", null);
}

video.el.addEventListener("loadedmetadata", () => {
  $("#videoDrop").hidden = true;
  $("#videoCanvas").hidden = false;
  $("#videoControls").hidden = false;
  $("#videoSnap").disabled = false;
  video.el.playbackRate = Number($("#videoSpeed").value);
  video.detector.start();
  video.el.play();
});
video.el.addEventListener("error", () => toast("Trình duyệt không phát được định dạng video này", "error"));
video.el.addEventListener("play", () => { $("#iconPlay").hidden = true; $("#iconPause").hidden = false; });
video.el.addEventListener("pause", () => { $("#iconPlay").hidden = false; $("#iconPause").hidden = true; });
video.el.addEventListener("timeupdate", () => {
  const d = video.el.duration || 0;
  $("#videoTime").textContent = `${fmtTime(video.el.currentTime)} / ${fmtTime(d)}`;
  if (d && !video.seeking) $("#videoSeek").value = (video.el.currentTime / d) * 1000;
});

$("#videoPlay").addEventListener("click", () => (video.el.paused ? video.el.play() : video.el.pause()));
$("#videoSeek").addEventListener("input", (e) => {
  video.seeking = true;
  if (video.el.duration) video.el.currentTime = (e.target.value / 1000) * video.el.duration;
});
$("#videoSeek").addEventListener("change", () => { video.seeking = false; });
$("#videoSpeed").addEventListener("change", (e) => { video.el.playbackRate = Number(e.target.value); });
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
      throw new Error("Trình duyệt không hỗ trợ camera. Hãy mở bằng http://127.0.0.1:5000");
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
    this.detector.start();
    if (!this.permission) {
      this.permission = true;
      this.listDevices(false); // lay ten camera sau khi da cap quyen
    }
  },

  async startIp() {
    const url = $("#ipcamUrl").value.trim();
    if (!/^(https?|rtsp):\/\//.test(url)) throw new Error("Địa chỉ phải bắt đầu bằng http://, https:// hoặc rtsp://");
    await this.stopping; // doi lenh dung luong cu xong, tranh dung nham luong moi
    await postJSON("/api/settings", { conf: state.conf, iou: state.iou });
    const img = $("#ipcamImg");
    const ip = { url, lastLoad: Date.now(), wasLive: false, warned: false, startedAt: Date.now() };
    this.ip = ip;

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
      ? "LIVE"
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
    this.setLinkStatus("live");
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

camera.select.addEventListener("change", () => {
  camera.onSelect();
  if (camera.running) camera.start();
});
$("#camStart").addEventListener("click", () => camera.start());
$("#camStop").addEventListener("click", () => camera.stop());
$("#camSnap").addEventListener("click", () => camera.snap());
$("#camRefresh").addEventListener("click", () => camera.listDevices(true));
navigator.mediaDevices?.addEventListener?.("devicechange", () => camera.listDevices(false));

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
