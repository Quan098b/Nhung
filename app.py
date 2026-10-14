"""
Web nhan dien va dem 4 nhom linh kien dien tu (Dien tro, Tu dien, LED, IC) bang YOLO.
Chay:  python app.py   roi mo  http://127.0.0.1:5000
"""

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ==========================================
# CAU HINH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model nap khi khoi dong: lay file dau tien ton tai trong danh sach.
# Uu tien YOLO11n (nhe, FPS cao), sau do YOLO11s 10 lop.
DEFAULT_MODELS = [
    MODELS_DIR / "best_7.pt",
    BASE_DIR / "best (5).pt",
]

# 4 nhom linh kien can nhan dien (thu tu = thu tu hien thi)
TARGET_CLASSES = {
    "Resistor": {"vi": "Điện trở", "color": "#3b82f6"},
    "Capacitor": {"vi": "Tụ điện", "color": "#f59e0b"},
    "LED": {"vi": "Đèn LED", "color": "#ef4444"},
    "IC": {"vi": "Vi mạch", "color": "#10b981"},
}

# Ten lop goc trong model (da chuan hoa: chu thuong, bo ky tu dac biet) -> nhom
CLASS_ALIASES = {
    "resistor": "Resistor", "resistors": "Resistor",
    "capacitor": "Capacitor", "capacitors": "Capacitor",
    "led": "LED", "leds": "LED",
    "ic": "IC", "ics": "IC", "integratedcircuit": "IC",
}

# Gop cac bien the tu dien vao nhom Tu dien (chi gop khi hien thi/dem,
# KHONG sua ten lop trong model). Dat False neu bao cao yeu cau tach rieng.
MERGE_CAPACITOR_VARIANTS = True
CAPACITOR_VARIANTS = {"ceramiccapacitor": "Capacitor", "electrolyticcapacitor": "Capacitor"}

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
ALLOWED_IMGSZ = {320, 416, 480, 640, 800, 960, 1280}

MAX_MODEL_MB = 300
MAX_IMAGE_MB = 25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Thong tin tac gia hien thi tren web - sua tai day
AUTHOR = {
    "name": "Tên tác giả",
    "role": "Sinh viên · Thiết kế & phát triển",
    "project": "Đồ án nhận diện 4 loại linh kiện điện tử bằng YOLO11",
    "email": "",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (MAX_MODEL_MB + 5) * 1024 * 1024


# ==========================================
# THIET BI (GPU / CPU)
# ==========================================


def pick_device():
    """Chi chon CUDA khi PyTorch thuc su chay duoc tren GPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return 0
        except Exception as e:
            print(f"[DEVICE] CUDA bao co nhung khong dung duoc: {e}")
    return "cpu"


DEVICE = pick_device()
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == 0 else None
GPU_SHORT = re.sub(r"^NVIDIA (GeForce )?|\s*GPU$", "", GPU_NAME) if GPU_NAME else None


# ==========================================
# QUAN LY MODEL
# ==========================================


def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass
class ModelBundle:
    """Model + bang anh xa lop. Duoc thay the nguyen khoi khi doi model."""

    model: YOLO
    path: Path
    class_map: dict  # class id goc -> nhom (key cua TARGET_CLASSES)
    class_ids: list  # cac class id goc duoc giu lai
    ignored: list  # ten lop goc bi bo qua
    merged: dict  # ten lop goc -> nhom (cac lop duoc gop)
    device: object = "cpu"
    device_label: str = "CPU"
    meta: dict = field(default_factory=dict)


state = {
    "bundle": None,
    "conf": DEFAULT_CONF,
    "iou": DEFAULT_IOU,
    "imgsz": DEFAULT_IMGSZ,
}
model_lock = threading.Lock()  # khoa suy luan va doi model


def build_class_map(names):
    aliases = dict(CLASS_ALIASES)
    if MERGE_CAPACITOR_VARIANTS:
        aliases.update(CAPACITOR_VARIANTS)

    class_map, ignored, merged = {}, [], {}
    for cid, cname in names.items():
        key = aliases.get(norm_name(cname))
        if key is None:
            ignored.append(cname)
            continue
        class_map[int(cid)] = key
        if norm_name(cname) != norm_name(key):
            merged[cname] = key
    return class_map, ignored, merged


def read_meta(model, path):
    ckpt = getattr(model, "ckpt", None) or {}
    args = ckpt.get("train_args") or {}
    metrics = ckpt.get("train_metrics") or {}

    arch_src = str(args.get("model") or "")
    if not arch_src:
        yaml = getattr(model.model, "yaml", {}) or {}
        arch_src = str(yaml.get("yaml_file") or "")
    arch = re.sub(r"^yolo", "YOLO", Path(arch_src).stem, flags=re.I) if arch_src else "Không rõ"

    def metric(k):
        v = metrics.get(k)
        return round(float(v), 4) if v is not None else None

    return {
        "arch": arch,
        "params_m": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
        "train_date": str(ckpt.get("date") or "")[:10] or None,
        "train_epochs": args.get("epochs"),
        "train_imgsz": args.get("imgsz"),
        "ultralytics": ckpt.get("version"),
        "metrics": {
            "precision": metric("metrics/precision(B)"),
            "recall": metric("metrics/recall(B)"),
            "map50": metric("metrics/mAP50(B)"),
            "map50_95": metric("metrics/mAP50-95(B)"),
        } if metrics else None,
    }


def device_label(device):
    return f"GPU · {GPU_SHORT}" if device == 0 else "CPU"


def load_model(path: Path):
    """Nap model, kiem tra, khoi dong (warmup) roi moi thay the model dang dung."""
    with open(path, "rb") as f:
        if f.read(4) != b"PK\x03\x04":
            raise ValueError("File không đúng định dạng trọng số PyTorch/Ultralytics")

    model = YOLO(str(path))
    if model.task not in ("detect", "segment"):
        raise ValueError(f"Chỉ hỗ trợ model phát hiện đối tượng (task hiện tại: {model.task})")

    class_map, ignored, merged = build_class_map(model.names)

    # Warmup: suy luan thu de biet thiet bi thuc te va tranh frame dau bi cham
    device = DEVICE
    dummy = np.zeros((DEFAULT_IMGSZ, DEFAULT_IMGSZ, 3), np.uint8)
    try:
        model.predict(dummy, device=device, imgsz=DEFAULT_IMGSZ, verbose=False)
    except Exception as e:
        if device == "cpu":
            raise
        print(f"[MODEL] Loi khi chay GPU, chuyen sang CPU: {e}")
        device = "cpu"
        model.predict(dummy, device=device, imgsz=DEFAULT_IMGSZ, verbose=False)

    actual = next(model.model.parameters()).device
    bundle = ModelBundle(
        model=model,
        path=path,
        class_map=class_map,
        class_ids=sorted(class_map),
        ignored=ignored,
        merged=merged,
        device=device,
        device_label=device_label(0 if actual.type == "cuda" else "cpu"),
        meta=read_meta(model, path),
    )
    with model_lock:
        state["bundle"] = bundle
    print(f"[MODEL] Da nap {path.name} | {bundle.meta['arch']} | {len(model.names)} lop | "
          f"giu: {sorted(set(class_map.values()))} | thiet bi: {actual}")
    return bundle


def list_models():
    files = sorted(BASE_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.pt"))
    return [
        {
            "path": rel_path(p),
            "name": p.name,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
        }
        for p in files
    ]


def rel_path(p):
    return str(p.relative_to(BASE_DIR)).replace("\\", "/")


def target_info(bundle):
    targets = []
    for key, cfg in TARGET_CLASSES.items():
        sources = [bundle.model.names[i] for i, k in bundle.class_map.items() if k == key] if bundle else []
        targets.append({"key": key, "vi": cfg["vi"], "color": cfg["color"],
                        "supported": bool(sources), "sources": sources})
    return targets


def model_info():
    b = state["bundle"]
    info = {
        "loaded": b is not None,
        "targets": target_info(b),
        "conf": state["conf"],
        "iou": state["iou"],
        "imgsz": state["imgsz"],
        "defaults": {"conf": DEFAULT_CONF, "iou": DEFAULT_IOU, "imgsz": DEFAULT_IMGSZ},
        "cuda_available": DEVICE == 0,
        "gpu_name": GPU_NAME,
        "device": b.device_label if b else device_label(DEVICE),
        "models": list_models(),
    }
    if b is None:
        return info

    names = {int(k): v for k, v in b.model.names.items()}
    missing = [t["key"] for t in info["targets"] if not t["supported"]]
    info.update(
        path=rel_path(b.path),
        name=b.path.name,
        task=b.model.task,
        names=names,
        nc=len(names),
        ignored=b.ignored,
        merged=b.merged,
        missing=missing,
        # Model huan luyen dung 4 lop muc tieu (khac voi model nhieu lop dang duoc loc)
        trained_on_targets=len(names) == len(b.class_map) and not missing,
        **b.meta,
    )
    return info


# ==========================================
# SUY LUAN
# ==========================================


def nms_xyxy(boxes, scores, iou_thr):
    """NMS don gian (numpy) - dung sau khi gop lop de tranh dem trung."""
    order = scores.argsort()[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def empty_counts():
    return {k: 0 for k in TARGET_CLASSES}


def run_detect(frame, conf=None, iou=None, imgsz=None):
    """Chay YOLO tren 1 khung hinh BGR. Chi tra ve cac detection thuoc 4 nhom muc tieu."""
    conf = state["conf"] if conf is None else conf
    iou = state["iou"] if iou is None else iou
    imgsz = state["imgsz"] if imgsz is None else imgsz

    t0 = time.perf_counter()
    with model_lock:
        b = state["bundle"]
        if b is None:
            raise RuntimeError("Chưa nạp model. Vào Cài đặt để chọn file .pt")
        result = None
        if b.class_ids:  # classes=[] se bi ultralytics hieu la "tat ca lop" -> phai chan
            kwargs = dict(source=frame, conf=conf, iou=iou, imgsz=imgsz,
                          classes=b.class_ids, verbose=False)
            try:
                result = b.model.predict(device=b.device, **kwargs)[0]
            except RuntimeError as e:
                if b.device == "cpu":
                    raise
                print(f"[MODEL] Loi GPU khi suy luan, chuyen sang CPU: {e}")
                b.device, b.device_label = "cpu", "CPU (GPU lỗi)"
                result = b.model.predict(device="cpu", **kwargs)[0]
        names = b.model.names
        class_map = b.class_map
    ms = (time.perf_counter() - t0) * 1000

    detections = []
    if result is not None and result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxyn.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        keys = np.array([class_map[c] for c in clss])

        # Loc trung giua cac lop goc da gop cung nhom (vd: Capacitor + Ceramic Capacitor)
        for key in TARGET_CLASSES:
            idx = np.where(keys == key)[0]
            if not idx.size:
                continue
            if len({clss[i] for i in idx}) > 1:
                idx = idx[nms_xyxy(boxes[idx], scores[idx], iou)]
            for i in idx:
                detections.append({
                    "key": key,
                    "label": TARGET_CLASSES[key]["vi"],
                    "orig": names[int(clss[i])],
                    "conf": round(float(scores[i]), 4),
                    "box": [round(float(v), 5) for v in boxes[i]],
                })
        detections.sort(key=lambda d: -d["conf"])

    counts = empty_counts()
    for d in detections:
        counts[d["key"]] += 1

    return {
        "detections": detections,
        "counts": counts,
        "total": len(detections),
        "avg_conf": round(sum(d["conf"] for d in detections) / len(detections), 4) if detections else None,
        "ms": round(ms, 1),
        "width": frame.shape[1],
        "height": frame.shape[0],
    }


# ==========================================
# VE KET QUA (cho luong camera IP) - dung PIL de hien tieng Viet co dau
# ==========================================


def _load_font(size):
    for f in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "C:/Windows/Fonts/arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return None


_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = _load_font(size)
    return _font_cache[size]


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def draw_detections(frame, detections):
    h, w = frame.shape[:2]
    lw = max(2, round(max(w, h) / 400))
    fs = max(14, round(max(w, h) / 50))
    font = get_font(fs)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h
        color = hex_rgb(TARGET_CLASSES[d["key"]]["color"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        text = f"{d['label']} {d['conf'] * 100:.0f}%"
        if font is None:
            text = strip_accents(text)
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, fs // 4)
        ty = y1 - th - pad * 2 if y1 - th - pad * 2 >= 0 else y1
        draw.rectangle([x1, ty, x1 + tw + pad * 2, ty + th + pad * 2], fill=color)
        draw.text((x1 + pad - tb[0], ty + pad - tb[1]), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def status_frame(base, text):
    """Khung hinh thong bao (giu ket noi trinh duyet song khi camera dang mat)."""
    if base is not None:
        img = cv2.addWeighted(base, 0.35, np.zeros_like(base), 0.65, 0)
    else:
        img = np.zeros((480, 640, 3), np.uint8)
    h, w = img.shape[:2]
    font = get_font(max(18, round(w / 36)))
    if font is None:
        text = strip_accents(text)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    tb = draw.textbbox((0, 0), text, font=font)
    draw.text(((w - (tb[2] - tb[0])) / 2, (h - (tb[3] - tb[1])) / 2), text, fill=(255, 200, 0), font=font)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# ==========================================
# TIEN ICH REQUEST
# ==========================================


def is_local_request():
    return request.remote_addr in ("127.0.0.1", "::1")


def read_upload_image(file_storage):
    ext = Path(file_storage.filename or "").suffix.lower()
    if ext and ext not in IMAGE_EXTS:
        raise ValueError("Định dạng ảnh không hỗ trợ. Dùng JPG, JPEG, PNG, WEBP hoặc BMP")
    raw = file_storage.read(MAX_IMAGE_MB * 1024 * 1024 + 1)
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(f"Ảnh vượt quá {MAX_IMAGE_MB} MB")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh (file hỏng hoặc sai định dạng)")
    return frame


def form_float(name, lo=0.01, hi=1.0):
    value = request.form.get(name)
    if value in (None, ""):
        return None
    return min(max(float(value), lo), hi)


# ==========================================
# ROUTES
# ==========================================


@app.route("/")
def index():
    return render_template("index.html", author=AUTHOR)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File quá lớn (tối đa {MAX_MODEL_MB} MB)"}), 413


@app.get("/api/status")
def api_status():
    return jsonify(model_info())


@app.post("/api/settings")
def api_settings():
    data = request.get_json(force=True, silent=True) or {}
    try:
        if "conf" in data:
            state["conf"] = min(max(float(data["conf"]), 0.01), 1.0)
        if "iou" in data:
            state["iou"] = min(max(float(data["iou"]), 0.01), 1.0)
        if "imgsz" in data:
            imgsz = int(data["imgsz"])
            if imgsz not in ALLOWED_IMGSZ:
                return jsonify({"error": "Kích thước ảnh không hợp lệ"}), 400
            state["imgsz"] = imgsz
    except (TypeError, ValueError):
        return jsonify({"error": "Giá trị cài đặt không hợp lệ"}), 400
    return jsonify(model_info())


@app.post("/api/model/load")
def api_model_load():
    if not is_local_request():
        return jsonify({"error": "Chỉ được đổi model từ chính máy chạy server"}), 403
    rel = (request.get_json(force=True, silent=True) or {}).get("path", "")
    path = (BASE_DIR / rel).resolve()
    if path.suffix != ".pt" or not path.is_file() or BASE_DIR not in path.parents:
        return jsonify({"error": "File model không hợp lệ"}), 400
    try:
        load_model(path)
    except Exception as e:
        return jsonify({"error": f"Không nạp được model: {e}"}), 400
    return jsonify(model_info())


@app.post("/api/model/upload")
def api_model_upload():
    if not is_local_request():
        return jsonify({"error": "Chỉ được tải model lên từ chính máy chạy server"}), 403
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".pt"):
        return jsonify({"error": "Hãy chọn file .pt"}), 400

    stem = Path(secure_filename(f.filename) or "model.pt").stem or "model"
    path = MODELS_DIR / f"{stem}.pt"
    n = 1
    while path.exists():  # khong ghi de file cu
        path = MODELS_DIR / f"{stem}_{n}.pt"
        n += 1
    f.save(path)
    try:
        load_model(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        return jsonify({"error": f"File .pt không phải model YOLO hợp lệ: {e}"}), 400
    return jsonify(model_info())


@app.post("/api/detect")
def api_detect():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "Thiếu ảnh"}), 400
    try:
        frame = read_upload_image(f)
        out = run_detect(frame, conf=form_float("conf"), iou=form_float("iou"))
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[DETECT] Loi: {e}")
        return jsonify({"error": f"Lỗi khi nhận diện: {e}"}), 500
    return jsonify(out)


# ---------- IP camera (vi du: app IP Webcam tren dien thoai) ----------


class IpCamReader:
    """
    Luong rieng chi de doc camera: luon giu khung hinh MOI NHAT (bo khung cu),
    co timeout va tu ket noi lai khi mat tin hieu.
    Ket qua nhan dien duoc dung chung cho moi nguoi xem (khong suy luan trung lap).
    """

    OPEN_TIMEOUT_MS = 6000
    READ_TIMEOUT_MS = 5000
    STALL_SECONDS = 6  # khong co khung moi qua lau -> ket noi lai

    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.frame_id = 0
        self.last_frame_at = 0.0
        self.status = "connecting"  # connecting | live | reconnecting
        self.reconnects = 0
        self.viewers = 0
        # Ket qua da xu ly (dung chung)
        self.process_lock = threading.Lock()
        self.out_id = 0
        self.out_part = b""
        self.last_annotated = None
        self.last_proc = time.perf_counter()
        self.stats = {"counts": empty_counts(), "total": 0, "avg_conf": None, "ms": 0.0, "fps": 0.0}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.READ_TIMEOUT_MS,
            ],
        )
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.url)  # thu backend mac dinh
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        backoff = 1.0
        while not self.stop_event.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                self.status = "reconnecting" if self.frame is not None else "connecting"
                print(f"[IPCAM] Khong mo duoc {self.url}, thu lai sau {backoff:.0f}s")
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            print(f"[IPCAM] Da ket noi {self.url}")
            self.last_frame_at = time.time()
            failures = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    failures = 0
                    backoff = 1.0
                    with self.lock:
                        self.frame = frame
                        self.frame_id += 1
                        self.last_frame_at = time.time()
                    self.status = "live"
                    continue
                failures += 1
                if failures >= 20 or time.time() - self.last_frame_at > self.STALL_SECONDS:
                    break
                time.sleep(0.05)

            cap.release()
            if not self.stop_event.is_set():
                self.reconnects += 1
                self.status = "reconnecting"
                print(f"[IPCAM] Mat tin hieu, dang ket noi lai (lan {self.reconnects})")
                self.stop_event.wait(0.5)
        print(f"[IPCAM] Da dong {self.url}")

    def latest(self):
        with self.lock:
            return self.frame_id, self.frame

    def processed(self, frame_id, frame):
        """Nhan dien khung frame_id (neu nguoi xem khac chua lam) va tra ve phan MJPEG."""
        with self.process_lock:
            if self.out_id != frame_id:
                try:
                    out = run_detect(frame)
                    annotated = draw_detections(frame, out["detections"])
                except Exception as e:
                    print(f"[IPCAM] Loi nhan dien: {e}")
                    out = {"counts": empty_counts(), "total": 0, "avg_conf": None, "ms": 0.0}
                    annotated = frame
                now = time.perf_counter()
                self.stats = {
                    "counts": out["counts"], "total": out["total"], "avg_conf": out["avg_conf"],
                    "ms": out["ms"], "fps": round(1 / max(now - self.last_proc, 1e-6), 1),
                }
                self.last_proc = now
                self.last_annotated = annotated
                self.out_part = mjpeg_part(annotated)
                self.out_id = frame_id
            return self.out_part

    def stop(self):
        self.stop_event.set()


ipcam = {"reader": None}
ipcam_lock = threading.Lock()


def get_reader(url):
    """Dung lai reader neu cung URL (khong mo 2 luong camera), neu khac thi dung reader cu."""
    with ipcam_lock:
        reader = ipcam["reader"]
        if reader is None or reader.url != url or reader.stop_event.is_set():
            if reader is not None:
                reader.stop()
            reader = IpCamReader(url)
            ipcam["reader"] = reader
        reader.viewers += 1
        return reader


def release_reader(reader):
    with ipcam_lock:
        reader.viewers -= 1
        if reader.viewers <= 0:
            reader.stop()
            if ipcam["reader"] is reader:
                ipcam["reader"] = None


def mjpeg_part(img):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"


def ipcam_frames(reader):
    last_id = 0
    last_sent = 0.0
    try:
        while not reader.stop_event.is_set():
            frame_id, frame = reader.latest()

            if frame_id == last_id:
                # Chua co khung moi: neu dang mat ket noi thi gui khung thong bao
                # moi 0.5s de trinh duyet khong cat ket noi vi im lang qua lau
                if reader.status != "live" and time.time() - last_sent > 0.5:
                    text = "Đang kết nối camera…" if reader.status == "connecting" \
                        else f"Mất tín hiệu – đang kết nối lại (lần {reader.reconnects})"
                    yield mjpeg_part(status_frame(reader.last_annotated, text))
                    last_sent = time.time()
                time.sleep(0.01)
                continue

            last_id = frame_id
            yield reader.processed(frame_id, frame)
            last_sent = time.time()
    finally:
        release_reader(reader)


@app.get("/api/ipcam/stream")
def api_ipcam_stream():
    url = request.args.get("url", "").strip()
    if not url.startswith(("http://", "https://", "rtsp://")):
        return "URL không hợp lệ", 400
    reader = get_reader(url)
    return Response(
        ipcam_frames(reader),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ipcam/stats")
def api_ipcam_stats():
    reader = ipcam["reader"]
    if reader is None:
        return jsonify({"counts": empty_counts(), "total": 0, "avg_conf": None, "ms": 0.0, "fps": 0.0,
                        "running": False, "status": "stopped", "reconnects": 0})
    return jsonify({
        **reader.stats,
        "running": not reader.stop_event.is_set(),
        "status": reader.status,
        "reconnects": reader.reconnects,
    })


@app.post("/api/ipcam/stop")
def api_ipcam_stop():
    with ipcam_lock:
        if ipcam["reader"] is not None:
            ipcam["reader"].stop()
            ipcam["reader"] = None
    return jsonify({"ok": True})


# ==========================================
# KHOI DONG
# ==========================================


def load_default_model():
    candidates = [p for p in DEFAULT_MODELS if p.exists()]
    candidates += [p for p in sorted(BASE_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.pt"))
                   if p not in candidates]
    for p in candidates:
        try:
            load_model(p)
            return
        except Exception as e:
            print(f"[MODEL] Bo qua {p.name}: {e}")
    print("[MODEL] Chua co model nao. Vao trang Cai dat de tai file .pt")


if __name__ == "__main__":
    print("Thiet bi:", device_label(DEVICE))
    load_default_model()
    print("Mo trinh duyet: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
