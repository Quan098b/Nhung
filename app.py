"""
Web nhan dien linh kien dien tu bang YOLO.
Chay:  python app.py   roi mo  http://127.0.0.1:5000
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ==========================================
# CAU HINH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = BASE_DIR / "best (5).pt"

# Thong tin tac gia hien thi tren web - sua tai day
AUTHOR = {
    "name": "Tên tác giả",
    "role": "Sinh viên · Thiết kế & phát triển",
    "project": "Đồ án nhận diện linh kiện điện tử bằng YOLO",
    "email": "",
}

DEVICE = 0 if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# ==========================================
# QUAN LY MODEL
# ==========================================

state = {
    "model": None,
    "model_path": None,
    "conf": 0.5,
    "iou": 0.45,
    "imgsz": 640,
}
model_lock = threading.Lock()

# Thong ke cua luong IP camera (cap nhat lien tuc)
stream_stats = {"counts": {}, "total": 0, "ms": 0.0, "fps": 0.0}


def list_models():
    files = sorted(BASE_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.pt"))
    return [
        {
            "path": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
            "name": p.name,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
        }
        for p in files
    ]


def load_model(path: Path):
    model = YOLO(str(path))
    with model_lock:
        state["model"] = model
        state["model_path"] = path
    print(f"Da nap model: {path.name} | classes: {model.names}")


def model_info():
    model = state["model"]
    path = state["model_path"]
    return {
        "loaded": model is not None,
        "path": str(path.relative_to(BASE_DIR)).replace("\\", "/") if path else None,
        "name": path.name if path else None,
        "classes": {int(k): v for k, v in model.names.items()} if model else {},
        "task": getattr(model, "task", None) if model else None,
        "device": "CUDA" if DEVICE == 0 else "CPU",
        "conf": state["conf"],
        "iou": state["iou"],
        "imgsz": state["imgsz"],
        "models": list_models(),
    }


def run_detect(frame, conf=None, iou=None, imgsz=None):
    """Chay YOLO tren 1 khung hinh BGR, tra ve danh sach box chuan hoa (0..1)."""
    model = state["model"]
    if model is None:
        raise RuntimeError("Chưa nạp model. Vào Cài đặt để chọn file .pt")

    t0 = time.perf_counter()
    with model_lock:
        result = model.predict(
            source=frame,
            conf=state["conf"] if conf is None else conf,
            iou=state["iou"] if iou is None else iou,
            imgsz=state["imgsz"] if imgsz is None else imgsz,
            device=DEVICE,
            verbose=False,
        )[0]
    ms = (time.perf_counter() - t0) * 1000

    detections = []
    counts = {}
    if result.boxes is not None:
        for xyxyn, c, cls in zip(
            result.boxes.xyxyn.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            cls = int(cls)
            name = model.names[cls]
            detections.append(
                {"cls": cls, "name": name, "conf": round(c, 4), "box": [round(v, 5) for v in xyxyn]}
            )
            counts[name] = counts.get(name, 0) + 1

    return {
        "detections": detections,
        "counts": counts,
        "total": len(detections),
        "ms": round(ms, 1),
        "width": frame.shape[1],
        "height": frame.shape[0],
    }, result


def read_upload_image(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh")
    return frame


def form_float(name):
    value = request.form.get(name)
    return float(value) if value not in (None, "") else None


# ==========================================
# ROUTES
# ==========================================


@app.route("/")
def index():
    return render_template("index.html", author=AUTHOR)


@app.get("/api/status")
def api_status():
    return jsonify(model_info())


@app.post("/api/settings")
def api_settings():
    data = request.get_json(force=True)
    if "conf" in data:
        state["conf"] = min(max(float(data["conf"]), 0.01), 1.0)
    if "iou" in data:
        state["iou"] = min(max(float(data["iou"]), 0.01), 1.0)
    if "imgsz" in data:
        state["imgsz"] = int(data["imgsz"])
    return jsonify(model_info())


@app.post("/api/model/load")
def api_model_load():
    rel = request.get_json(force=True).get("path", "")
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
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".pt"):
        return jsonify({"error": "Hãy chọn file .pt"}), 400
    name = secure_filename(f.filename) or f"model_{int(time.time())}.pt"
    path = MODELS_DIR / name
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
        out, _ = run_detect(frame, conf=form_float("conf"), iou=form_float("iou"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(out)


# ---------- IP camera (vi du: app IP Webcam tren dien thoai) ----------


class IpCamReader:
    """
    Luong rieng chi de doc camera: luon giu khung hinh MOI NHAT (bo khung cu),
    co timeout va tu ket noi lai khi mat tin hieu.
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

    def latest(self):
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.stop_event.set()


ipcam = {"reader": None}
ipcam_lock = threading.Lock()


def get_reader(url):
    """Dung lai reader neu cung URL, neu khac thi dung reader cu."""
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


def status_frame(base, text):
    """Khung hinh thong bao (giu ket noi trinh duyet song khi camera dang mat)."""
    img = base.copy() if base is not None else np.zeros((480, 640, 3), np.uint8)
    if base is not None:
        img = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
    h, w = img.shape[:2]
    scale = max(w / 900, 0.6)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def mjpeg_part(img):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"


def ipcam_frames(reader):
    last_id = 0
    last_annotated = None
    last_sent = 0.0
    last = time.perf_counter()
    try:
        while not reader.stop_event.is_set():
            frame_id, frame = reader.latest()

            if frame_id == last_id:
                # Chua co khung moi: neu dang mat ket noi thi gui khung thong bao
                # moi 0.5s de trinh duyet khong cat ket noi vi im lang qua lau
                if reader.status != "live" and time.time() - last_sent > 0.5:
                    text = "Dang ket noi camera..." if reader.status == "connecting" \
                        else f"Mat tin hieu - dang ket noi lai (lan {reader.reconnects})"
                    yield mjpeg_part(status_frame(last_annotated, text))
                    last_sent = time.time()
                time.sleep(0.01)
                continue

            last_id = frame_id
            try:
                out, result = run_detect(frame)
                annotated = result.plot()
            except Exception:
                out, annotated = {"counts": {}, "total": 0, "ms": 0}, frame
            now = time.perf_counter()
            stream_stats.update(
                counts=out["counts"], total=out["total"], ms=out["ms"],
                fps=round(1 / max(now - last, 1e-6), 1),
            )
            last = now
            last_annotated = annotated
            yield mjpeg_part(annotated)
            last_sent = time.time()
    finally:
        release_reader(reader)


@app.get("/api/ipcam/stream")
def api_ipcam_stream():
    url = request.args.get("url", "")
    if not url.startswith(("http://", "https://", "rtsp://")):
        return "URL không hợp lệ", 400
    reader = get_reader(url)
    return Response(
        ipcam_frames(reader),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ipcam/stop")
def api_ipcam_stop():
    with ipcam_lock:
        if ipcam["reader"] is not None:
            ipcam["reader"].stop()
            ipcam["reader"] = None
    return jsonify({"ok": True})


@app.get("/api/ipcam/stats")
def api_ipcam_stats():
    reader = ipcam["reader"]
    return jsonify({
        **stream_stats,
        "running": reader is not None and not reader.stop_event.is_set(),
        "status": reader.status if reader else "stopped",
        "reconnects": reader.reconnects if reader else 0,
    })


# ==========================================
# KHOI DONG
# ==========================================

if __name__ == "__main__":
    startup = DEFAULT_MODEL if DEFAULT_MODEL.exists() else next(iter(BASE_DIR.glob("*.pt")), None)
    if startup:
        load_model(startup)
    print("Thiet bi:", "CUDA" if DEVICE == 0 else "CPU")
    print("Mo trinh duyet: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
