
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

# ==========================================
# CAU HINH
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

# DOI IP NAY THANH DIA CHI TREN DIEN THOAI
PHONE_URL = "http://192.168.1.10:8080/video"

CONFIDENCE = 0.40
IMG_SIZE = 640

# ==========================================
# NAP MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError("Khong tim thay best (5).pt")

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH)
print("Device:", device)
print("Classes:", model.names)

# ==========================================
# KET NOI CAMERA DIEN THOAI
# ==========================================

cap = cv2.VideoCapture(PHONE_URL)

if not cap.isOpened():
    raise RuntimeError(
        "Khong ket noi duoc camera dien thoai. "
        "Hay kiem tra IP va Wi-Fi."
    )

print("DA KET NOI CAMERA DIEN THOAI")
print("Nhan Q de thoat")
print("Nhan S de luu anh")

# ==========================================
# NHAN DIEN TRUC TIEP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Mat tin hieu camera!")
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    counts = Counter()

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = model.names[class_id]
            counts[name] += 1

    # Tong so linh kien
    total = sum(counts.values())

    cv2.putText(
        annotated,
        f"Total: {total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # So luong theo lop
    y = 80

    for name, count in counts.items():

        cv2.putText(
            annotated,
            f"{name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        y += 35

    cv2.imshow(
        "YOLO11 - PHONE CAMERA",
        annotated
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        cv2.imwrite("ketqua.jpg", annotated)
        print("Da luu ketqua.jpg")

cap.release()
cv2.destroyAllWindows()
