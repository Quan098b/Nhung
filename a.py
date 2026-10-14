
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

# =====================================================
# 1. CAU HINH
# =====================================================

MODEL_PATH = Path(__file__).resolve().parent / "best (5).pt"

CAMERA_INDEX = 1
CONFIDENCE = 0.7
IMG_SIZE = 640

# =====================================================
# 2. KIEM TRA MODEL
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}"
    )

device = 0 if torch.cuda.is_available() else "cpu"

print("Thiet bi:", "CUDA" if device == 0 else "CPU")

# =====================================================
# 3. NAP MODEL YOLO11
# =====================================================

model = YOLO(str(MODEL_PATH))

print("Da nap model thanh cong!")
print("Danh sach linh kien:")

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")

# =====================================================
# 4. KHOI TAO CAMERA
# =====================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Khong the mo camera!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

print("\nCAMERA DANG HOAT DONG")
print("Nhan Q de thoat")
print("Nhan S de luu anh nhan dien")

# =====================================================
# 5. NHAN DIEN TRUC TIEP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc hinh anh!")
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Dem linh kien trong khung hinh
    counts = {}

    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]

            counts[name] = counts.get(name, 0) + 1

    # Hien thi tong so linh kien
    total = sum(counts.values())

    cv2.putText(
        annotated_frame,
        f"Total components: {total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Hien thi so luong tung loai
    y = 80

    for name, count in counts.items():

        cv2.putText(
            annotated_frame,
            f"{name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        y += 35

    cv2.imshow(
        "YOLO11 - Electronic Components Detection",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        cv2.imwrite(
            "ketqua.jpg",
            annotated_frame
        )
        print("Da luu anh ketqua.jpg")

# =====================================================
# 6. GIAI PHONG CAMERA
# =====================================================

cap.release()
cv2.destroyAllWindows()

print("Da dong chuong trinh!")
