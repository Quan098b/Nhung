# Nhung – Web nhận diện linh kiện điện tử bằng YOLO

Giao diện web chạy local để phát hiện và đếm linh kiện điện tử (tụ, điện trở, LED, IC…) bằng model YOLO (`.pt`).

## Tính năng
- 3 chế độ: **nhận diện qua ảnh**, **qua video**, **camera trực tiếp** (webcam hoặc camera điện thoại qua app IP Webcam)
- Chọn camera, chụp/lưu ảnh kết quả
- Cài đặt: tải lên / chọn file `.pt`, chỉnh ngưỡng tin cậy, IoU, kích thước ảnh
- Camera IP tự kết nối lại khi mất tín hiệu

## Cài đặt & chạy
```bash
pip install flask ultralytics opencv-python
python app.py
```
Mở http://127.0.0.1:5000

> File model `.pt` không có trong repo (quá nặng). Đặt file `.pt` vào thư mục dự án, hoặc tải lên trong trang **Cài đặt model**.

## Cấu trúc
| File | Mô tả |
|---|---|
| `app.py` | Backend Flask + YOLO |
| `templates/index.html` | Giao diện |
| `static/app.js`, `static/style.css` | Logic & giao diện phía trình duyệt |
| `a.py`, `b.py` | Script nhận diện bằng OpenCV (webcam / camera điện thoại) |
| `c.py` | Gọi workflow Roboflow |
| `PROJECT_CONTEXT.md` | Mô tả chi tiết kiến trúc dự án |
