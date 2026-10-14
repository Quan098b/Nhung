# Nhung – Nhận diện và đếm 4 loại linh kiện điện tử bằng YOLO11

Ứng dụng web chạy local (Flask + Ultralytics YOLO) để phát hiện và đếm **4 nhóm linh kiện**:

| Nhóm | Tên tiếng Việt |
|---|---|
| Resistor | Điện trở |
| Capacitor | Tụ điện (gộp cả Ceramic Capacitor nếu model có lớp này) |
| LED | Đèn LED |
| IC | Vi mạch |

Các lớp khác mà model có thể nhận ra (Diode, Transistor, Transformer…) sẽ **bị bỏ qua**: không hiển thị và không được đếm.

## Tính năng
- **3 chế độ nhận diện**
  - **Ảnh:** JPG, JPEG, PNG, WEBP, BMP. Có bảng chi tiết từng đối tượng và cho lưu ảnh kết quả.
  - **Video:** có 2 cách xử lý. *Thời gian thực* thì phát đúng tốc độ, bỏ qua khung khi nhận diện không kịp. *Từng khung hình* thì xử lý lần lượt, không bỏ khung nào.
  - **Camera trực tiếp:** webcam laptop, camera USB hoặc camera điện thoại qua app IP Webcam. Camera IP tự kết nối lại khi mất tín hiệu.
- **Thống kê:** tổng số linh kiện, số lượng theo từng nhóm, độ tin cậy trung bình, thời gian suy luận, FPS, thiết bị đang chạy (GPU hay CPU).
- **Cài đặt:**
  - Tải lên hoặc chọn file `.pt`.
  - Xem bảng ánh xạ từ lớp gốc của model sang 4 nhóm.
  - Chỉnh Confidence (mặc định 0.35), IoU (mặc định 0.45) và kích thước ảnh đầu vào (mặc định 640).
- **Cảnh báo** khi model thiếu một trong 4 nhóm, hoặc khi model có nhiều lớp và đang được lọc còn 4 nhóm.
- Giao diện tiếng Việt, có chế độ sáng/tối, dùng được trên điện thoại.

> **Lưu ý:** việc lọc còn 4 nhóm diễn ra lúc suy luận. Một model 10 lớp được lọc như vậy **không phải** là model được huấn luyện với đúng 4 lớp.

## Cài đặt và chạy
Yêu cầu: Python 3.12, Windows 11 (đã thử nghiệm). GPU NVIDIA là không bắt buộc; nếu không có, chương trình tự chạy bằng CPU.

```bash
pip install flask ultralytics opencv-python
python app.py
```
Sau đó mở http://127.0.0.1:5000. Phải dùng `127.0.0.1` hoặc `localhost` thì trình duyệt mới cho phép dùng webcam.

## Model
File `.pt` **không có trong repo** vì quá nặng. Có 3 cách đưa model vào:
- Đặt file `.pt` vào thư mục dự án hoặc thư mục `models/`, rồi chọn trong trang **Cài đặt model**.
- Tải file lên trong trang **Cài đặt model**. File sẽ được lưu vào `models/`.
- Đổi model mặc định khi khởi động bằng danh sách `DEFAULT_MODELS` trong `app.py`.

**Thay bằng model YOLO11n mới (ví dụ huấn luyện 4 lớp):**
1. Chép `best.pt` vào thư mục `models/` và đổi tên, ví dụ `models/yolo11n_4lop.pt`.
2. Đưa đường dẫn đó lên đầu danh sách `DEFAULT_MODELS` trong `app.py`, hoặc bấm "Nạp" trong trang Cài đặt.
3. Tên lớp trong model có thể là `Resistor`, `Capacitor`, `LED`, `IC`. Chương trình không phân biệt hoa thường, ký tự `_`/`-`/khoảng trắng, hay dạng số nhiều. Với tên khác, thêm vào `CLASS_ALIASES` trong `app.py`.
4. Mở trang Cài đặt và kiểm tra bảng "Ánh xạ lớp". Nếu model đúng 4 lớp mục tiêu, trang sẽ ghi “Model được huấn luyện đúng 4 lớp mục tiêu”.

Chỉ nạp file `.pt` từ nguồn tin cậy, vì file `.pt` có thể chứa mã thực thi khi được nạp.

## Cấu trúc
| File | Mô tả |
|---|---|
| `app.py` | Backend Flask + YOLO: cấu hình 4 nhóm, ánh xạ lớp, API, luồng camera IP |
| `templates/index.html` | Giao diện |
| `static/app.js`, `static/style.css` | Logic và giao diện phía trình duyệt |
| `a.py`, `b.py` | Script cũ, nhận diện bằng OpenCV qua webcam hoặc camera điện thoại (không lọc 4 nhóm) |
| `c.py` | Script cũ, gọi workflow Roboflow |
| `PROJECT_CONTEXT.md` | Mô tả chi tiết kiến trúc và trạng thái dự án |
