# Bối cảnh dự án: Web nhận diện linh kiện điện tử bằng YOLO

> File này tóm tắt dự án để một phiên chat AI khác có thể hiểu và làm tiếp.
> Cập nhật: 2026-10-14.

## 1. Mục tiêu

Đồ án sinh viên nhận diện và **đếm linh kiện điện tử** (tụ, điện trở, LED, IC…) bằng model YOLO đã huấn luyện sẵn (file `.pt`).
Có một **giao diện web chạy local** với:

- 3 chế độ nhận diện: **qua ảnh**, **qua video**, **qua camera trực tiếp** (webcam máy tính hoặc camera điện thoại qua app IP Webcam).
- Chọn được camera.
- Trang **Cài đặt**: tải lên / chọn file `.pt`, chỉnh **ngưỡng tin cậy** (confidence), IoU, kích thước ảnh đầu vào.
- **Ảnh tác giả** trên giao diện, để người xem biết ai làm web.

Người dùng nói tiếng Việt, nên trả lời bằng tiếng Việt. Giao diện có dấu, còn comment và `print` trong code Python viết không dấu.

## 2. Môi trường

- Windows 11, chạy trong VS Code. Đường dẫn dự án: `C:\Users\khanh\Desktop\Nhung1`. Không phải git repo.
- Python 3.12.10, ultralytics 8.4.116, OpenCV 4.12.0, Flask 3.1.3. Mọi thư viện đã cài đủ.
- Khi chạy, app báo thiết bị là **CUDA**. Một lần kiểm tra `torch.__version__` thì in ra `2.14.1+cpu`, có thể máy có nhiều bản Python. Nếu gặp lỗi GPU thì kiểm tra lại điểm này.

## 3. Cấu trúc thư mục

```
Nhung1/
├── app.py                  # Backend Flask + YOLO (file chính của web)
├── templates/index.html    # Giao diện (Jinja2)
├── static/
│   ├── style.css           # CSS, có giao diện sáng/tối theo hệ điều hành, responsive
│   ├── app.js              # Toàn bộ logic phía trình duyệt (JS thuần, không framework)
│   └── author.png          # Ảnh tác giả (ảnh chân dung người dùng gửi)
├── models/                 # Nơi lưu các file .pt tải lên qua web
├── best (5).pt             # Model mặc định khi khởi động (~18 MB, 10 lớp)
├── best.pt                 # Model nhỏ (~6 MB, 6 lớp)
├── best1.pt                # Cùng kích thước với best (5).pt, nhiều khả năng là bản trùng
├── a.py, b.py, c.py        # Script cũ, viết trước khi có web (xem mục 7)
└── Electronic-Component-Detection-using-YOLOv8-main.zip   # Repo YOLOv8 gốc, chưa giải nén
```

**Các lớp của model:**
- `best (5).pt`: Capacitor, Ceramic Capacitor, Diode, IC, LED, Potentiometer, Resistor, Transformer, Trigger Button, Voltage Regulator
- `best.pt`: Capacitor, Diode, LED, Resistor, Transistor, Zener Diode

## 4. Cách chạy

```
python app.py
```
Sau đó mở **http://127.0.0.1:5000**. Phải mở bằng `127.0.0.1` hoặc `localhost`, nếu mở bằng IP của máy thì trình duyệt sẽ chặn webcam vì không phải secure context.

Khi khởi động, app tự nạp `best (5).pt`. Nếu không có file này thì app nạp file `.pt` đầu tiên tìm thấy.

## 5. Kiến trúc

### 5.1 Backend: `app.py` (Flask, `threaded=True`)

Trạng thái toàn cục nằm trong `state = {model, model_path, conf, iou, imgsz}`. Mọi lần gọi `model.predict` đều được bọc trong `model_lock`.

Hàm lõi là `run_detect(frame, conf, iou, imgsz)`. Hàm này nhận ảnh BGR và trả về:
```json
{"detections": [{"cls": 0, "name": "LED", "conf": 0.91, "box": [x1,y1,x2,y2]}],
 "counts": {"LED": 1}, "total": 1, "ms": 15.2, "width": 640, "height": 480}
```
Toạ độ `box` đã được **chuẩn hoá về 0..1** (`xyxyn`). Trình duyệt tự vẽ khung lên canvas.

**API:**

| Method | Route | Chức năng |
|---|---|---|
| GET | `/` | Trang chính. Truyền biến `AUTHOR` vào template |
| GET | `/api/status` | Thông tin model, các lớp, thiết bị, conf/iou/imgsz, danh sách file .pt |
| POST | `/api/settings` | JSON `{conf?, iou?, imgsz?}` để cập nhật state của server |
| POST | `/api/model/load` | JSON `{path}` (đường dẫn tương đối trong dự án) để nạp model. Có kiểm tra đuôi `.pt` và file phải nằm trong thư mục dự án |
| POST | `/api/model/upload` | multipart `file`. Lưu vào `models/` rồi nạp. Nếu không phải model YOLO hợp lệ thì xoá file và trả lỗi 400 |
| POST | `/api/detect` | multipart `image`, kèm `conf` và `iou` (không bắt buộc). Dùng cho cả ảnh, video và webcam |
| GET | `/api/ipcam/stream?url=` | Luồng MJPEG đã vẽ khung sẵn, lấy từ camera IP |
| GET | `/api/ipcam/stats` | `{counts,total,ms,fps,running,status,reconnects}` |
| POST | `/api/ipcam/stop` | Dừng luồng đọc camera IP |

### 5.2 Camera IP: xử lý ở server (class `IpCamReader`)

Phần này đã được viết lại vì **người dùng báo camera IP chạy lâu hay bị ngắt**. Thiết kế hiện tại:

- **Một luồng riêng** chỉ làm việc đọc camera và chỉ giữ **khung hình mới nhất** (`frame_id` tăng dần). YOLO chạy trong generator của response, luôn lấy khung mới nhất, nhờ vậy không bị dồn bộ đệm và độ trễ không tăng dần.
- Mở camera bằng `cv2.CAP_FFMPEG`, kèm `CAP_PROP_OPEN_TIMEOUT_MSEC=6000` và `CAP_PROP_READ_TIMEOUT_MSEC=5000`. Nếu không mở được thì thử lại với backend mặc định.
- Tự kết nối lại khi: đọc lỗi 20 lần liên tiếp, hoặc quá 6 giây không có khung mới. Thời gian chờ giữa các lần thử bắt đầu từ 1 giây, tăng dần đến tối đa 5 giây.
- `status` có 3 giá trị: `connecting`, `live`, `reconnecting`.
- Khi mất tín hiệu, generator vẫn gửi **khung thông báo** mỗi 0,5 giây (ảnh cuối làm tối đi, kèm chữ "Mat tin hieu…") để trình duyệt không cắt kết nối.
- Chỉ có một reader toàn cục (`ipcam["reader"]`), đếm số người xem (`viewers`). Khi không còn ai xem thì reader dừng. Mở URL khác thì reader cũ bị dừng.
- **Đã thử** với một camera MJPEG giả, cứ 5 giây tự ngắt và có một lần treo 8 giây: server tự nối lại 3 lần, hình gửi lên trình duyệt chạy liên tục, ngắt xem thì reader được giải phóng. **Chưa thử với điện thoại thật.**

### 5.3 Frontend: `static/app.js`

- **Chuyển chế độ:** hàm `setMode()` chuyển giữa `image | video | camera | settings`. Khi rời chế độ camera thì tắt camera; khi rời chế độ video thì tạm dừng video. Kết quả của từng chế độ được lưu trong `state.results[mode]`.
- **Thanh ngưỡng tin cậy:** có 2 thanh trượt đồng bộ với nhau, nhận diện qua class `.conf-input` (một ở cột phải, một trong Cài đặt). Mỗi lần nhận diện, giá trị `conf` và `iou` được gửi kèm request nên thay đổi có hiệu lực ngay. Đồng thời có gửi `/api/settings` (debounce) để luồng camera IP cũng dùng giá trị mới.
- **Chế độ ảnh:** gửi file gốc lên `/api/detect` rồi vẽ khung lên canvas. Khi kéo thanh ngưỡng thì tự nhận diện lại (debounce 250 ms).
- **Video và webcam** dùng chung class `LiveDetector`:
  - Một vòng `requestAnimationFrame` vẽ liên tục khung video cùng các box gần nhất.
  - Một vòng async khác thu nhỏ khung hình (cạnh lớn nhất bằng `max(imgsz, 640)`), nén JPEG 0.85, gửi `/api/detect`, đợi kết quả rồi mới gửi khung tiếp theo.
  - Video được phát ngay trên trình duyệt bằng `URL.createObjectURL`, không tải lên server. Có điều khiển phát/dừng, tua và tốc độ 0.25×, 0.5×, 1×.
- **Chế độ camera:**
  - Danh sách camera lấy từ `enumerateDevices()`, thêm một lựa chọn `__ipcam__` cho camera điện thoại.
  - Webcam dùng `getUserMedia`.
  - Camera IP hiển thị bằng thẻ `<img src="/api/ipcam/stream…">`. Trình duyệt hỏi `/api/ipcam/stats` mỗi 700 ms và **không tự dừng** khi mất tín hiệu, chỉ đổi nhãn LIVE sang màu cam ("ĐANG KẾT NỐI LẠI"). Nếu `img` báo lỗi hoặc server báo `running=false` thì trình duyệt mở lại luồng.
  - Khi bắt đầu lại, trình duyệt đợi lệnh `/api/ipcam/stop` của lần trước chạy xong, để lệnh dừng cũ không dừng nhầm luồng mới.
- **Giao diện:**
  - Sidebar tối bên trái gồm logo "ComponentVision", menu và thẻ tác giả (bấm vào mở khung giới thiệu có ảnh lớn).
  - Phía trên có nút trạng thái model (bấm vào chuyển sang Cài đặt).
  - Cột phải hiển thị ngưỡng tin cậy, tổng số linh kiện, ms/khung, FPS và số lượng theo từng loại (có thanh màu).
  - Ở chế độ ảnh có thêm bảng chi tiết từng vật thể.
  - Mỗi lớp có một màu cố định, lấy từ `PALETTE[cls % 12]`.
  - Font Be Vietnam Pro (Google Fonts).

## 6. Việc còn dở / cần hỏi người dùng

- **Tên tác giả** đang để tạm là "Tên tác giả". Cần sửa biến `AUTHOR` (name, role, project, email) ở đầu `app.py`. Chưa biết tên thật, **đừng tự đoán**, hãy hỏi người dùng.
- **Giao diện chưa được kiểm tra bằng mắt trong trình duyệt.** Mới chỉ thử API bằng curl và kiểm tra cú pháp JS bằng `node --check`. Webcam thật và điện thoại thật cũng chưa thử.
- `/api/detect` gửi từng khung hình qua HTTP, nên FPS của webcam và video bị giới hạn bởi tốc độ request. Nếu cần nhanh hơn thì có thể chuyển sang WebSocket.
- Server là Flask dev server, chỉ để chạy local, chưa có xác thực. Không nên mở ra mạng ngoài, vì `torch.load` một file `.pt` lạ có thể chạy mã tuỳ ý.

## 7. Các script cũ (viết trước khi có web, vẫn giữ lại)

- `a.py`: nhận diện qua webcam bằng OpenCV (`CAMERA_INDEX=1`, model `best (5).pt`, conf 0.7). Code đặt độ phân giải 640×640, nhưng đa số webcam không hỗ trợ khung vuông.
- `b.py`: nhận diện qua camera điện thoại, URL `http://192.168.1.10:8080/video`, model `best.pt`, conf 0.4. Thông báo lỗi ghi nhầm tên file là "best (5).pt".
- `c.py`: gọi workflow segmentation trên Roboflow cho ảnh `led.jpg`, cần biến môi trường `ROBOFLOW_API_KEY`. File `led.jpg` không có trong thư mục.

## 8. Gợi ý cho người dùng khi dùng camera điện thoại (app IP Webcam)

- Bật "Prevent going to sleep" và tắt chế độ tiết kiệm pin cho app.
- Dùng độ phân giải 640×480 hoặc 800×600.
- Để điện thoại và máy tính cùng mạng Wi-Fi, ưu tiên băng tần 5 GHz.
