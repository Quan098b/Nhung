# Bối cảnh dự án: Web nhận diện 4 loại linh kiện điện tử bằng YOLO11

> File này tóm tắt dự án để một phiên chat AI khác có thể hiểu và làm tiếp.
> Cập nhật: 2026-10-14.

## 1. Mục tiêu

Đồ án sinh viên: phát hiện và **đếm 4 nhóm linh kiện điện tử** gồm **Resistor (Điện trở), Capacitor (Tụ điện), LED (Đèn LED), IC (Vi mạch)** bằng YOLO (ưu tiên YOLO11n).
Có một web chạy local với 3 chế độ (ảnh, video, camera trực tiếp) và trang cài đặt model. Giao diện có ảnh tác giả, dùng để trình bày trước hội đồng.

- **Không có file báo cáo đồ án** trong thư mục. Danh sách 4 lớp lấy theo yêu cầu của người dùng.
- Người dùng nói tiếng Việt, nên trả lời bằng tiếng Việt. Giao diện có dấu, còn comment và log trong Python viết **không dấu**.
- Repo: https://github.com/Quan098b/Nhung (public). Không đẩy `*.pt`, `*.zip`, `models/` lên repo, vì đã có trong `.gitignore`.

## 2. Môi trường

- Windows 11, VS Code, thư mục dự án `C:\Users\khanh\Desktop\Nhung1`.
- Python 3.12.10, ultralytics 8.4.116, OpenCV 4.12.0, Flask 3.1.3, Pillow (đi kèm ultralytics).
- GPU: **NVIDIA RTX 3050 6GB Laptop**. CUDA hoạt động, dù `torch.__version__` hiển thị `2.14.1+cpu`, nên đừng dựa vào chuỗi phiên bản này. App kiểm tra CUDA bằng cách tạo thử một tensor trên GPU, và hiển thị thiết bị thực tế lấy từ tham số của model sau khi warmup.

## 3. Các model hiện có (đã kiểm tra bằng `ultralytics.YOLO()`)

| File | MD5 | Kiến trúc | Lớp | Đủ 4 nhóm? | mAP50 (checkpoint) |
|---|---|---|---|---|---|
| `models/best_7.pt` | a426… | **YOLO11n**, 2.6M tham số, huấn luyện 2026-10-12, 62 epoch | 10: Capacitor, Connection_cable, Crystal_oscilator, Diode, IC, Inductor, LED, Potentiometer, Resistor, Transistor | Có | 0.908 |
| `best (5).pt` = `best1.pt` = `models/best_4.pt` (**3 file giống hệt nhau**) | 6c04… | YOLO11s, 9.4M, 2026-10-10, 120 epoch | 10: Capacitor, Ceramic Capacitor, Diode, IC, LED, Potentiometer, Resistor, Transformer, Trigger Button, Voltage Regulator | Có (Ceramic Capacitor được gộp vào Tụ điện) | 0.982 |
| `best.pt` | 6af1… | YOLOv8n, 3.0M, 2025-05, imgsz huấn luyện 800 | 6: Capacitor, Diode, LED, Resistor, Transistor, Zener Diode | **Thiếu IC** | 0.715 |

- **Chưa có model nào được huấn luyện với đúng 4 lớp.** Mọi model đều có nhiều lớp hơn và được lọc khi suy luận.
- Model mặc định lúc khởi động là `DEFAULT_MODELS` trong `app.py`: `models/best_7.pt` (YOLO11n), nếu không có thì `best (5).pt`.
- Khi thử trên 6 ảnh mẫu (ảnh chụp màn hình trong zip YOLOv8, không có nhãn chuẩn), 2 model YOLO11 mạnh ở những ảnh khác nhau. Trên GPU, cả hai đều mất khoảng 15–25 ms mỗi ảnh. Chưa có tập kiểm thử có nhãn nên **không kết luận được model nào chính xác hơn**.

## 4. Cấu trúc thư mục

```
app.py                  # Backend Flask + YOLO
templates/index.html    # Giao diện (Jinja2)
static/app.js           # Logic trình duyệt (JS thuần)
static/style.css        # CSS: biến màu, sáng/tối (data-theme), responsive
static/author.png       # Ảnh tác giả
models/                 # File .pt tải lên qua web (gitignore)
a.py, b.py, c.py        # Script cũ, không dùng trong web, không lọc 4 nhóm
Nhung2.zip              # Bản sao lưu cũ của dự án (64 MB, gitignore)
Electronic-Component-Detection-using-YOLOv8-main.zip  # Repo tham khảo, có 6 ảnh mẫu
```

## 5. Kiến trúc backend (`app.py`)

### 5.1 Cấu hình 4 nhóm (đầu file)
- `TARGET_CLASSES`: mỗi nhóm có `{key: {vi, color}}`. Thứ tự trong dict là thứ tự hiển thị. Màu được dùng chung cho frontend và cho phần vẽ khung trên luồng camera IP.
- `CLASS_ALIASES`: ánh xạ tên lớp gốc (sau khi chuẩn hoá bằng `norm_name`: chữ thường, bỏ ký tự đặc biệt) sang nhóm.
- `MERGE_CAPACITOR_VARIANTS = True` và `CAPACITOR_VARIANTS`: gộp Ceramic/Electrolytic Capacitor vào Tụ điện. Đặt `False` nếu báo cáo yêu cầu tách riêng.
- Mặc định: `DEFAULT_CONF=0.35`, `DEFAULT_IOU=0.45`, `DEFAULT_IMGSZ=640`. Kích thước ảnh hợp lệ nằm trong `ALLOWED_IMGSZ`.
- Giới hạn file: `MAX_MODEL_MB=300`, `MAX_IMAGE_MB=25`. Định dạng ảnh cho phép nằm trong `IMAGE_EXTS`.
- `AUTHOR`: tên tác giả, **vẫn đang để tạm "Tên tác giả"**. Hãy hỏi người dùng, đừng tự đoán.

### 5.2 Model
- `ModelBundle` (dataclass) gồm: model, path, `class_map` (id gốc → nhóm), `class_ids`, `ignored`, `merged`, `device`, `device_label`, `meta`. Khi đổi model, cả bundle được thay **nguyên khối** trong `model_lock`, nên một request đang suy luận dở không bị dính model mới.
- `load_model(path)` lần lượt:
  1. Kiểm tra magic bytes `PK\x03\x04` (định dạng zip của torch).
  2. Chỉ chấp nhận task `detect` hoặc `segment`.
  3. Dựng bảng ánh xạ lớp bằng `build_class_map`.
  4. Warmup trên GPU; nếu GPU lỗi thì chuyển sang CPU.
  5. Đọc meta từ `model.ckpt`: kiến trúc lấy từ `train_args.model`, cùng với tham số, ngày huấn luyện, epoch, và `train_metrics`.
- **Không đổi tên lớp hay class ID bên trong model.** Chỉ ánh xạ ở bước hiển thị.

### 5.3 Suy luận: `run_detect(frame, conf, iou, imgsz)`
- Gọi `predict(classes=b.class_ids, device=b.device)`, nên YOLO chỉ trả về các lớp thuộc 4 nhóm.
- **Nếu `class_ids` rỗng thì không gọi predict**, vì ultralytics hiểu `classes=[]` là lấy tất cả các lớp.
- Sau khi gộp nhóm, chạy thêm **NMS cho từng nhóm** (`nms_xyxy`, viết bằng numpy) khi một nhóm có từ 2 lớp gốc trở lên. Bước này tránh đếm trùng khi Capacitor và Ceramic Capacitor cùng khoanh một vật.
- Nếu GPU lỗi lúc đang suy luận thì tự chuyển sang CPU.
- Kết quả trả về:
```json
{"detections":[{"key":"Capacitor","label":"Tụ điện","orig":"Ceramic Capacitor","conf":0.8,"box":[x1,y1,x2,y2]}],
 "counts":{"Resistor":0,"Capacitor":1,"LED":0,"IC":0}, "total":1, "avg_conf":0.8, "ms":20.1, "width":640, "height":480}
```
  - `box` được chuẩn hoá về khoảng 0..1.
  - `counts` luôn có đủ 4 khoá.

### 5.4 API (giữ nguyên route; dữ liệu trả về đã mở rộng)

| Method | Route | Ghi chú |
|---|---|---|
| GET | `/api/status` | Ngoài thông tin cũ, có thêm: `targets` (4 nhóm, kèm `supported` và `sources`), `names`, `nc`, `ignored`, `merged`, `missing`, `trained_on_targets`, `arch`, `params_m`, `metrics`, `train_*`, `device`, `defaults` |
| POST | `/api/settings` | `{conf?, iou?, imgsz?}`. Giá trị imgsz không hợp lệ sẽ trả 400 |
| POST | `/api/model/load` | `{path}` nằm trong thư mục dự án. **Chỉ nhận request từ localhost** |
| POST | `/api/model/upload` | Lưu vào `models/` và **không ghi đè** file đã có (tự thêm `_1`, `_2`…). File không hợp lệ bị xoá ngay. Chỉ nhận request từ localhost |
| POST | `/api/detect` | multipart `image`, kèm `conf` và `iou` (không bắt buộc) |
| GET | `/api/ipcam/stream?url=` | Luồng MJPEG; khung được vẽ bằng PIL nên nhãn tiếng Việt có dấu |
| GET | `/api/ipcam/stats` | `{counts, total, avg_conf, ms, fps, running, status, reconnects}` |
| POST | `/api/ipcam/stop` | Dừng camera IP. Trình duyệt cũng gọi qua `sendBeacon` khi đóng trang |

### 5.5 Camera IP (`IpCamReader`)
- Có một luồng riêng chỉ để đọc camera và chỉ giữ khung hình mới nhất.
- Mở camera với timeout: 6 giây khi mở, 5 giây khi đọc.
- Tự kết nối lại; thời gian chờ giữa các lần thử tăng dần từ 1 đến 5 giây.
- Trạng thái: `connecting`, `live` hoặc `reconnecting`.
- Chỉ có **một reader cho mỗi URL**, kèm bộ đếm `viewers`. Khi không còn ai xem thì reader dừng.
- **Kết quả nhận diện dùng chung** cho mọi người xem (qua `processed()` và `process_lock`), nên 2 tab mở cùng lúc không làm suy luận 2 lần.

## 6. Frontend (`static/app.js`)
- `state.targets` lấy từ server; màu và tên tiếng Việt đều lấy từ đây.
- Cột kết quả **luôn hiện đủ 4 nhóm**. Nhóm nào model không hỗ trợ thì hiện “—” kèm dòng “Model hiện tại không có lớp này”.
- Thanh cảnh báo `#modelAlert` có 2 mức:
  - **warn:** model thiếu nhóm, hoặc chưa nạp model.
  - **info:** model nhiều lớp đang được lọc. Người dùng có thể ẩn thông báo này trong phiên làm việc.
- Thẻ "Mô hình" ở cột phải; trang Cài đặt có bảng ánh xạ lớp, các chỉ số đọc từ checkpoint, và nút “Khôi phục mặc định”.
- `LiveDetector` dùng chung cho video và webcam:
  - Mỗi lúc chỉ có 1 request đang chờ.
  - Khung hình gửi đi được thu nhỏ, cạnh lớn nhất bằng `max(imgsz, 640)`.
  - Có callback `onDetected`.
- Video có 2 cách xử lý (`#videoProcMode`):
  - **Thời gian thực:** video phát đúng tốc độ.
  - **Từng khung hình:** video luôn ở trạng thái tạm dừng. Mỗi lần nhận diện xong, `video.next()` tua thêm 1/30 giây.
  - Số đếm luôn là của **khung hiện tại**, không cộng dồn giữa các khung.
- Camera:
  - Danh sách lấy từ `enumerateDevices`, gồm webcam laptop và camera USB, cộng thêm lựa chọn camera IP.
  - URL camera IP **không hardcode**; URL đã dùng được lưu trong localStorage.
  - Khi chuyển sang chế độ khác thì camera bị tắt.
- Sáng/tối: nút xoay vòng Tự động → Sáng → Tối, lưu trong localStorage. CSS dùng `:root[data-theme]` và `prefers-color-scheme`.

## 7. Đã kiểm thử (2026-10-14)
- `py_compile` cho `app.py` và `node --check` cho `app.js` đều đạt. Server Flask khởi động được.
- Các API: status, nạp 3 loại model, nhận diện ảnh PNG và WEBP. Các trường hợp lỗi đều trả 400: imgsz sai, đường dẫn ra ngoài thư mục dự án, file không phải `.pt`, ảnh `.gif`, ảnh hỏng, file `.pt` rác. Tải lên trùng tên thì không ghi đè.
- **Lọc 4 nhóm:** so với model gốc không lọc, web đã bỏ đúng các lớp ngoài phạm vi (ví dụ Diode). Tổng `counts` bằng `total`. Với `best.pt`, API báo đúng là thiếu IC.
- Kiểm thử đơn vị cho NMS gộp lớp và cho `build_class_map` (tên dạng số nhiều, gạch dưới, gạch ngang).
- **Camera IP:** chỉ thử với **camera MJPEG giả lập**. Kết quả: tự nối lại sau khi mất kết nối và sau khi bị treo 8 giây; 2 người xem dùng chung 1 kết nối; reader được giải phóng khi đóng. **Chưa thử với điện thoại thật.**
- **Giao diện:** đã chụp bằng Chrome headless (qua DevTools Protocol) ở các khổ 1440px, 820px và 390px, cả sáng lẫn tối. Đã thấy nhận diện ảnh, video (cả 2 cách xử lý), trang Cài đặt và camera ở trạng thái chưa bật. Đã sửa lỗi tràn ngang trên điện thoại.
- **Chưa thử:** webcam thật, camera USB, điện thoại thật, trình duyệt có người dùng thao tác thật.

## 8. Việc còn dở
- Tên tác giả trong `AUTHOR`.
- Chưa có model được huấn luyện với đúng 4 lớp. Khi có, làm theo hướng dẫn trong README (mục “Thay bằng model YOLO11n mới”).
- `b.py` vẫn hardcode IP camera. Đây là script cũ, chưa sửa.
- Server là Flask dev server, chỉ chạy local. Không mở ra mạng ngoài.
