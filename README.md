# Face Recognition Check-in App

Đây là một ứng dụng FastAPI để thực hiện check-in thông qua nhận dạng khuôn mặt. Ứng dụng sử dụng mô hình học sâu để nhận diện và kiểm tra khuôn mặt từ ảnh gửi lên.

## Yêu cầu

- Python 3.7 hoặc cao hơn
- Một môi trường ảo (Virtual Environment) được cài đặt (khuyến khích)
- `fastapi`, `uvicorn`, và các thư viện cần thiết khác

## Cài đặt

1. **Clone hoặc tải mã nguồn về máy tính**:

   Nếu bạn chưa tải về mã nguồn, hãy dùng lệnh sau để clone từ GitHub (hoặc tải mã nguồn theo cách khác):

   ```bash
   git clone https://github.com/luan-thnh/face_recognition.git
   cd face_recognition
   ```

2. **Tạo môi trường ảo (Virtual Environment)**:

   Nếu bạn chưa có môi trường ảo, bạn có thể tạo một môi trường ảo mới với lệnh:

   ```bash
   python -m venv venv
   ```

3. **Kích hoạt môi trường ảo**:

   - Trên Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Trên macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Cài đặt các thư viện yêu cầu**:

   Sau khi đã kích hoạt môi trường ảo, hãy cài đặt các thư viện cần thiết:

   ```bash
   pip install -r requirements.txt
   ```

   Hoặc cài đặt từng thư viện cần thiết nếu không có file `requirements.txt`:

   ```bash
   pip install fastapi uvicorn torch torchvision pillow numpy
   ```

   > Nếu bạn sử dụng mô hình nhận diện khuôn mặt tùy chỉnh, hãy đảm bảo rằng bạn có file mô hình (`face_recognition_model.pth`) và danh sách lớp (`classes.txt`).

## Chạy ứng dụng

1. **Chạy server FastAPI**:

   Sử dụng `uvicorn` để chạy ứng dụng FastAPI. Từ thư mục chứa mã nguồn, hãy chạy lệnh sau:

   ```bash
   uvicorn main:app --reload
   ```

   - `main`: tên file Python chứa ứng dụng FastAPI (nếu file của bạn là `main.py`).
   - `--reload`: cho phép tự động tải lại server khi có thay đổi trong mã nguồn (chỉ dành cho môi trường phát triển).

   Server sẽ chạy tại `http://127.0.0.1:8000`.

2. **Kiểm tra API**:

   Bạn có thể truy cập tài liệu API của FastAPI tại địa chỉ:

   ```
   http://127.0.0.1:8000/docs
   ```

   Tại đây, bạn có thể thử nghiệm các endpoint của API, bao gồm:

   - **POST /checkin/**: Endpoint nhận ảnh và trả về kết quả nhận diện khuôn mặt.

## Ví dụ sử dụng API

1. **Gửi ảnh qua API**:

   Bạn có thể gửi ảnh (dạng JPEG hoặc PNG) qua API `POST /checkin/` để nhận diện khuôn mặt. Ví dụ, bạn có thể sử dụng công cụ như Postman hoặc cURL để gửi ảnh.

   - **Postman**: Tạo một yêu cầu POST đến `http://127.0.0.1:8000/checkin/` và chọn file ảnh từ máy tính.
   - **cURL**:
     ```bash
     curl -X 'POST' \
       'http://127.0.0.1:8000/checkin/' \
       -H 'accept: application/json' \
       -H 'Content-Type: multipart/form-data' \
       -F 'file=@path/to/your/image.jpg'
     ```

2. **Phản hồi từ API**:

   - Nếu nhận diện thành công:

     ```json
     {
       "status": "success",
       "person_id": "person_1",
       "confidence": 0.92
     }
     ```

   - Nếu nhận diện thất bại:
     ```json
     {
       "status": "fail",
       "reason": "low confidence"
     }
     ```

## Các bước tiếp theo

- Tối ưu hóa mô hình nhận diện khuôn mặt để đạt được độ chính xác cao hơn.
- Cải thiện giao diện người dùng (UI) với các công nghệ frontend như React hoặc Vue.js.
- Cấu hình ứng dụng để chạy trên một server thực tế (ví dụ: Heroku, DigitalOcean).

## Giấy phép

Dự án này sử dụng giấy phép MIT. Bạn có thể tự do sử dụng, sao chép, chỉnh sửa và phân phối lại mã nguồn với điều kiện giữ lại giấy phép này.

```

### Các lưu ý:

1. **Cấu trúc thư mục dự án**:
   Đảm bảo rằng bạn có cấu trúc thư mục hợp lý, chẳng hạn như:
```

/your-project-directory
├── app.py # File chính của ứng dụng FastAPI
├── face_recognition_model.pth # Mô hình nhận diện khuôn mặt
├── classes.txt # Danh sách các lớp khuôn mặt
├── requirements.txt # Các thư viện cần cài đặt
└── README.md # File hướng dẫn

```

2. **`requirements.txt`**:
Đảm bảo bạn có file `requirements.txt` với các thư viện cần thiết:
```

fastapi
uvicorn
torch
torchvision
pillow
numpy

```

```
