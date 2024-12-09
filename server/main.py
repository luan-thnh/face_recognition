from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import transforms
from PIL import Image
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = []
try:
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    classes = ["Unknown"]

model = InceptionResnetV1(pretrained="vggface2",
                          classify=True, num_classes=len(classes))
model.load_state_dict(torch.load(
    "arcface_recognition_model.pth", map_location=device))
model = model.to(device)
model.eval()

mtcnn = MTCNN(keep_all=True, device=device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Đọc file Excel
        df = pd.read_excel("excel_files/ST21A2A.xlsx")

        video_path = f"temp_video_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400, detail="Unable to process video")

        # Giới hạn số khung hình để xử lý
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Chọn mẫu khung hình (ví dụ: 1 khung/giây)
        frame_interval = max(1, int(fps))

        predictions = []
        confidences = {}

        # Sử dụng ThreadPoolExecutor để xử lý song song
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Danh sách nhiệm vụ
            future_to_frame = {}

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Chỉ xử lý các khung hình theo interval
                if frame_count % frame_interval == 0:
                    # Giảm độ phân giải để tăng tốc độ
                    frame = cv2.resize(frame, (640, 360))

                    # Gửi nhiệm vụ xử lý
                    future = executor.submit(process_frame, frame)
                    future_to_frame[future] = frame_count

                frame_count += 1

            # Xử lý kết quả
            for future in as_completed(future_to_frame):
                result = future.result()
                if result:
                    predicted_class, conf_value = result
                    predictions.append(predicted_class)

                    # Cập nhật confidence
                    if predicted_class not in confidences or conf_value > confidences.get(predicted_class, 0):
                        confidences[predicted_class] = conf_value

        cap.release()
        os.remove(video_path)

        if not predictions:
            return {"message": "No faces detected in the video", "prediction": None, "confidence": None}

        # Tìm class xuất hiện nhiều nhất
        most_common_class = Counter(predictions).most_common(1)[0][0]
        highest_confidence = confidences[most_common_class]

        # Tìm tên sinh viên dựa trên ID
        student_row = df[df['ID'] == int(most_common_class)]
        if not student_row.empty:
            student_name = student_row['Họ và tên'].values[0]

            # Tạo tên cột check-in theo ngày hiện tại
            current_date = datetime.now().strftime("%d/%m/%Y")
            checkin_column = current_date

            # Nếu cột check-in chưa tồn tại, tạo mới
            if checkin_column not in df.columns:
                df[checkin_column] = ''

            # Cập nhật check-in cho sinh viên
            df.loc[df['ID'] == int(most_common_class), checkin_column] = 'X'

            # Lưu file Excel
            df.to_excel("excel_files/ST21A2A.xlsx", index=False)

            return {
                "message": "Success",
                "prediction": most_common_class,
                "student_name": student_name,
                "confidence": round(highest_confidence, 2),
            }
        else:
            return {"message": "Student not found", "prediction": most_common_class}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


def process_frame(frame):
    """
    Xử lý một khung hình riêng lẻ
    """
    try:
        # Chuyển sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face = frame_rgb[y1:y2, x1:x2]

                if face.size > 0:
                    # Xử lý khuôn mặt
                    face_pil = Image.fromarray(face)
                    face_tensor = transform(
                        face_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(face_tensor)
                        prob = torch.nn.functional.softmax(output, dim=1)
                        confidence, prediction = torch.max(prob, 1)

                        predicted_class = classes[prediction.item()]
                        conf_value = confidence.item()

                        return predicted_class, conf_value

        return None
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


@app.get("/checkin_history/")
async def get_checkin_history():
    try:
        # Đọc file Excel
        df = pd.read_excel("excel_files/ST21A2A.xlsx")

        # Xử lý các giá trị NaN
        df = df.fillna('')

        # Tạo list để lưu kết quả
        result = []

        # Lấy danh sách các cột check-in (các cột có định dạng ngày)
        checkin_columns = [col for col in df.columns if col not in [
            'STT', 'ID', 'Họ và tên']]

        # Duyệt qua từng hàng trong DataFrame
        for _, row in df.iterrows():
            # Tạo dictionary checkins
            checkins = {date: str(row[date]) for date in checkin_columns}

            # Tạo dictionary cho mỗi sinh viên
            student_dict = {
                "id": str(row['ID']),
                "name": str(row['Họ và tên']),
                "checkins": checkins
            }
            result.append(student_dict)

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
