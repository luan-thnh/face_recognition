from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo model và các công cụ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model nhận diện
classes = []
try:
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    classes = ["Unknown"]

model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=len(classes))
model.load_state_dict(torch.load("arcface_recognition_model.pth", map_location=device))
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
    """
    Nhận video từ Frontend, xử lý và trả về danh mục nhận diện.
    """
    try:
        # Lưu file tạm thời
        video_path = f"temp_video_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Mở video để xử lý
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to process video")

        results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển đổi BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Dò khuôn mặt
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = frame_rgb[y1:y2, x1:x2]

                    if face.size > 0:
                        face_pil = Image.fromarray(face)
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)

                        # Dự đoán danh mục
                        with torch.no_grad():
                            output = model(face_tensor)
                            prob = torch.nn.functional.softmax(output, dim=1)
                            confidence, prediction = torch.max(prob, 1)
                            predicted_class = classes[prediction.item()]
                            conf_value = confidence.item()

                            results.append({
                                "class": predicted_class,
                                "confidence": round(conf_value, 2)
                            })

        cap.release()
        os.remove(video_path)  # Xóa file tạm thời

        if not results:
            return {"message": "No faces detected in the video", "predictions": []}

        return {"message": "Success", "predictions": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
