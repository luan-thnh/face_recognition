from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from face_recognition import FaceRecognition 
from PIL import Image
import numpy as np
import os
import train_model
from datetime import datetime
# import requests
# import io

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo đối tượng FaceRecognition
# def download_model_from_github():
#     url = "https://github.com/luan-thnh/face_recognition/blob/main/server/face_recognition_model.pth"  # Replace with your GitHub raw URL
#     response = requests.get(url)

#     if response.status_code == 200:
#         model_data = io.BytesIO(response.content)
#         return model_data
#     else:
#         raise Exception(f"Failed to download model from GitHub, status code: {response.status_code}")
    
# model_path = download_model_from_github()
face_recognition = FaceRecognition(model_path='./face_recognition_model.pth', classes_path="classes.txt")
# face_recognition.load(model_path)  # Loading the model


@app.get("/")
async def root():
    return {"message": "Hello, Welcome to Checkify API"}

@app.post("/checkin/")
async def checkin(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        person_id, confidence = face_recognition.predict(image_np)

        if confidence > 0.7:
            dataset_dir = "dataset"
            person_dir = os.path.join(dataset_dir, str(person_id))
            os.makedirs(person_dir, exist_ok=True)

            existing_files = os.listdir(person_dir)
            count = len([f for f in existing_files if f.endswith(".jpg")]) + 1

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(person_dir, f"{timestamp}_{count}.jpg")
            image.save(file_path)

            return {
                "status": "success",
                "person_id": person_id,
                "confidence": confidence,
                "file_path": file_path,
            }

        return {"status": "fail", "reason": "low confidence"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/retrain/")
def retrain_model():
    train_model.train_model()
    return {"status": "success", "message": "Model retrained successfully."}
