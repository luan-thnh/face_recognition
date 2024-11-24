from fastapi import FastAPI, File, UploadFile
from mangum import Mangum
import numpy as np
import os
from PIL import Image
from datetime import datetime
from face_recognition import FaceRecognition
import train_model

# Initialize the FastAPI app
app = FastAPI()

# Initialize the FaceRecognition model
face_recognition = FaceRecognition(model_path='../face_recognition.py', classes_path="../classes.txt")

# CORS configuration (if needed)
@app.middleware("http")
async def add_cors(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello, Welcome to Checkify API"}

# Check-in endpoint to upload and predict using the face recognition model
@app.post("/checkin/")
async def checkin(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        person_id, confidence = face_recognition.predict(image_np)

        if confidence > 0.7:
            dataset_dir = "../dataset"
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

# Retrain model endpoint
@app.post("/retrain/")
def retrain_model():
    train_model.train_model()
    return {"status": "success", "message": "Model retrained successfully."}

# Wrap the FastAPI app with Mangum to handle AWS Lambda events
handler = Mangum(app)
