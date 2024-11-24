from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from face_recognition import FaceRecognition 
from PIL import Image
import numpy as np

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
face_recognition = FaceRecognition()

@app.post("/checkin/")
async def checkin(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = np.array(image)

    person_id, confidence = face_recognition.predict(image)

    if confidence > 0.7:
        return {"status": "success", "person_id": person_id, "confidence": confidence}
    
    return {"status": "fail", "reason": "low confidence"}
