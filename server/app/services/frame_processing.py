import cv2
from PIL import Image
from models.face_recognition import FaceRecognitionModel
from models.student_data import StudentData
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import os

classes = []
try:
    with open("../../classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    classes = ["Unknown"]


def process_video(file):
    # Đọc file Excel
    df = StudentData.get_students_data()
    video_path = f"temp_video_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to process video")

    predictions = []
    confidences = {}

    face_model = FaceRecognitionModel(
        model_path="arcface_recognition_model.pth", classes=classes)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_frame = {}
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
                future = executor.submit(process_frame, frame, face_model)
                future_to_frame[future] = frame_count

            frame_count += 1

        for future in as_completed(future_to_frame):
            result = future.result()
            if result:
                predicted_class, conf_value = result
                predictions.append(predicted_class)
                if predicted_class not in confidences or conf_value > confidences.get(predicted_class, 0):
                    confidences[predicted_class] = conf_value

    cap.release()
    os.remove(video_path)

    if not predictions:
        return {"message": "No faces detected in the video", "prediction": None, "confidence": None}

    most_common_class = Counter(predictions).most_common(1)[0][0]
    highest_confidence = confidences[most_common_class]

    student_row = df[df['ID'] == int(most_common_class)]
    if not student_row.empty:
        student_name = student_row['Họ và tên'].values[0]
        current_date = datetime.now().strftime("%d/%m/%Y")
        checkin_column = current_date
        df = StudentData.update_checkin(
            df, int(most_common_class), checkin_column)
        df.to_excel("excel_files/ST21A2A.xlsx", index=False)

        return {
            "message": "Success",
            "prediction": most_common_class,
            "student_name": student_name,
            "confidence": round(highest_confidence, 2),
        }
    else:
        return {"message": "Student not found", "prediction": most_common_class}
