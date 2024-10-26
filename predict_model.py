import cv2
import numpy as np
from tensorflow.keras.models import load_model


def predict_face(model):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print('\n[INFO] Starting camera for face recognition...')

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                raise Exception("Error: Could not read frame.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (100, 100))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)

                # Dự đoán nhãn và lấy nhãn có xác suất cao nhất cho mỗi khuôn mặt
                prediction = model.predict(face)
                label = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, f'ID: {label} ({confidence:.2f})', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow('Face Recognition', img)

            # Kiểm tra phím ESC và thoát khỏi vòng lặp nếu được nhấn
            if cv2.waitKey(1) & 0xFF == 27:
                print("\n[INFO] ESC pressed. Exiting face recognition.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Face recognition interrupted by user.")
    finally:
        # Đảm bảo tài nguyên được giải phóng
        cam.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Camera and windows released successfully.")


if __name__ == "__main__":
    model = load_model('./model/face_recognition_model.keras')
    predict_face(model)
