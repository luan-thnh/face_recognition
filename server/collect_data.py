import cv2
import os
from PIL import Image
import numpy as np


def collect_data():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    person_id = input("Nhập ID/tên người dùng: ")

    dataset_dir = 'dataset'
    person_dir = os.path.join(dataset_dir, str(person_id))
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face_img = frame[y:y+h, x:x+w]

            face_img = cv2.resize(face_img, (160, 160))

            img_path = os.path.join(person_dir, f'{count}.jpg')
            cv2.imwrite(img_path, face_img)
            count += 1

        cv2.putText(frame, f'Captured: {count}/100', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Collecting Data', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập xong {count} ảnh cho {person_id}")


if __name__ == "__main__":
    collect_data()
