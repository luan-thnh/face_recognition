import cv2
import os
from datetime import datetime

def collect_face_data(name, user_id, num_samples=100):
    # Tạo thư mục để lưu ảnh
    folder_name = f"{user_id}_{name}"
    dataset_path = f'dataset/{folder_name}'
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            
            # Lưu ảnh khuôn mặt
            if count < num_samples:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'{dataset_path}/{timestamp}_{count}.jpg', face)
                count += 1
        
        cv2.imshow('Collecting Face Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Nhập tên người dùng: ")
    user_id = input("Nhập ID người dùng: ")
    collect_face_data(name, user_id)
