import os
import cv2


def init_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("Error: Could not open camera.")
    return cam


def collect_data():
    cam = init_camera()
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    face_id = input('\nEnter Face ID <return> ==>  ')
    save_path = os.path.join('datasets', face_id)

    os.makedirs(save_path, exist_ok=True)

    print('\n[INFO] Starting camera ...')
    count = 0

    try:
        while count < 100:
            ret, img = cam.read()
            if not ret:
                raise Exception("Error: Could not read frame.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                face_color = cv2.resize(img[y:y + h, x:x + w], (96, 96))
                cv2.imwrite(os.path.join(
                    save_path, f"{count}.jpg"), face_color)
                cv2.imshow('image', img)

            if cv2.waitKey(100) & 0xff == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Data collection interrupted by user.")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print(f'\nData collection complete. Collected {
              count} images in folder: {save_path}')


if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)
    collect_data()
