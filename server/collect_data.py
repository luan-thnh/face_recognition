import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

def collect_data(name, num_images=100):
    save_dir = os.path.join('dataset', name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                face = frame_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    face_pil = Image.fromarray(face)
                    face_pil.save(os.path.join(save_dir, f'{count}.jpg'))
                    count += 1
                    print(f'Saved image {count}/{num_images}')
        
        cv2.imshow('Collecting Face Data', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection completed. {count} images saved.")

