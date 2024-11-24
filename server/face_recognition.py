import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features

        # Đọc các lớp từ file classes.txt
        with open('classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Thay đổi lớp phân loại
        self.model.fc = nn.Linear(num_features, len(self.classes))
        self.model.load_state_dict(torch.load('face_recognition_model.pth'))
        self.model.eval()

        # Chuẩn bị các phép biến đổi cho hình ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: np.ndarray):
        pil_image = Image.fromarray(image)
        face_tensor = self.transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(face_tensor)
            _, predicted = torch.max(output.data, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0]
        
        person_id = self.classes[predicted.item()]
        return person_id, confidence[predicted.item()].item()
