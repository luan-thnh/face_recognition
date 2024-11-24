import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import io

class FaceRecognition:
    def __init__(self, model_path: str = None, classes_path: str = None):
        """
        Khởi tạo lớp FaceRecognition.

        Args:
        - model_path (str or None): Đường dẫn tới file chứa trạng thái mô hình (.pth).
        - classes_path (str or None): Đường dẫn tới file chứa danh sách lớp (classes.txt).
        """
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features

        # Đọc danh sách các lớp từ file classes.txt (nếu có)
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = []

        # Thay thế lớp đầu ra để phù hợp với số lượng lớp
        self.model.fc = nn.Linear(num_features, len(self.classes))

        # Nếu có model_path, ta sẽ tải model từ file
        if model_path:
            self.load(model_path)

        # Cấu hình transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh về 224x224
            transforms.ToTensor(),  # Chuyển ảnh sang tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
        ])

    def load(self, model_path):
        """
        Tải trọng số của mô hình từ file hoặc byte stream.

        Args:
        - model_path (str or io.BytesIO): Đường dẫn tới file mô hình hoặc byte stream mô hình.
        """
        if isinstance(model_path, str):  # Nếu là đường dẫn đến file
            self.model.load_state_dict(torch.load(model_path))
        elif isinstance(model_path, io.BytesIO):  # Nếu là byte stream (dành cho trường hợp download từ GitHub)
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, image: np.ndarray):
        """
        Dự đoán lớp của một ảnh khuôn mặt.

        Args:
        - image (np.ndarray): Ảnh khuôn mặt dưới dạng numpy array (RGB).

        Returns:
        - person_id (str): ID của người được nhận diện.
        - confidence (float): Độ tin cậy của dự đoán.
        """
        # Chuyển numpy array sang PIL Image
        pil_image = Image.fromarray(image)
        
        # Áp dụng transform và thêm batch dimension
        face_tensor = self.transform(pil_image).unsqueeze(0)

        # Tắt gradient trong quá trình dự đoán
        with torch.no_grad():
            output = self.model(face_tensor)
            
            # Tìm lớp dự đoán có xác suất cao nhất
            _, predicted = torch.max(output.data, 1)
            
            # Tính toán độ tin cậy sử dụng softmax
            confidence = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Lấy ID của người và độ tin cậy
        person_id = self.classes[predicted.item()]
        return person_id, confidence[predicted.item()].item()
