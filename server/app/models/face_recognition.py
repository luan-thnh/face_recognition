import torch
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FaceRecognitionModel:
    def __init__(self, model_path, classes):
        self.model = InceptionResnetV1(
            pretrained="vggface2", classify=True, num_classes=len(classes))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

    def predict(self, face_tensor):
        with torch.no_grad():
            output = self.model(face_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(prob, 1)
            return confidence.item(), prediction.item()
