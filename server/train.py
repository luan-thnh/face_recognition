import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model():

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    dataset = FaceDataset('dataset', transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(dataset.classes))
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    all_accuracy = []
    all_f1_score = []
    all_precision = []
    all_recall = []


    num_epochs = 15  # Thay đổi số lượng epochs thành 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
    
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        
    
        all_accuracy.append(accuracy)
        all_f1_score.append(f1)
        all_precision.append(precision)
        all_recall.append(recall)


    torch.save(model.state_dict(), 'arcface_recognition_model.pth')
    

    with open('classes.txt', 'w') as f:
        for cls in dataset.classes:
            f.write(f'{cls}\n')
    

    epochs = range(1, num_epochs + 1)


    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_accuracy, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()


    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_f1_score, label='F1-Score', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score over Epochs')
    plt.legend()


    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_precision, label='Precision', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision over Epochs')
    plt.legend()


    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_recall, label='Recall', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')
    plt.legend()

if __name__ == "__main__":
    train_model()