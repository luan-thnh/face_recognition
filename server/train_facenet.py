import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import classification_report, accuracy_score
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    all_accuracy = []
    all_preds = []
    all_labels = []
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
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
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        
        all_accuracy.append(accuracy)

    # Save the model after training
    torch.save(model.state_dict(), 'arcface_recognition_model.pth')

    with open('classes.txt', 'w') as f:
        for cls in dataset.classes:
            f.write(f'{cls}\n')

    # Display the classification report
print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    # Plot the accuracy over epochs
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_accuracy, label='Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

    # Overall accuracy after training
    final_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Final Accuracy after Training: {final_accuracy:.4f}')

if __name__ == "__main__":
    train_model()