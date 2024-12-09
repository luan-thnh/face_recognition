import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def load_trained_samples(file_path):
    """Load the list of trained samples."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return set(f.read().splitlines())
    return set()


def save_trained_samples(file_path, trained_samples):
    """Save the list of trained samples."""
    with open(file_path, 'w') as f:
        for sample in trained_samples:
            f.write(sample + '\n')


def get_samples(dataset_dir, trained_samples):
    """Get all samples and filter out trained ones."""
    all_samples = []
    class_to_idx = {}
    classes = os.listdir(dataset_dir)
    for idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls_name)
        if os.path.isdir(cls_dir):
            class_to_idx[cls_name] = idx
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path not in trained_samples:
                    all_samples.append((img_path, idx))
    return all_samples, class_to_idx


def plot_metrics(epochs, metrics, metric_name, color):
    """Plot training metrics."""
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics, label=metric_name, color=color)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.savefig(f'images/{metric_name.lower()}_epochs.png')


def train_model():
    DATASET_DIR = 'dataset'
    TRAINED_SAMPLES_FILE = 'trained_samples.txt'

    # Load trained samples and prepare new data
    trained_samples = load_trained_samples(TRAINED_SAMPLES_FILE)
    new_samples, class_to_idx = get_samples(DATASET_DIR, trained_samples)

    if not new_samples:
        print("No new samples to train.")
        return

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Data loader
    dataset = FaceDataset(new_samples, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model setup
    model = InceptionResnetV1(pretrained='vggface2',
                              classify=True, num_classes=len(class_to_idx))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 15
    metrics = {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Metrics calculation
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(
            all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds,
                              average='weighted', zero_division=0)

        metrics['accuracy'].append(accuracy)
        metrics['f1_score'].append(f1)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Save model and update trained samples
    torch.save(model.state_dict(), 'arcface_recognition_model.pth')
    trained_samples.update([sample[0] for sample in new_samples])
    save_trained_samples(TRAINED_SAMPLES_FILE, trained_samples)

    # Plot metrics
    epochs = range(1, num_epochs + 1)
    plot_metrics(epochs, metrics['accuracy'], 'Accuracy', 'blue')
    plot_metrics(epochs, metrics['f1_score'], 'F1 Score', 'orange')
    plot_metrics(epochs, metrics['precision'], 'Precision', 'green')
    plot_metrics(epochs, metrics['recall'], 'Recall', 'red')


if __name__ == "__main__":
    os.makedirs('images', exist_ok=True)  # Ensure images directory exists
    train_model()
