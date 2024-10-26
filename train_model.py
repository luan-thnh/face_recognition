import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns


def load_images_from_dataset(dataset_path):
    face_data, labels = [], []
    for label_name in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_name)
        if os.path.isdir(label_path):
            label = int(label_name)
            for file in os.listdir(label_path):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (100, 100))
                face_data.append(gray)
                labels.append(label)
    return np.array(face_data), np.array(labels)


def preprocess_data(face_data, labels):
    face_data = face_data.astype('float32') / 255.0
    face_data = np.expand_dims(face_data, axis=-1)
    return train_test_split(face_data, labels, test_size=0.2, random_state=42)


def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_training_history(history, output_dir='images'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def compute_metrics(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    precision = precision_score(
        y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    return precision, recall, f1


def plot_roc_curve(y_test, y_pred, output_dir='images'):
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_dir='images'):
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(
        y_true_labels), yticklabels=np.unique(y_true_labels))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def train_model(dataset_path='datasets'):
    face_data, labels = load_images_from_dataset(dataset_path)
    X_train, X_test, y_train, y_test = preprocess_data(face_data, labels)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = create_model((100, 100, 1), y_train.shape[1])

    history = model.fit(X_train, y_train, epochs=30,
                        batch_size=32, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)

    precision, recall, f1 = compute_metrics(y_test, y_pred)
    print(f'Precision: {precision:.4f}, Recall: {
          recall:.4f}, F1 Score: {f1:.4f}')

    model.save('./model/face_recognition_model.keras')
    print("Model trained and saved.")

    plot_training_history(history)
    plot_roc_curve(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

    return history


if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    history = train_model('datasets')
