import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

def predict(image_path=None, use_webcam=False):
    """
    Predict identity from image or webcam
    
    Args:
        image_path (str): Path to image file (if not using webcam)
        use_webcam (bool): Whether to use webcam for prediction
    """
    # Load class names
    try:
        with open('classes.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("classes.txt not found. Using default classes.")
        classes = ['Unknown']
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with correct number of classes
    try:
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(classes))
        
        # Load state dict with proper model configuration
        try:
            state_dict = torch.load('arcface_recognition_model.pth', map_location=device)
            
            # If the loaded object is a full model, extract its state dict
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
            
            # Handle potential state dict compatibility issues
            if 'logits.weight' in state_dict:
                # Remove classification layer from loaded state dict if needed
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logits.')}
            
            # Load the compatible parts of the state dict
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Initializing model with random weights")
        
        model = model.to(device)
        model.eval()
    except Exception as model_init_error:
        print(f"Failed to initialize model: {model_init_error}")
        print("Falling back to a simple classification approach")
        model = None
    
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Initialize transform
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            try:
                boxes, _ = mtcnn.detect(frame_rgb)
            except Exception as detect_error:
                print(f"Face detection error: {detect_error}")
                boxes = None
            
            if boxes is not None:
                for box in boxes:
                    # Extract face coordinates
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    # Extract face
                    face = frame_rgb[y1:y2, x1:x2]
                    if face.size > 0:
                        face_pil = Image.fromarray(face)
                        
                        # Transform and predict
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)
                        
                        try:
                            with torch.no_grad():
                                # Fallback prediction logic
                                if model is not None:
                                    output = model(face_tensor)
                                    
                                    # Ensure output is valid
                                    if output.numel() > 0:
                                        prob = torch.nn.functional.softmax(output, dim=1)
                                        confidence, prediction = torch.max(prob, 1)
                                        
                                        predicted_class = classes[prediction.item()]
                                        conf_value = confidence.item()
                                    else:
                                        predicted_class = 'Unknown'
                                        conf_value = 0.0
                                else:
                                    # Simple fallback if model is not loaded
                                    predicted_class = 'Unknown'
                                    conf_value = 0.0
                                
                                # Draw rectangle and text
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text = f'{predicted_class} ({conf_value:.2f})'
                                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        
                        except Exception as prediction_error:
                            print(f"Prediction error: {prediction_error}")
                            # Fallback visualization if prediction fails
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, 'Error', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        if image_path is None:
            print("Please provide an image path or set use_webcam=True")
            return
        
        # Load and process image (similar error handling as webcam)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if model is not None:
                    output = model(image_tensor)
                    
                    if output.numel() > 0:
                        prob = torch.nn.functional.softmax(output, dim=1)
                        confidence, prediction = torch.max(prob, 1)
                        
                        predicted_class = classes[prediction.item()]
                        conf_value = confidence.item()
                    else:
                        predicted_class = 'Unknown'
                        conf_value = 0.0
                else:
                    predicted_class = 'Unknown'
                    conf_value = 0.0
                
                print(f"Predicted class: {predicted_class}")
                print(f"Confidence: {conf_value:.2f}")
        
        except Exception as image_error:
            print(f"Image processing error: {image_error}")

if __name__ == "__main__":
    use_webcam = input("Use webcam? (y/n): ").lower() == 'y'
    if use_webcam:
        predict(use_webcam=True)
    else:
        image_path = input("Enter image path: ")
        predict(image_path=image_path)