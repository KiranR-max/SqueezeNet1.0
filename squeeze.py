import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2

# Load pre-trained SqueezeNet 1.1
model = torchvision.models.squeezenet1_1(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify a single frame
def classify_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make sure you have a GPU available, and if yes, move the model and input tensor to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    input_tensor = input_tensor.to(device)

    # Perform the inference
    with torch.no_grad():
        output = model(input_tensor)

    # Process the output probabilities and find the predicted class
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probability, top_class = torch.topk(probabilities, k=1)

    # Load ImageNet labels (1000 classes)
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    LABELS_PATH = "imagenet-simple-labels.json"
    import requests
    import json

    response = requests.get(LABELS_URL)
    with open(LABELS_PATH, 'wb') as f:
        f.write(response.content)

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    predicted_label = labels[top_class.item()]
    return predicted_label, top_probability.item()

# Classify frames from a video file
def classify_from_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Classify the current frame
        predicted_label, probability = classify_frame(frame)

        # Display the result on the frame
        cv2.putText(frame, f"{predicted_label} ({probability:.2f})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Classify frames from a live video stream
def classify_from_live_stream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Classify the current frame
        predicted_label, probability = classify_frame(frame)

        # Display the result on the frame
        cv2.putText(frame, f"{predicted_label} ({probability:.2f})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Image Classification")
    print("2. Video Classification from an mp4 file")
    print("3. Live Stream Classification")
    
    choice = int(input("Enter the number corresponding to your choice: "))

    if choice == 1:
        # Replace 'path/to/your/image.jpg' with the path to your image
        image_path = input("Enter the path to the image: ")
        image = Image.open(image_path)
        predicted_label, probability = classify_frame(image)
        print(f"Predicted class: {predicted_label} with probability: {probability:.2f}")
    elif choice == 2:
        # Replace 'path/to/your/video.mp4' with the path to your mp4 file
        video_path = input("Enter the path to the video: ")
        classify_from_video_file(video_path)
    elif choice == 3:
        classify_from_live_stream()
    else:
        print("Invalid choice. Please choose a valid option (1, 2, or 3).")
