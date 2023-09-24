import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Load pre-trained SqueezeNet 1.1
model = torchvision.models.squeezenet1_1(pretrained=True)
model.eval()  # Set the model to evaluation mode

# You can print the model architecture if you want to see the layers
# print(model)

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
image_path = "bird.jpg"
image = Image.open(image_path)
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
print(f"Predicted class: {predicted_label} with probability: {top_probability.item()}")
