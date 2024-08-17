import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

# Define the path to your image (relative path from the script location)
image_path = "imagesm.jpg"

# Load the image
img = Image.open(image_path)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Perform the inference
with torch.no_grad():
    out = model(batch_t)

# Load the labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Get the top 5 predictions
_, indices = torch.topk(out, 5)
percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
top5 = [(labels[idx], percentages[idx].item()) for idx in indices[0]]

# Print the results
for label, percentage in top5:
    print(f"{label}: {percentage:.2f}%")
