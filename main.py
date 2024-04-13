import torch
from torchvision import transforms
from PIL import Image
from model_arc import model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to the saved model
PATH = 'plant-model.pth'
classes = ['Elichi',
 'Neem',
 'Palak',
 'Tulsi',
 'mint',
 'Vitaceae',
 'tecoma',
 'Fabaceae',
 'Putranjiva',
 'AavniPan',
 'papilionaceae',
 'Piperaceae',
 'chaulmugra',
 'iripa',
 'bhedeli',
 'Polygonaceae']
# Load the model
model = model  
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to (256, 256)
    transforms.ToTensor()  # Convert images to tensors
])

# Load and preprocess the image
img_path = 'IMG20240310135625.jpg'
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]

print(f"prediction : {predicted_class}")
