from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from model_arc import model

app = FastAPI()
# Define the classes
classes = ['Elichi', 'Neem', 'Palak', 'Tulsi', 'mint', 'Vitaceae', 'tecoma', 'Fabaceae', 'Putranjiva', 'AavniPan', 'papilionaceae', 'Piperaceae', 'chaulmugra', 'iripa', 'bhedeli', 'Polygonaceae']
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'plant-model.pth'
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
from fastapi.middleware.cors import CORSMiddleware
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to (256, 256)
    transforms.ToTensor()  # Convert images to tensors
])


# Define the endpoint to handle image upload
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    return {"prediction": predicted_class}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
