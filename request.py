import requests

# Assuming `file_path` is the path to the image file you want to upload
file_path = "IMG20240310135625.jpg"

# Open the image file in binary mode and read its contents
with open(file_path, "rb") as file:
    img = file.read()

# Create a dictionary to hold the form data
files = {"file": img}

# Send a POST request to the `/predict/` endpoint with the image file as part of form data
response = requests.post("https://aushdhi-ai.onrender.com/predict/", files=files)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the prediction from the response JSON
    prediction = response.json()["prediction"]
    print("Prediction:", prediction)
else:
    print("Error:", response.text)
