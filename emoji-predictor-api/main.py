import base64
import io
import os  # Needed for checking file existence and getting env vars
import requests  # Needed for downloading the model
from typing import List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as transforms

# Import only the loading function as the class itself isn't directly used here
from model import load_emoji_model

# --- Configuration ---
MODEL_PATH = "emoji_model.ckpt"  # Local path where the model will be stored/loaded from
MODEL_URL = os.getenv("MODEL_URL")  # Get download URL from environment variable
NUM_CLASSES = 18  # Make sure this matches your trained model
INPUT_SIZE = (128, 128)  # Expected input size for the model (adjust if needed)

# Define the expected emoji classes/labels (replace with your actual labels)
# The order MUST match the output indices of your model (from FlafyDev/emoji-drawings)
EMOJI_CLASSES = [
    "😁",  # 0: beaming-face
    "☁️",  # 1: cloud
    "😵‍💫",  # 2: face-spiral
    "😳",  # 3: flushed-face
    "😬",  # 4: grimacing-face
    "😃",  # 5: grinning-face
    "😆",  # 6: grinning-squinting
    "❤️",  # 7: heart
    "😡",  # 8: pouting-face
    "🤨",  # 9: raised-eyebrow
    "😌",  # 10: relieved-face
    "😋",  # 11: savoring-food
    "😍",  # 12: smiling-heart
    "😈",  # 13: smiling-horns
    "😎",  # 14: smiling-sunglasses
    "🥲",  # 15: smiling-tear
    "😏",  # 16: smirking-face
    "😂"   # 17: tears-of-joy
]

# --- App Initialization ---
app = FastAPI()

# Allow CORS for frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# --- Model Loading (with download logic) ---

def download_model(url: str, destination: str):
    print(f"Downloading model from {url} to {destination}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model from URL. Error: {e}")
    except IOError as e:
        print(f"Error saving model file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save downloaded model file. Error: {e}")

model = None
# Check if model needs to be downloaded
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found locally at {MODEL_PATH}.")
    if MODEL_URL:
        download_model(MODEL_URL, MODEL_PATH)
    else:
        print("MODEL_URL environment variable not set. Cannot download model.")
        # Optional: raise an error or exit if model is mandatory
        # raise RuntimeError("Model file missing and MODEL_URL not set.")

# Load the model if it exists locally (either pre-existing or downloaded)
if os.path.exists(MODEL_PATH):
    try:
        model = load_emoji_model(MODEL_PATH, NUM_CLASSES)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        model = None  # Ensure model is None if loading fails after download/check
else:
    print(f"Model could not be loaded. File still not found at {MODEL_PATH}.")

# --- Data Validation Models ---
class PredictionRequest(BaseModel):
    image_data_url: str  # Expecting base64 data URL from frontend

class PredictionResponse(BaseModel):
    predicted_emoji: str
    confidence: float | None = None  # Optional confidence score

# --- Image Preprocessing ---
def preprocess_image(image_data_url: str) -> torch.Tensor:
    """Converts base64 image data URL to a PyTorch tensor, matching training transforms."""
    try:
        # Extract base64 data from data URL
        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        
        # Open image using Pillow (don't force RGB here, Grayscale handles it)
        image = Image.open(io.BytesIO(image_data))
        
        # Define transformations (matching training in NUM-Lab1/data_module.py)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale then duplicate channels
            transforms.Resize(INPUT_SIZE),  # Resize to 128x128
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            # Normalize using ImageNet stats (as used in training)
            # --- TEMPORARILY REMOVED FOR TESTING --- Maybe unsuitable for drawings?
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        
        # Apply transformations and add batch dimension
        tensor = transform(image).unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data or preprocessing failed")

# --- WebSocket Connection Management ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"New WebSocket connection: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket connection closed: {websocket.client}")

    async def broadcast(self, message: str):
        print(f"Broadcasting message: {message}")
        # Use json.dumps to send structured data if needed, here just the emoji string
        for connection in self.active_connections:
            try:
                await connection.send_text(message)  # Send the raw emoji string
            except Exception as e:
                print(f"Error sending to websocket {connection.client}: {e}")
                # Optionally remove broken connections here

manager = ConnectionManager()

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Emoji Predictor API is running!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, assign to _data to indicate intentional non-use
            _data = await websocket.receive_text()
            pass  # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error for {websocket.client}: {e}")
        manager.disconnect(websocket)  # Ensure disconnect on other errors

@app.post("/predict", response_model=PredictionResponse)
async def predict_emoji(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess the image data URL
        image_tensor = preprocess_image(request.image_data_url)
        
        # Perform prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class_index = predicted_idx.item()
        
        # Get the corresponding emoji
        if 0 <= predicted_class_index < len(EMOJI_CLASSES):
            predicted_emoji = EMOJI_CLASSES[predicted_class_index]
        else:
            predicted_emoji = "❓"  # Default if index is out of bounds
            print(f"Warning: Predicted index {predicted_class_index} out of bounds for EMOJI_CLASSES")

        # Broadcast the predicted emoji to all connected WebSocket clients
        await manager.broadcast(predicted_emoji)

        return PredictionResponse(
            predicted_emoji=predicted_emoji,
            confidence=confidence.item()
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions from preprocessing
        raise http_exc
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# --- Run with Uvicorn (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # Usually, you run uvicorn from the terminal:
    # uvicorn main:app --reload --host 0.0.0.0 --port 8000
    # This block allows running with `python main.py` for simple testing if needed
    print("Starting Uvicorn server directly (use 'uvicorn main:app --reload' for development)")
    uvicorn.run(app, host="127.0.0.1", port=8000) 