from PIL import Image
from typing import Dict

import bentoml

import torch

import torchvision.transforms as transforms

EMOJI_MAP = [
    "😁",
    "☁️",
    "😵‍💫",
    "😳",
    "😬",
    "😃",
    "😆",
    "❤️",
    "😡",
    "🤨",
    "😌",
    "😋",
    "😍",
    "😈",
    "😎",
    "🥲",
    "😏",
    "😂",
]


@bentoml.service(
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["http://localhost:3000"],
            "access_control_allow_credentials": True,
            "access_control_allow_methods": ["*"],
            "access_control_allow_headers": ["*"],
        }
    }
)
class EmojiClassifierService:
    def __init__(self):
        """
        Load the model into memory for the service.
        Define the inference transformation pipeline.
        """
        self.model = bentoml.pytorch_lightning.load_model("emoji_classifier:latest")
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @bentoml.api
    async def predict(self, image_input: Image.Image) -> Dict:
        """
        Predicts the emoji class for a given image.
        Applies the same preprocessing as used during training/validation.
        """
        rgb_image = image_input.convert("RGB")

        image_tensor = self.transform(rgb_image)

        image_tensor = image_tensor.unsqueeze(0)

        predicted_emoji = "❓"
        predicted_class = -1
        probabilities = []

        try:
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities_tensor = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities_tensor, dim=1).item()
                probabilities = probabilities_tensor.squeeze().tolist()

                if 0 <= predicted_class < len(EMOJI_MAP):
                    predicted_emoji = EMOJI_MAP[predicted_class]
                else:
                    print(
                        f"Warning: Predicted class index {predicted_class} is out of bounds for EMOJI_MAP."
                    )

        except Exception as e:
            print(f"Error during inference or emoji lookup: {e}")

        return {
            "predicted_class": predicted_class,
            "predicted_emoji": predicted_emoji,  # Add the emoji character
            "probabilities": probabilities,
        }
