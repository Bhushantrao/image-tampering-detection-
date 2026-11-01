import torch
from torchvision import transforms
from PIL import Image

from .vit_model import ViTModel


# ------------------------------------
# Device
# ------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------
# Load Model
# ------------------------------------

model = ViTModel()

checkpoint = torch.load(
    "saved_models/best_vit_model.pth",
    map_location=device
)

# If checkpoint contains state_dict
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove "module." prefix if present
new_state_dict = {}

for key, value in checkpoint.items():

    if key.startswith("module."):
        key = key[7:]

    new_state_dict[key] = value

model.load_state_dict(
    new_state_dict,
    strict=False
)

model.to(device)

model.eval()


# ------------------------------------
# Image Transform
# ------------------------------------

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# ------------------------------------
# Class Labels
# Training Mapping:
# {'fake':0,'real':1}
# ------------------------------------

classes = [
    "Fake",
    "Original"
]


# ------------------------------------
# Prediction Function
# ------------------------------------

def predict_image(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(
            image
        )

        image = image.unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                1
            )

        result = classes[
            prediction.item()
        ]

        confidence = round(
            confidence.item() * 100,
            2
        )

        # -------------------------
        # DEBUG
        # -------------------------

        print("\n========== DEBUG ==========")

        print("Raw Output:")
        print(outputs)

        print("\nProbabilities:")
        print(probabilities)

        print("\nFake Probability:",
              round(probabilities[0][0].item() * 100, 2),
              "%")

        print("Original Probability:",
              round(probabilities[0][1].item() * 100, 2),
              "%")

        print("\nPrediction:", result)

        print("Confidence:", confidence)

        print("===========================\n")

        return result, confidence

    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return "Error", 0