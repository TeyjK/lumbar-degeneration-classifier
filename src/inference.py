import torch
from torchvision import transforms
from PIL import Image, ImageFile
from io import BytesIO
from src.model import load_trained_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


def predict(image_bytes: BytesIO, model_type: str):
    try:
        image = Image.open(image_bytes).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return {
            "Normal/Mild": 0.0,
            "Moderate": 0.0,
            "Severe": 0.0,
            "Error": "Invalid or corrupted image file.",
        }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)
    model = load_trained_model(model_type)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)

    return {
        "Normal/Mild": round(probs[0].item(), 4),
        "Moderate": round(probs[1].item(), 4),
        "Severe": round(probs[2].item(), 4),
    }
