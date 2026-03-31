from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_similarity(text, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits = outputs.logits_per_image
    score = logits.softmax(dim=1).detach().numpy()[0][0]

    return float(score)