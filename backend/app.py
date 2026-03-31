from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from text_model import get_text_score
from image_model import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Fake Content Detection API Running"}

@app.post("/predict")
async def predict(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    text_score = None
    image_score = None

    if text:
        text_score = get_text_score(text)

    if file:
        image_score = await predict_image(file)

    # FUSION
    scores = []
    if text_score is not None:
        scores.append(text_score)
    if image_score is not None:
        scores.append(image_score)

    final_score = sum(scores) / len(scores) if scores else 0

    # DECISION
    if final_score > 0.7:
        decision = "Fake"
    elif final_score < 0.3:
        decision = "Real"
    else:
        decision = "Uncertain"

    # CONFIDENCE
    confidence = abs(final_score - 0.5) * 2

    return {
        "text_score": text_score,
        "image_score": image_score,
        "final_score": round(final_score, 3),
        "decision": decision,
        "confidence": round(confidence, 2),
        "reason": generate_reason(text_score, image_score)
    }


def generate_reason(text_score, image_score):
    reasons = []

    if text_score is not None:
        if text_score > 0.7:
            reasons.append("Text strongly indicates fake content")
        elif text_score < 0.3:
            reasons.append("Text appears reliable")
        else:
            reasons.append("Text is ambiguous")

    if image_score is not None:
        if image_score > 0.7:
            reasons.append("Image-text mismatch detected")
        else:
            reasons.append("Image consistent with text")

    return ", ".join(reasons)