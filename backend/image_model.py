from clip_model import get_similarity

async def predict_image(file):
    contents = await file.read()
    score = get_similarity("fake news", contents)
    return float(score)