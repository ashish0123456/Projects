from fastapi import APIRouter, UploadFile, File
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from app.models.model import ImageCaptioning
from app.utils.inference import generate_caption
from app.utils.preprocess import preprocess_image
from data_preprocess import data_ingestor, resnet_model

tokenizer = data_ingestor.tokenizer
max_length = data_ingestor.max_length

router = APIRouter()

# Load trained model
model_path = "image_captioning_model.h5"
captioning_model = ImageCaptioning(len(tokenizer.word_index) + 1, max_length)
captioning_model.load_trained_model(model_path)

@router.post('/predict/')
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image_feature = preprocess_image(image, resnet_model)

    caption = generate_caption(captioning_model.model, image_feature, tokenizer, max_length)
    return {"caption": caption}