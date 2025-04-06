import os
import yaml
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from app.models.model import ImageCaptioning
from app.utils.inference import generate_caption
from app.utils.preprocess import preprocess_image
from app.utils.save_load import load_tokenizer, load_metadata
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50

config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load tokenizer and metadata
tokenizer = load_tokenizer(config['tokenizer']['path'])
max_length = load_metadata(config['metadata']['path'])['max_length']

# Load ResNet model
resnet_model = ResNet50(weights='imagenet')
resnet_model = Model(resnet_model.input, resnet_model.layers[-2].output)

# Load trained model
model_path = config['model']['path']
captioning_model = ImageCaptioning(len(tokenizer.word_index) + 1, max_length)
captioning_model.load_trained_model(model_path)

router = APIRouter()

@router.post('/predict/')
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image_feature = preprocess_image(image, resnet_model)

    caption = generate_caption(captioning_model.model, image_feature, tokenizer, max_length)
    return {"caption": caption}