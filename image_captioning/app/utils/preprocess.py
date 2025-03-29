from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image, model):
    """
    Preprocess a PIL image and extract features using the provided ResNet model.
    """
    # Resize the image to match ResNet50's expected input size
    image = image.resize((224, 224))
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Remove alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    # Expand dimensions to match the model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image using ResNet50's preprocessing function
    img_array = preprocess_input(img_array)
    
    # Extract features with the model; expected output shape is (1, 2048)
    features = model.predict(img_array, verbose=0)
    
    # Return a flattened feature vector
    return features.flatten()
