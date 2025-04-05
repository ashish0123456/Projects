import os
import pandas as pd
import numpy as np
import string
import kagglehub
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import to_categorical


class DataIngestor:
    """
    Handles preprocessing of captions, feature extraction from images, and preparation of training data.
    """

    def __init__(self):
        self.captions_dict = {}
        self.tokenizer = Tokenizer()
        self.vocab_size = 0
        self.max_length = 0

    def preprocess_captions(self, df):
        """
        Converts captions to lowercase, removes punctuation, and adds start/end tokens.
        Stores them in a dictionary where keys are image filenames.
        """
        captions_dict = {}

        for _, row in df.iterrows():
            image_id = row['image']
            # convert the caption into lower case and remove the extra spaces
            caption = row['caption'].strip().lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            caption = f'startseq {caption} endseq'  # Add sequence tokens

            if image_id not in captions_dict:
                captions_dict[image_id] = []
            captions_dict[image_id].append(caption) # multiple captions per image

        self.captions_dict = captions_dict
        return captions_dict

    def tokenize_captions(self, captions):
        """
        Tokenizes text, builds vocabulary, and determines max caption length.
        """
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(caption.split()) for caption in captions)

    def extract_features(self, image_path, model):
        """
        Preprocesses an image and extracts features using ResNet50.
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        # Open the image
        img = Image.open(image_path)
        
        # Resize to match ResNet50 input
        img = img.resize((224,224))

        # Convert an image to numpy array
        img_array = np.array(img)

        # Remove alpha channel if present
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        # Expand dimensions to match expected input shape: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # preprocess image as the ResNet50 requirements
        img_array = preprocess_input(img_array)

        # Extract features, result is a (1, 2048) vector
        features = model.predict(img_array, verbose=1) 
        return features.flatten()  # Convert to 1D vector
    
    def prepare_training_data(self, features_dict):
        """
        Converts tokenized captions into sequences and pairs them with corresponding image features.
        """
        X1, X2, y = [], [], []

        for image_id, captions in self.captions_dict.items():
            if image_id not in features_dict:
                continue

            image_feature = features_dict[image_id]

            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=self.max_length, padding='post')[0]
                    X1.append(image_feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)


# ---------------------------------------
# Load and preprocess dataset
# ---------------------------------------
dataset_path = kagglehub.dataset_download('adityajn105/flickr8k')
caption_file_path = os.path.join(dataset_path, 'captions.txt')
df = pd.read_csv(caption_file_path)

data_ingestor = DataIngestor()
captions_dict = data_ingestor.preprocess_captions(df) # preprocess the captions

# Gather all captions for tokenization
all_captions = [caption for captions in captions_dict.values() for caption in captions]
data_ingestor.tokenize_captions(all_captions)

# -----------------------------------------------------
# Image preprocessing & feature extraction (encoder)
# -----------------------------------------------------
# Load the pretrained ResNet50 model with ImageNet weights
# Remove the final classification layer to obtain features
resnet_model = ResNet50(weights='imagenet')
# Use the second last layer's output
resnet_model = Model(resnet_model.input, resnet_model.layers[-2].output)

# Extract features
features_dict = {
    image_id: data_ingestor.extract_features(os.path.join(dataset_path, 'Images', image_id), resnet_model)
    for image_id in captions_dict.keys()
}

# -----------------------------------------------
# Prepare the training data (Decoder input)
# -----------------------------------------------
# For each caption, we will create input sequences that progressively build the caption.
# For example, if a caption is "startseq a dog runs endseq", then we create:
#    Input: "startseq"        Output: "a"
#    Input: "startseq a"      Output: "dog"
#    Input: "startseq a dog"  Output: "runs"
#    Input: "startseq a dog runs" Output: "endseq"

X1, X2, y = data_ingestor.prepare_training_data(features_dict)

batch_size = 64
total_size = len(X1)
split_index = int(0.8 * total_size)

train_dataset = tf.data.Dataset.from_tensor_slices(((X1[:split_index], X2[:split_index]), y[:split_index]))\
    .shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(((X1[split_index:], X2[split_index:]), y[split_index:]))\
    .batch(batch_size).prefetch(tf.data.AUTOTUNE)