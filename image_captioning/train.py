import os
from data_preprocess import train_dataset, val_dataset, data_ingestor, features_dict
from app.models.model import ImageCaptioning
import yaml
import tensorflow as tf

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

epochs = config['training']['epochs']
model_path = config['model']['path']

tokenizer = data_ingestor.tokenizer
max_length = data_ingestor.max_length

# Create the model instance
model_instance = ImageCaptioning(len(tokenizer.word_index) + 1, max_length)
model = model_instance.model

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(train_dataset, epochs=epochs, validation_data = val_dataset)

# Save the trained model
model.save(model_path)
