from data_preprocess import X1, X2, y, data_ingestor, features_dict
from app.models.model import ImageCaptioning

tokenizer = data_ingestor.tokenizer
max_length = data_ingestor.max_length

# Create the model instance
model_instance = ImageCaptioning(len(tokenizer.word_index) + 1, max_length)
model = model_instance.model

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Prepare training data
X1, X2, y = data_ingestor.prepare_training_data(features_dict)

# Train the model
epochs = 10
batch_size = 64
model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save the trained model
model.save('image_captioning_model.h5')
