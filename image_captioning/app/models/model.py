import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, LSTM, Embedding, Dense, Add
from tensorflow.keras.models import Model

class ImageCaptioning:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = self.build_model()

    def build_model(self):
        image_input = Input(shape=(2048,))
        fe1 = Dropout(0.5)(image_input)
        fe2 = Dense(256, activation='relu')(fe1)

        text_input = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(text_input)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = Add()([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[image_input, text_input], outputs=outputs)
        return model

    def load_trained_model(self, model_path):
        self.model.load_weights(model_path)
