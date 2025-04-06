import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_caption(model, image_feature, tokenizer, max_length):
    in_text = 'startseq'
    
    # Ensure image_feature has a batch dimension
    if image_feature.ndim == 1:
        image_feature = np.expand_dims(image_feature, axis=0)

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        yhat = model.predict([np.array(image_feature), sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if yhat == index:
                word = w
                break

        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()
