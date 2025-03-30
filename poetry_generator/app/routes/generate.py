from fastapi import APIRouter, Query
import torch
import yaml
from app.models.transformer import TransformerModel
from data_prep import word_to_index, index_to_word

router = APIRouter()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["path"]
vocab_size = len(word_to_index) + 1

# Instantiate the model
model = TransformerModel(vocab_size=vocab_size, max_len=300, d_model=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

@router.get('/generate')
def generate_text(prompt: str = Query(..., description='Input text prompt'),
                  max_len: int = Query(50, description='Maximum length of generate response')):
    # Convet the prompt words to token indices
    tokens = [word_to_index.get(word, 0) for word in prompt.split()]
    input_tensor = torch.tensor(tokens).unsqueeze(0)

    response = input_tensor
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(response)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            response = torch.cat((response, next_token), dim=-1)
            if next_token.item() == 0:       # if padding token is generated, stop
                break

    # Convert generated tokens back to words
    generated_tokens = response.squeeze().tolist()
    generated_text = " ".join([index_to_word.get(token, "") for token in generated_tokens])
    return {"generated_text": generated_text}