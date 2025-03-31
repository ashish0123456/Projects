import os
import torch
import yaml
from app.models.transformer import TransformerModel, train_model, evaluate_model
from data_prep import train_loader, valid_loader, word_to_index

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
model_path = config["model"]["path"]
vocab_size = len(word_to_index) + 1 

# Instantiate and train the model
model = TransformerModel(vocab_size=vocab_size, max_len=config["training"]["max_len"], d_model=config["training"]["d_model"])
train_model(model, train_loader, epochs=config["training"]["epochs"])
evaluate_model(model, valid_loader)

# Save the model weights
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")