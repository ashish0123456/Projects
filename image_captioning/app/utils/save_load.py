import pickle
import yaml

def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_metadata(metadata, path):
    with open(path, 'w') as f:
        yaml.safe_dump(metadata, f)

def load_metadata(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)