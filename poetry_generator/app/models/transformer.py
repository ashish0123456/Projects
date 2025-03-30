import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Positional Encoding to inject order information into token embeddings
class PositionEncoding(nn.Module):
    def __init__(self, max_len=3000, d_model=6):
        super().__init__()
        
        # Create a matrix of shape (max_len, d_model) initialized to zero
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector of positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the scaling factors for sine and cosine functions
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index/d_model)
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Store positional encodings as a buffer (not a trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, word_embedding):
        """
        Add positional encoding to word embeddings
        """
        return word_embedding + self.pe[:word_embedding.size(1), :]

# Self-Attention mechanism
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        # Linear layers for computing Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_input, k_input, v_input, mask=None):
        """
        Compute scaled dot-product attention
        """
        q = self.W_q(q_input)
        k = self.W_k(k_input)
        v = self.W_v(v_input)

        # Compute similarity scores (QK^T)
        similarity = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

        # Apply mask (prevent cheating by not looking ahead)
        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, v)
        return attention_output

# Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=3000, max_len=3000, d_model=6):
        super().__init__()
        
        torch.manual_seed(42)
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionEncoding(max_len=max_len, d_model=d_model)
        
        # Self-attention layer
        self.self_attention = Attention(d_model)
        
        # Fully connected layer (final output projection)
        self.fc_layer = nn.Linear(d_model, vocab_size)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_tokens):
        """
        Forward pass through the Transformer model
        """
        # Embed input tokens
        embeddings = self.embedding(input_tokens)
        
        # Add positional encoding
        position_encoded = self.pos_encoding(embeddings)
        
        # Create a lower triangular mask for autoregressive processing
        seq_len = input_tokens.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_tokens.device)
        
        # Apply self-attention with the mask
        attention_output = self.self_attention(position_encoded, position_encoded, position_encoded, mask=mask)
        
        # Add residual connection
        residual_output = attention_output + position_encoded
        
        # Final output projection
        output_logits = self.fc_layer(residual_output)
        
        return output_logits
    
    def configure_optimizer(self):
        """
        Define the optimizer for training
        """
        return Adam(self.parameters(), lr=0.01)
    

# Function for training the model
def train_model(model, train_loader, epochs=10):
    optimizer = model.configure_optimizer()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Function for evaluating the model
def evaluate_model(model, valid_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = model.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
    
    print(f"Validation Loss: {total_loss/len(valid_loader):.4f}")