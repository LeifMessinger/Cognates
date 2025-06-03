import torch
import torch.nn as nn
import math

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerCognateModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(TransformerCognateModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)

        dropout = .2
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fixed: Use embedding_dim instead of hidden_dim for the input size
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),  # Changed from hidden_dim * 2 to embedding_dim * 2
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode_word(self, x):
        # Debug: Print input shape
        #print(f"Input shape: {x.shape}")
        
        # Ensure x is 2D: [batch_size, seq_len]
        if x.dim() > 2:
            # If input has extra dimensions, squeeze them out
            x = x.squeeze()
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dim if we squeezed too much
        elif x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        #print(f"After embedding shape: {x.shape}")
        
        x = x.transpose(0, 1)  # [seq_len, batch_size, embedding_dim] (required for transformer)
        #print(f"After transpose shape: {x.shape}")
        
        pos_encoded_x = self.pos_encoder(x)  # [seq_len, batch_size, embedding_dim]
        #print(f"After pos encoding shape: {pos_encoded_x.shape}")
        
        encoded = self.transformer_encoder(pos_encoded_x)  # [seq_len, batch_size, embedding_dim]
        #print(f"After transformer shape: {encoded.shape}")
        
        # Option 1: Use mean pooling to get a fixed-size representation
        return encoded.mean(dim=0)  # [batch_size, embedding_dim]
        
        # Option 2: Use the last token (uncomment if preferred)
        # return encoded[-1]  # [batch_size, embedding_dim]

    def forward(self, input1, input2):
        enc1 = self.encode_word(input1)  # [batch_size, embedding_dim]
        enc2 = self.encode_word(input2)  # [batch_size, embedding_dim]
        combined = torch.cat([enc1, enc2], dim=1)  # [batch_size, embedding_dim * 2]
        return self.fc(combined)  # [batch_size, 1]

class UnbatchedWrapper(nn.Module):
    """
    A wrapper that allows a batched model to accept single inputs without batch dimensions.
    
    This wrapper automatically adds batch dimensions to inputs and removes them from outputs,
    making it easier to use batched models for single predictions.
    
    Args:
        model: The underlying batched model to wrap
    
    Example:
        >>> batched_model = TransformerCognateModel(vocab_size)
        >>> unbatched_model = UnbatchedWrapper(batched_model)
        >>> 
        >>> # Now you can use single inputs without worrying about batch dimensions
        >>> word1 = torch.tensor([1, 2, 3, 4, 0])  # [seq_len] - no batch dimension
        >>> word2 = torch.tensor([5, 6, 7, 0, 0])  # [seq_len] - no batch dimension
        >>> output = unbatched_model(word1, word2)  # Returns scalar tensor
    """
    
    def __init__(self, model):
        super(UnbatchedWrapper, self).__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that handles adding/removing batch dimensions.
        
        Automatically adds batch dimension (unsqueeze(0)) to all tensor arguments,
        calls the underlying model, then removes the batch dimension from the output.
        """
        # Add batch dimension to all tensor arguments
        batched_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                batched_args.append(arg.unsqueeze(0))  # Add batch dimension
            else:
                batched_args.append(arg)
        
        # Add batch dimension to all tensor keyword arguments
        batched_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                batched_kwargs[key] = value.unsqueeze(0)  # Add batch dimension
            else:
                batched_kwargs[key] = value
        
        # Call the underlying model with batched inputs
        output = self.model(*batched_args, **batched_kwargs)
        
        # Remove batch dimension from output
        if isinstance(output, torch.Tensor):
            return output.squeeze(0)  # Remove batch dimension
        elif isinstance(output, (tuple, list)):
            # Handle multiple outputs
            return type(output)(o.squeeze(0) if isinstance(o, torch.Tensor) else o for o in output)
        else:
            return output
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model.
        This allows access to the wrapped model's methods and properties.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)