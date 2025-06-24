def ldistance(str1, str2):
    # Get the lengths of the input strings
    m = len(str1)
    n = len(str2)

    # Initialize two rows for dynamic programming
    prev_row = [{
            "distance": j,
            "operation": ("Start"),
            "history": dict()
        } for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    # Dynamic programming to fill the matrix
    for i in range(1, m + 1):
        # Initialize the first element of the current row
        curr_row[0] = {
            "distance": i,
            "operation": ("Start"),
            "history": dict()
        }

        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                #curr_row[j] = prev_row[j - 1] #History doesn't matter
                curr_row[j] = { #History does matter
                    "distance": prev_row[j - 1]["distance"],
                    "operation": ("Nothing", str1[i - 1]),
                    "history": prev_row[j - 1]
                }
            else:
                # Choose the minimum cost operation
                curr_row[j] = min([
                    {
                        "distance": 1 + prev_row[j - 1]["distance"],    # Replace
                        "operation": ("Substitute", str1[i - 1], str2[j - 1]),
                        "history": prev_row[j- 1]
                    },
                    {
                        "distance": 1 + curr_row[j - 1]["distance"],  # Insert
                        "operation": ("Insert", str2[j - 1]),
                        "history": curr_row[j - 1]
                    },{
                        "distance": 1 + prev_row[j]["distance"],      # Remove
                        "operation": ("Remove", str1[i - 1]),
                        "history": prev_row[j]
                    }
                ], key=lambda cell: cell["distance"])

        # Update the previous row with the current row
        prev_row = curr_row.copy()

    def extract_operations(obj):
        operations = []
        while obj:
            operations.append(obj['operation'])
            obj = obj['history']
        return operations[::-1]

    # The final element in the last row contains the Levenshtein distance
    return extract_operations(curr_row[n])[1:]

import torch
import torch.nn as nn

def operation_encoder(operations, embedding: nn.Embedding):
    operation_encodings = {
        'Delete': torch.tensor([1, 0]),
        'Insert': torch.tensor([0, 1]),
        'Nothing': torch.tensor([0, 0])
    }

    EMBED_PLUS_OPERATION_SIZE = 2+embedding.embedding_dim

    def embed_operation(operation_name, character):
        tensor_a = torch.zeros([EMBED_PLUS_OPERATION_SIZE])
        tensor_a[:2] = operation_encodings[operation_name]
        tensor_a[2:] = embedding(character)
        return tensor_a
    
    def embed_operation_substitute(operation_name, character):
        tensor_a = embed_operation(operation_name, character)
        tensor_a[0] /= 2.0
        tensor_a[1] /= 2.0
        return tensor_a

    resulting_sequence_length = 0
    for operation in operations:
        if operation[0] == "Substitute":
            resulting_sequence_length += 2
        else:
            resulting_sequence_length += 1

    result = torch.empty([resulting_sequence_length, EMBED_PLUS_OPERATION_SIZE])

    index = 0
    for operation in operations:
        if operation[0] == "Substitute":
            result[index] = embed_operation_substitute("Delete", operation[1])
            index += 1
            result[index] = embed_operation_substitute("Insert", operation[2])
        else:
            result[index] = embed_operation(operation[0], operation[1])
        index += 1

    return result

def create_batch_from_string_pairs(string_pairs, embedding, pad_value=0.0):
    """
    Convert a list of string pairs to a batched tensor of operation sequences.
    
    Args:
        string_pairs: List of tuples [(str1, str2), (str1, str2), ...]
        embedding: The embedding function/model
        pad_value: Value to use for padding shorter sequences
    
    Returns:
        batched_tensor: Tensor of shape [batch_size, max_seq_len, feature_dim]
        lengths: List of actual sequence lengths for each item in batch
    """
    sequences = []
    lengths = []
    
    for str1, str2 in string_pairs:
        # Get the edit operations for this pair
        operations = ldistance(str1, str2)
        
        # Convert operations to tensor sequence
        if operations:  # Check if there are any operations
            sequence_tensor = operation_encoder(operations, embedding)
        else:
            assert False, "This code should never be reached."
        
        sequences.append(sequence_tensor)
        lengths.append(sequence_tensor.shape[0])
    
    # Pad sequences to same length
    # pad_sequence expects list of tensors, pads along first dimension
    from torch.nn.utils.rnn import pad_sequence
    batched_tensor = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    
    return batched_tensor, lengths

def create_batch_with_masks(string_pairs, embedding, pad_value=0.0):
    """
    Same as above but also returns attention masks for the sequences.
    
    Returns:
        batched_tensor: Tensor of shape [batch_size, max_seq_len, feature_dim]
        attention_masks: Boolean tensor of shape [batch_size, max_seq_len]
        lengths: List of actual sequence lengths
    """
    batched_tensor, lengths = create_batch_from_string_pairs(string_pairs, embedding, pad_value)
    
    batch_size, max_seq_len = batched_tensor.shape[0], batched_tensor.shape[1]
    
    # Create attention masks (True for real tokens, False for padding)
    attention_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_masks[i, :length] = True
    
    return batched_tensor, attention_masks, lengths

import math
from transformer_stuff import PositionalEncoding

class LDistanceModel(nn.Module):
    def __init__(self, vocab_size, embedding: nn.Embedding, hidden_dim=64):
        super(LDistanceModel, self).__init__()
        self.embedding = embedding

        dropout = .2
        self.pos_encoder = PositionalEncoding(embedding.embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding.embedding_dim, dim_feedforward=hidden_dim, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, mask_check=True) #mask_check makes sure we're only masking off the padding

        self.fc = nn.Sequential(
            nn.Linear(embedding.embedding_dim * 2, 64),  # Changed from hidden_dim * 2 to embedding_dim * 2
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.use_cosine_similarity = False
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def encode_word(self, operations):
        # Debug: Print input shape
        #print(f"Input shape: {x.shape}")
        
        # Ensure operations is 2D: [batch_size, seq_len]
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
        
        pos_encoded_x = self.pos_encoder(x)  # [batch_size, seq_len, embedding_dim]
        #print(f"After pos encoding shape: {pos_encoded_x.shape}")
        
        encoded = self.transformer_encoder(pos_encoded_x)  # [batch_size, seq_len, embedding_dim]
        #print(f"After transformer shape: {encoded.shape}")
        
        # Option 1: Use mean pooling to get a fixed-size representation
        return encoded.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Option 2: Use the last token (uncomment if preferred)
        # return encoded[-1]  # [batch_size, embedding_dim]

    def forward(self, input1, input2):
        enc1 = self.encode_word(input1)  # [batch_size, embedding_dim]
        enc2 = self.encode_word(input2)  # [batch_size, embedding_dim]
        
        # Compute cosine similarity

        #similarity = torch.norm(enc1 - enc2, dim=1)
        #import random
        #combined = torch.cat([enc1, enc2] if random.choice([True, False]) else [enc2, enc1], dim=1)  # [batch_size, embedding_dim * 2]

        if self.use_cosine_similarity:
            similarity = self.cosine_similarity(enc1, enc2)
            return torch.sigmoid(similarity)
        
        #similarity = torch.norm(enc1 - enc2, dim=1)
        
        import random
        combined = torch.cat([enc1, enc2] if random.choice([True, False]) else [enc2, enc1], dim=1)  # [batch_size, embedding_dim * 2]
        return self.fc(combined)