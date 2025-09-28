import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the position of tokens in a sequence using sine and cosine functions.
    Accepts input of shape [batch_size, seq_len, embedding_dim].
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor of same shape as input with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CognateEncoder(nn.Module):
    """Base encoder for cognate words"""
    def __init__(self, embedder, hidden_dim=64, positional_dropout=0.2, dropout=0.0, layers=1, output_dim=128):
        super(CognateEncoder, self).__init__()
        
        self.embedder = embedder
        embedding_dim = embedder.embedding_dim
        self.embedding_dim = embedding_dim

        self.pos_encoder = PositionalEncoding(embedding_dim, positional_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            dim_feedforward=hidden_dim, 
            nhead=2, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Projection head to map to output dimension
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode_word(self, x, mask):
        # Ensure x is 2D: [batch_size, seq_len]
        if x.dim() > 2:
            x = x.squeeze()
            if x.dim() == 1:
                x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            
        # x shape: [batch_size, seq_len]
        x = self.embedder(x)
        pos_encoded_x = self.pos_encoder(x)
        encoded = self.transformer_encoder(pos_encoded_x, src_key_padding_mask=mask)
        
        # Use mean pooling to get fixed-size representation
        pooled = encoded.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Project to output dimension
        return self.projection(pooled)  # [batch_size, output_dim]

    def forward(self, x, mask):
        return self.encode_word(x, mask)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class CognateEmbeddingModel(nn.Module):
    """
    More memory-efficient version of SINCERE loss
    Processes positives and negatives in a single forward pass
    """
    def __init__(self, embedder, hidden_dim=64, positional_dropout=0.2, dropout=0.0, 
                 layers=1, output_dim=128, temperature=0.1):
        super(CognateEmbeddingModel, self).__init__()
        
        self.embedder = embedder
        embedding_dim = embedder.embedding_dim
        self.temperature = temperature

        self.pos_encoder = PositionalEncoding(embedding_dim, positional_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            dim_feedforward=hidden_dim, 
            nhead=2, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode_word(self, x: torch.Tensor, mask: torch.Tensor):
        """Encode a single word to normalized embedding"""
        assert len(x.shape) == 2, "Input x must be of shape [batch_size, seq_len]"
        assert x.shape == mask.shape, "Input x and mask must have the same shape"
    
        x = self.embedder(x)
        pos_encoded_x = self.pos_encoder(x)
        encoded = self.transformer_encoder(pos_encoded_x, src_key_padding_mask=mask)
        pooled = encoded.mean(dim=1)
        return F.normalize(self.projection(pooled), dim=1)

    def forward(self, batch_word_characters, batch_word_characters_masks):
        """
        Efficient SINCERE loss computation
        """
        assert batch_word_characters.dim() == 3, "Input must be of shape [batch_size, num_words, seq_len]"
        assert batch_word_characters.shape == batch_word_characters_masks.shape, "Characters and masks must have same shape"
        
        device = batch_word_characters.device
        
        batch_size, num_words, seq_len = batch_word_characters.shape
        output_dim = self.projection[-1].out_features

        all_embeddings = torch.zeros(batch_size, num_words, output_dim, device=device, dtype=torch.float)
        for i in range(batch_size):
            all_embeddings[i] = self.encode_word(
            batch_word_characters[i], batch_word_characters_masks[i]
            )

        return all_embeddings

# Stolen from https://github.com/tufts-ml/SupContrast/blob/master/revised_losses.py
class SINCERELoss(nn.Module):
    def __init__(self, temperature=0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor):
        """Supervised InfoNCE REvisited loss with cosine distance

        Args:
            embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        #assert embeds.dim() == 2 or embeds.dim() == 3, "Embeddings must be 2D or 3D tensor"
        #assert labels.dim() == 1 or labels.dim() == 2, "Labels must be 1D tensor or 2D tensor (batch_size, num_labels)"
        #assert (embeds.dim() - 1) == labels.dim(), "Embeddings have an embedding dim, while labels do not"

        if embeds.dim() == 3:
            #assert embeds.dim() == 3, "Embeddings must be 2D or 3D tensor"
            assert labels.dim() == 2, "Labels must be 1D tensor or 2D tensor (batch_size, num_labels)"
            losses = torch.zeros(embeds.size(0), device=embeds.device)
            for i in range(embeds.size(0)):
                loss = self.forward(embeds[i], labels[i])
                losses[i] = loss
            return losses.mean()
        
        assert embeds.dim() == 2, "Embeddings must be 2D or 3D tensor"
        assert labels.dim() == 1, "Labels must be 1D tensor or 2D tensor (batch_size, num_labels)"

        # Mask out embeddings and labels where label == 0 (padding)
        valid_mask = labels != 0
        embeds = embeds[valid_mask]
        labels = labels[valid_mask]

        # calculate logits (activations) for each embeddings pair (B, B)
        # using matrix multiply instead of cosine distance function for ~10x cost reduction
        logits = embeds @ embeds.T
        logits /= self.temperature
        # determine which logits are between embeds of the same label (B, B)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # masking with -inf to get zeros in the summation for the softmax denominator
        denom_activations = torch.full_like(logits, float("-inf"))
        denom_activations[~same_label] = logits[~same_label]
        # get logsumexp of the logits between embeds of different labels for each row (B,)
        base_denom_row = torch.logsumexp(denom_activations, dim=0)
        # reshape to be (B, B) with row values equivalent, to be masked later
        base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

        # get mask for numerator terms by removing comparisons between an image and itself (B, B)
        in_numer = same_label
        in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
        # delete same_label so don't need to copy for in_numer
        del same_label
        # count numerator terms for averaging (B,)
        numer_count = in_numer.sum(dim=0)
        if (numer_count != 0).any():
            # Mask out embeddings and labels with zero numerator count
            valid = numer_count != 0
            if not valid.all():
                embeds = embeds[valid]
                labels = labels[valid]
                return self.forward(embeds, labels)
        # numerator activations with others zeroed (B, B)
        numer_logits = torch.zeros_like(logits)
        numer_logits[in_numer] = logits[in_numer]

        # construct denominator term for each numerator via logsumexp over a stack (B, B)
        log_denom = torch.zeros_like(logits)
        log_denom[in_numer] = torch.stack(
            (numer_logits[in_numer], base_denom[in_numer]), dim=0).logsumexp(dim=0)

        # cross entropy loss of each positive pair with the logsumexp of the negative classes (B, B)
        # entries not in numerator set to 0
        ce = -1 * (numer_logits - log_denom)
        import math
        assert not torch.isnan(ce).any(), "Cross entropy is NaN. Try reducing the learning rate."
        assert ce.shape[0] >= 1
        # take average over rows with entry count then average over batch
        loss = torch.sum(ce / numer_count) / ce.shape[0]

        import math
        assert not math.isnan(loss.item()), "Loss is NaN. Try reducing the learning rate."

        return loss