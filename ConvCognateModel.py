import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math
from torch import Tensor

class CognateEncoder(nn.Module):
    """Base encoder for cognate words"""
    def __init__(self, embedder, hidden_dim=64, positional_dropout=0.2, dropout=0.0, layers=1, output_dim=128):
        super(CognateEncoder, self).__init__()
        
        self.embedder = embedder
        embedding_dim = embedder.embedding_dim
        self.embedding_dim = embedding_dim
        
        # Enhanced convolutional block with skip connections and more layers
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.projection = nn.Linear(hidden_dim, output_dim)

    def _conv_with_skip(self, emb):
        # emb: [batch_size, embedding_dim, seq_len]
        x = emb
        out1 = self.conv[0](x)
        out1 = self.conv[1](out1)
        out2 = self.conv[2](out1)
        out2 = self.conv[3](out2)
        skip1 = out2 + out1  # First skip connection
        out3 = self.conv[4](skip1)
        out3 = self.conv[5](out3)
        skip2 = out3 + skip1  # Second skip connection
        out4 = self.conv[6](skip2)
        out4 = self.conv[7](out4)
        out = self.conv[8](out4)
        return out

    def encode_words(self, x, mask):
        """
        x: [batch_size, seq_len] (indices of letters)
        mask: [batch_size, seq_len] (bool mask, True for padding)
        """
        assert x.dim() == 2, "Input x must be of shape [batch_size, seq_len]"
        emb = self.embedder(x)  # [batch_size, seq_len, embedding_dim]
        emb = emb.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        conv_out = self.conv(emb).squeeze(-1)  # [batch_size, hidden_dim]
        out = self.projection(conv_out)  # [batch_size, output_dim]
        return F.normalize(out, dim=1)

    def forward(self, x, mask):
        return torch.vmap(self.encode_words, in_dims=(0, 0))(x, mask)

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