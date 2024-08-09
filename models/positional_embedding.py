import torch
import torch.nn as nn


class PositionalEncodeing(nn.Module):
    def __init__(self, base: int = 10000):
        super().__init__()
        self.base = base

    def forward(self, x):
        r"""
        x shape: (batch, seq_len, emb_dim)
        """
        bs, seq_len, emb_dim = x.shape
        pe = torch.zeros_like(x)
        
        for pos in range(seq_len):
            for i in range(0, self.emb_dim, 2):
                pe[:, pos, 2*i] = torch.sin(pos/torch.pow(self.base, 2*i / emb_dim))
                pe[:, pos, 2*i + 1] = torch.cos(pos/torch.pow(self.base, 2*i / emb_dim))
        return pe
