import torch
import torch.nn as nn


class PositionalEncodeing(nn.Module):
    pe_type = "absolute"

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


class RotaryPositonalEncoding(nn.Module):
    pe_type = "rotary"

    def __init__(self, base: int = 10000):
        super().__init__()
        self.base = base

    def forward(self, x):
        r"""
        x shape: (batch, seq_len, emb_dim)
        """
        bs, seq_len, emb_dim = x.shape
        theta = torch.zeros(seq_len, emb_dim)
        for i in range(seq_len):
            theta[i] = [
                torch.pow(self.base, -2 * (torch.ceil(d/2)-1) / emb_dim)
                for d in range(1, emb_dim)
            ]
            theta[i] *= i

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        cos_x = x
        sin_x = torch.zeros_like(cos_x)
        sin_x[..., 0::2] = -x[..., 1::2]
        sin_x[..., 1::2] = x[..., 0::2]

        rope = cos_x * cos + sin_x * sin
        return rope
