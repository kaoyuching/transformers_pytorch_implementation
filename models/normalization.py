import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    r"""
    RMSNorm re-scales invariance and regularizes the summed inputs according to the root mean square.
    RMSNorm is equal to LayerNorm when the mean of summed inputs is zero.
    """
    def __init__(self, emb_dim: int, eps: float = 1e-8, affine: bool = False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        if affine:
            self.gain = nn.Parameter(torch.ones(emb_dim))
            self.bias = nn.Parameter(torch.zeros(emb_dim))
        else:
            self.gain = torch.ones(emb_dim)
            self.bias = torch.zeros(emb_dim)

    def forward(self, x: torch.Tensor):
        r"""
        s shape: (b, seq_len, emb_dim)
        """
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True))
        x = x / (rms + self.eps)
        x = x * self.gain + self.bias
        return x
