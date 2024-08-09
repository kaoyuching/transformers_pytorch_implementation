from typing import Optional
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.emb_q = nn.Linear(emb_dim, hidden_dim)
        self.emb_k = nn.Linear(emb_dim, hidden_dim)
        self.emb_v = nn.Linear(emb_dim, hidden_dim)
        self.softmax = nn.Softmax()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        r"""
        (q, k, v) shape: (batch, seq_len, num_heads, emb_dim)
        (q, k, v) shape: (batch, seq_len, emb_dim)
        """
        ndim = q.ndim
        if ndim == 3:
            q = q.unsqueeze(2)
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)

        q = self.emb_q(q)
        k = self.emb_k(k)
        v = self.emb_v(v)
        dim_k = k.shape[-1]

        if attn_mask is None:
            attn_mask = torch.ones_like(q)
        attn_mask = (1 - attn_mask) * -10000  # mask out the leftward information (set values to -inf)

        # scaled dot-product attention
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, emb_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        qk_dot = torch.matmul(q, k.transpose(2, 3))
        # apply attention mask
        sacle_qk_dot = qk_dot / torch.sqrt(dim_k)
        attn = self.softmax(scale_qk_dot * attn_mask) * v
        if ndim == 3:
            return attn.squeeze(dim=1)  # (batch, seq_len, emb_dim)
        else:
            return attn.transpose(1, 2)  # (batch, seq_len, num_heads, emb_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attn = SelfAttention(emb_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        r"""
        (q, k, v) shape: (batch, seq_len, emb_dim) -> (batch, seq_len, num_heads, emb_dim/num_heads)
        """
        q_heads = torch.stack(torch.chunk(q, self.num_heads, dim=-1), dim=2)
        k_heads = torch.stack(torch.chunk(k, self.num_heads, dim=-1), dim=2)
        v_heads = torch.stack(torch.chunk(v, self.num_heads, dim=-1), dim=2)
        attn = self.attn(q_heads, k_heads, v_heads, attn_mask)
        
        bs, seq_len = attn.shape[0, 1]
        output = self.linear(attn.reshape(bs, seq_len, -1))  # (batch, seq_len, emb_dim)
        return output
