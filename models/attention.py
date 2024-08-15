from typing import Optional
import torch
import torch.nn as nn

from models.positional_embedding import PositionalEncodeing, RotaryPositonalEncoding


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
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


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(emb_dim, emb_dim, num_heads)
        # feed-forward network
        self.mlp = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        ])
        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        attn_x = self.multi_head_attn(x, x, x, attn_mask=None)
        x = self.norm_attn(x + attn_x)

        mlp_x = torch.clone(x)
        for mlp_layer in self.mlp:
            mlp_x = mlp_layer(mlp_x)
        x = self.norm_mlp(x + mlp_x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.mask_multi_head_attn = MultiHeadAttention(emb_dim, emb_dim, num_heads)
        self.multi_head_attn = MultiHeadAttention(emb_dim, emb_dim, num_heads)
        # feed-forward network
        self.mlp = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        ])
        self.norm_mask_attn = nn.LayerNorm(emb_dim)
        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, encoder_output, attn_mask: Optional[torch.tensor] = None):
        mask_attn_x = self.mask_multi_head_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm_mask_attn(x + mask_attn_x) # next input q

        attn_x = self.multi_head_attn(x, encoder_output, encoder_output, attn_mask=None)
        x = self.norm_attn(x + attn_x) # next input q

        mlp_x = torch.clone(x)
        for mlp_layer in self.mlp:
            mlp_x = mlp_layer(mlp_x)
        x = self.norm_mlp(x + mlp_x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_emb: int,
        emb_dim: int,
        num_heads: int,
        emb_max_norm: bool = True,
        n_blocks: int = 6,
        pos_encoding_type: Literal["absolute", "rotary"] = "absolute",
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(num_emb, emb_dim, max_norm=emb_max_norm)
        self.pos_encoding_type = self._check_pos_type(pos_encoding_type)
        if self.pos_encoding_type == "absolute":
            self.pos_encoding = PositionalEncodeing()
        elif self.pos_encoding_type == "rotary":
            self.pos_encoding = RotaryPositonalEncoding()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(emb_dim, num_heads) for i in range(n_blocks)
        ])

    def _check_pos_type(self, pos_type: str):
        pos_type = pos_type.lower()
        if pos_type not in ["absolute", "rotary"]:
            return "absolute"
        else:
            return pos_type

    def forward(self, x: torch.Tensor):
        # input > emb > add pos_encoding > multi-head attn > MLP
        x_emb = self.input_embedding(x)

        if self.pos_encoding_type == "rotary":
            x_emb = self.pos_encoding(x_emb)
        else:
            pe = self.pos_encoding(x)
            x_emb = x_emb + pe

        x_block = torch.clone(x_emb)
        for block in self.encoder_blocks:
            x_block = block(x_block)
        return x_block


class Decoder(nn.Module):
    def __init__(
        self,
        num_emb: int,
        emb_dim: int,
        num_heads: int,
        emb_max_norm: bool = True,
        n_blocks: int = 6,
        pos_encoding_type: Literal["absolute", "rotary"] = "absolute",
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(num_emb, emb_dim, max_norm=emb_max_norm)
        self.pos_encoding_type = self._check_pos_type(pos_encoding_type)
        if self.pos_encoding_type == "absolute":
            self.pos_encoding = PositionalEncodeing()
        elif self.pos_encoding_type == "rotary":
            self.pos_encoding = RotaryPositonalEncoding()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads) for i in range(n_blocks)
        ])
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.softmax = nn.Softmax(emb_dim)

    def _check_pos_type(self, pos_type: str):
        pos_type = pos_type.lower()
        if pos_type not in ["absolute", "rotary"]:
            return "absolute"
        else:
            return pos_type

    def forward(self, x: torch.Tensor, encoder_output, attn_mask: Optional[torch.tensor] = None):
        x_emb = self.input_embedding(x)

        if self.pos_encoding_type == "rotary":
            x_emb = self.pos_encoding(x_emb)
        else:
            pe = self.pos_encoding(x)
            x_emb = x_emb + pe

        x_block = torch.clone(x_emb)
        for block in self.decoder_blocks:
            x_block = block(x_block, encoder_output, attn_mask=attn_mask)
        output = self.linear(x_block)
        return self.softmax(output)
