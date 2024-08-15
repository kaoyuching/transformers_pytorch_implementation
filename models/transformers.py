from typing import Literal
import torch
import torch.nn as nn

from models.attention import Encode, Decoder


class Transformers(nn.Module):
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

        self.encoder = Encoder(num_emb, emb_dim, num_heads, emb_max_norm=emb_max_norm, n_blocks=n_blocks, pos_encoding_type=pos_encoding_type)
        self.decoder = Decoder(num_emb, emb_dim, num_heads, emb_max_norm=emb_max_norm, n_blocks=n_blocks, pos_encoding_type=pos_encoding_type)

    def forward(self, encode_x: torch.Tensor, decode_x: torch.Tensor, attn_mask: Optional[torch.Tendor] = None):
        encoder_out = self.encoder(encode_x)
        decoder_out = self.decoder(decode_x, encoder_out, attn_mask=attn_mask)
        return decoder_out
