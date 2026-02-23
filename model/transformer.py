from .attention import Attention
from .mlp import MLP
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm)
        x = x + attn_out

        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x