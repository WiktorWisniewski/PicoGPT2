import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.d_head = int(d_model / n_heads)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(p=0.1)
        self.resid_dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        B, T, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # split into multi-head attention
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # attention scores
        attention_sc = (Q@K.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.full_like(attention_sc, 
                                        -1e6  #   float('-inf'), for mac training
                                          ), diagonal=1)
        attention_sc = attention_sc + mask

        # attention weights
        attn = torch.softmax(attention_sc, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V

        # concatenate heads
        out = out.transpose(1, 2) 
        out = out.contiguous()
        out = out.view(B, T, self.d_model)

        # mix heads
        out = self.W_O(out)
        out = self.resid_dropout(out)

        assert out.shape == (B, T, self.d_model)

        return out


         