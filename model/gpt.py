import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .embeddings import Embeddings


class GPT(nn.Module):
    def __init__(self, d_model, vocab_size, max_T, n_layers, n_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_T = max_T
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.blocks = nn.ModuleList([])
        self.emb = Embeddings(vocab_size, d_model, max_T)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.head(x)

        return logits
    

