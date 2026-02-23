import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_T):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_T = max_T
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_T, d_model)
        self.embedding_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        (B, T) = x.shape
        assert T <= self.max_T, "Sequence length exceeds max_T"
        t = torch.arange(0, T, dtype=torch.long, device=x.device)

        embds = self.token_embedding(x)
        pos = self.positional_embedding(t)
        x = embds + pos
        x = self.embedding_dropout(x)

        return x