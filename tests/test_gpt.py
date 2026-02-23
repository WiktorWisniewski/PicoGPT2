import torch
from model.gpt import GPT


def test_gpt_forward_shape():
    B, T = 2, 8
    vocab_size = 100
    d_model = 32
    n_heads = 4
    n_layers = 2
    d_ff = 64
    max_T = 16

    model = GPT(
        d_model=d_model,
        vocab_size=vocab_size,
        max_T=max_T,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
    )

    x = torch.randint(0, vocab_size, (B, T))
    logits = model(x)

    assert logits.shape == (B, T, vocab_size)
