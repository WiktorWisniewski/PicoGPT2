import torch
from model.attention import Attention

def test_attention_shape():
    B, T, d_model, n_heads = 2, 8, 32, 4
    x = torch.randn(B, T, d_model)

    attn = Attention(d_model=d_model, n_heads=n_heads)
    y = attn(x)

    assert y.shape == x.shape


def test_attention_causal_mask():
    torch.manual_seed(0)

    B, T, d_model, n_heads = 1, 4, 16, 4
    attn = Attention(d_model=d_model, n_heads=n_heads)
    attn.eval() 

    x1 = torch.randn(B, T, d_model)

    x2 = x1.clone()
    x2[:, -1] = torch.randn(d_model)

    y1 = attn(x1)
    y2 = attn(x2)

    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)

