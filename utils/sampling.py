import torch
from typing import Optional

NEG_INF = -1e5  # MPS-safe

def top_k_logits(logits, k: Optional[int]):
    if k is None:
        return logits
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = NEG_INF
    return logits
