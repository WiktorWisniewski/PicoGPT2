import torch
import tiktoken
from model.gpt import GPT
from utils.sampling import top_k_logits
from typing import Optional


@torch.no_grad()
def generate_text(
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
):

    device = next(model.parameters()).device
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc.eot_token

    idx = torch.tensor(
        [enc.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_T:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, top_k)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        idx = torch.cat([idx, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return enc.decode(idx[0].tolist())