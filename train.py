import yaml
import numpy as np
import torch
from torch import nn
import os
from model import GPT
import torch.optim as optim



with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    

model_cfg = config["model"]
train_cfg = config["training"]
data_cfg  = config["data"]

dataset_path = config["data"]['dataset_path']

vocab_size = model_cfg['vocab_size']
max_T = model_cfg['max_T']
d_model = model_cfg['d_model']
n_heads = model_cfg['n_heads']
d_ff = model_cfg['d_ff']
n_layers = model_cfg['n_layers']

batch_size = train_cfg['batch_size']
learning_rate = float(train_cfg['learning_rate'])
total_steps = train_cfg['total_steps']
weight_decay = train_cfg['weight_decay']
learning_rate = train_cfg['learning_rate']

# this is only for mac people, and for poor mac people to be precise
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load only part of the data into memory
tokens = np.memmap(
    os.path.join(dataset_path, "train.bin"),
    dtype=np.uint16,
    mode="r"
)



def get_batch(tokens, batch_size, block_size, device):
    ix = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))

    x = torch.stack([
        torch.from_numpy(tokens[i : i + block_size].astype(np.int64)).long()
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(tokens[i + 1 : i + block_size + 1].astype(np.int64)).long()
        for i in ix
    ])

    return x.to(device), y.to(device)

model = GPT(d_model, vocab_size, max_T, n_layers, n_heads, d_ff)
model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

start_step = 0
if os.path.exists("checkpoint.pt"):
    ckpt = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1


def save_checkpoint(state, path="checkpoint.pt"):
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  


for step in range(start_step, total_steps):
    x, y = get_batch(tokens, batch_size, max_T, device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)
    loss = criterion(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        with torch.no_grad():
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": config,
            })

    if step % 10 == 0:
        print(f"step {step} | loss {loss.item():.4f}")