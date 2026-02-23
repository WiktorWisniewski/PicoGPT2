# nano-gpt2
Minimal GPT-2 style Transformer implemented from scratch in PyTorch, focusing on architectural clarity, mathematical correctness, and research readability.

## Architecture
The model follows the GPT-2 architecture:

- BPE tokenization
- learned token and positional embeddings
- multihead attention causal self-attention
- transofrmer blocks
- MLP with GELU activations

## Self-Attention

Scaled dot-product attention is implemented as:

Attention(Q, K, V) = softmax(QKᵀ / √dₕ + M) V

where M is a causal mask preventing access to future tokens.

## Tokenization

Data is tokenized using byte-pair encoding (BPE) algorithm. Tokens are stored as memory-mapped '.bin' file. It allows streaming during training on datasets larger than system memory.


## Training Objective

The model is trained autoregressively using next-token prediction:

P(xₜ | x₀, …, xₜ₋₁)

Cross-entropy loss is computed over the vocabulary. No softmax is applied inside the model, logits are passed directly to the loss.

## Text Generation

Text generation is performed autoregressively:

Forward pass predicts logits for the next token. Only the last time step is used. Sampling supports temperature and top-k

## Usage

Train:
python train.py --config config/gpt2.yaml

Generate:
Use `generate_text` from `generate.py` inside a notebook or script.

## Experiments

Qualitative experiments are provided in `notebooks/experiments.ipynb`,
including temperature scaling, top-k sampling, and reproducibility via random seeds.
