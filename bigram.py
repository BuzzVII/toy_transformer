# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
# ]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser(description="Train a bigram language model.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (e.g., 'cpu' or 'cuda').")
args = parser.parse_args()

# Tiny training text. Replace this with any small text file later.
text = """
hello transformer
this is a tiny bigram model
it predicts the next character from the current character
"""

# Character vocabulary.
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids):
    return "".join(itos[int(i)] for i in ids)

if args.device:
    device = args.device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

data = encode(text).to(device)

batch_size = 8
block_size = 16

def get_batch():
    # Pick random starting positions.
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    # x is the context.
    # y is the next character at each position.
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # This table maps each current token directly to logits for the next token.
        # Shape: vocab_size by vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: B, T
        # logits shape: B, T, vocab_size
        logits = self.token_embedding_table(idx)

        # REPLACE HERE LATER:
        #
        # In the transformer version, this direct lookup will become:
        #
        # token embeddings:
        #   x = token_embedding(idx)
        #
        # positional embeddings:
        #   x = x + position_embedding
        #
        # attention block:
        #   x = self_attention_block(x)
        #
        # final vocabulary projection:
        #   logits = lm_head(x)
        #
        # For now, the embedding table directly returns logits.

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # Cross entropy wants:
            # logits:  B*T, vocab_size
            # targets: B*T
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: B, T
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            # Use only the final time step to predict the next token.
            logits = logits[:, -1, :]  # B, vocab_size

            probs = F.softmax(logits, dim=-1)  # B, vocab_size

            # Sample one next token from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1

            # Append it to the sequence.
            idx = torch.cat([idx, idx_next], dim=1)  # B, T + 1

        return idx

model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(1000):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, float(loss))

# Generate from a single starting newline character.
start = torch.tensor([[stoi["\n"]]], dtype=torch.long)
out = model.generate(start, max_new_tokens=200)

print()
print(decode(out[0]))
