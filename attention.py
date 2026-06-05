# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
# ]
# ///

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Train a tiny single-head self-attention language model.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training, e.g. 'cpu' or 'cuda'.",
)
parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
parser.add_argument("--generate-tokens", type=int, default=200, help="Number of new tokens to generate.")
args = parser.parse_args()

device = args.device
print(f"using device: {device}")

# Tiny training text. Replace this with any small text file later.
text = """
hello transformer
this is a tiny single head self attention model
it predicts the next character from the context before it
i like ham and eggs
as well as spam and peas
kristian is a nice guy
paul is a jerk
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


data = encode(text).to(device)

batch_size = 8
block_size = 16
n_embd = 32


def get_batch():
    # Pick random starting positions.
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=device)

    # x is the context.
    # y is the next character at each position.
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class SingleSelfAttentionHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        # Each token representation is projected into query, key, and value vectors.
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        # Causal mask. A token can only read itself and earlier tokens.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: B, T, C
        B, T, C = x.shape

        q = self.query(x)  # B, T, C
        k = self.key(x)    # B, T, C
        v = self.value(x)  # B, T, C

        # Attention scores. For each token position, compare its query to all keys.
        scores = q @ k.transpose(-2, -1)  # B, T, T
        scores = scores / math.sqrt(C)

        # Hide future positions before softmax.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Attention weights are probabilities over context positions.
        weights = F.softmax(scores, dim=-1)  # B, T, T

        # Each token receives the weighted average of the value vectors it attends to.
        out = weights @ v  # B, T, C
        return out


class TinySelfAttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Unlike the bigram model, token embeddings are no longer logits.
        # They are hidden vectors that attention can operate on.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # REPLACED HERE:
        # The old bigram model did this:
        #   logits = token_embedding_table(idx)
        #
        # Now we do this:
        #   x = token_embedding(idx)
        #   x = x + position_embedding
        #   x = self_attention(x)
        #   logits = lm_head(x)
        self.self_attention = SingleSelfAttentionHead(n_embd)

        # Final projection from hidden vectors to next-token logits.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: B, T
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # B, T, C
        pos = torch.arange(T, device=idx.device)     # T
        pos_emb = self.position_embedding_table(pos) # T, C

        x = token_emb + pos_emb                      # B, T, C
        x = self.self_attention(x)                   # B, T, C
        logits = self.lm_head(x)                     # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: B, T
        for _ in range(max_new_tokens):
            # Positional embeddings only exist up to block_size, so crop long context.
            idx_context = idx[:, -block_size:]

            logits, loss = self(idx_context)
            logits = logits[:, -1, :]              # B, vocab_size
            probs = F.softmax(logits, dim=-1)      # B, vocab_size
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat([idx, idx_next], dim=1) # B, T + 1

        return idx


model = TinySelfAttentionLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(args.steps):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, float(loss.detach()))

# Generate from a single starting newline character.
start = torch.tensor([[stoi["\n"]]], dtype=torch.long, device=device)
out = model.generate(start, max_new_tokens=args.generate_tokens)

print()
print(decode(out[0]))
