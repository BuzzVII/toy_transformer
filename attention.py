# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Train a tiny single-head self-attention language model.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training, e.g. 'cpu' or 'cuda'.",
)
parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
parser.add_argument("--generate-tokens", type=int, default=200, help="Number of new tokens to generate.")
parser.add_argument("--context", type=str, default="\n", help="Input text to continue from. If empty, starts from a newline character.")
parser.add_argument("--no-plots", action="store_true", help="Do not show matplotlib plots.")
args = parser.parse_args()

device = args.device
print(f"using device: {device}")

# Tiny training text. Replace this with any small text file later.
text = """
hello transformer
this is a tiny single head self attention model
it predicts the next character from the context before it
i like ham and eggs
i am legion
as well as spam and peas
kristian is a nice guy
paul is a jerk
i like to eat pizza
i am a language model
i look forward to learning more about attention
i do not work
i was made by kristian
the quick brown fox jumped over the lazy dog
she sells sea shells by the sea shore
mary had a little lamb
how much wood would a woodchuck chuck if a woodchuck could chuck wood
it was the best of times it was the worst of times
to be or not to be that is the question
attention learns which previous characters matter
logits predict the next character
"""

# Character vocabulary.
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    unknown = sorted(set(s) - set(stoi))
    if unknown:
        raise ValueError(
            "Context contains characters that are not in the training vocabulary: "
            + ", ".join(repr(c) for c in unknown)
        )
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

        # store the last attention weights for visualization only.
        self.last_weights = None

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

        self.last_weights = weights.detach().cpu()  # for visualization only

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

losses = []

for step in range(args.steps):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(float(loss.detach().cpu()))

    if step % 100 == 0:
        print(step, float(loss.detach()))

if not args.no_plots:
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy Loss")
    plt.tight_layout()
    plt.show()

# Generate from the model. Start with the input text or a newline if no input is provided.
context = args.context if args.context else "\n"

# The model only has positional embeddings from 0 to block_size - 1.
# So direct model inspection must use at most block_size tokens.
# Generation already crops internally on each step.
context_for_model = context[-block_size:]

start = encode(context).unsqueeze(0).to(device)              # B, T, used for generation
inspect = encode(context_for_model).unsqueeze(0).to(device) # B, <= block_size, used for plots

logits, loss = model(inspect)
next_logits = logits[0, -1, :]
probs = F.softmax(next_logits, dim=-1).detach().cpu()

if not args.no_plots:
    plt.figure()
    plt.bar(range(vocab_size), probs)
    plt.xticks(
        range(vocab_size),
        [repr(itos[i]) for i in range(vocab_size)],
        rotation=90,
    )
    plt.title(f"Next character probabilities after {repr(context_for_model)}")
    plt.xlabel("Next character")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

    weights = model.self_attention.last_weights[0]  # T, T
    labels = [repr(c) for c in context_for_model]

    plt.figure()
    plt.imshow(weights)
    plt.title("Attention Weights")
    plt.xlabel("Source position read from")
    plt.ylabel("Target position being updated")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

out = model.generate(start, max_new_tokens=args.generate_tokens)

print()
print(decode(out[0]))
