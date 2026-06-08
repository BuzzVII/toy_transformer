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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT = """
hello transformer
this is a tiny multi head self attention model
it predicts the next character from the context before it
i like ham and eggs
as well as spam and peas
kristian is a nice guy
paul is a jerk
the quick brown fox jumped over the lazy dog
she sells sea shells by the sea shore
mary had a little lamb
how much wood would a woodchuck chuck if a woodchuck could chuck wood
it was the best of times it was the worst of times
to be or not to be that is the question
attention learns which previous characters matter
logits predict the next character
"""


class Vocabulary:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_text(cls, text):
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def size(self):
        return len(self.stoi)

    def encode(self, s):
        unknown = sorted(set(s) - set(self.stoi))
        if unknown:
            raise ValueError(
                "Context contains characters that are not in the training vocabulary: "
                + ", ".join(repr(c) for c in unknown)
            )
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.last_scores = None
        self.last_weights = None

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        self.last_scores = scores.detach().cpu()
        self.last_weights = weights.detach().cpu()

        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        if n_head < 1:
            raise ValueError("n_head must be at least 1")
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head}) "
                "so the concatenated heads return to n_embd."
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [SelfAttentionHead(n_embd, self.head_size, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out

    def get_last_weights(self):
        return [head.last_weights for head in self.heads]

    def get_last_scores(self):
        return [head.last_scores for head in self.heads]


class TinySelfAttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.self_attention = MultiHeadAttention(n_embd, n_head, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Input length {T} is longer than block_size {self.block_size}. "
                "Crop the context before calling forward, or use generate which crops internally."
            )

        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = token_emb + pos_emb
        x = self.self_attention(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size :]

            logits, loss = self(idx_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


def get_device(requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def load_training_text(text_file):
    if text_file is None:
        return DEFAULT_TEXT
    return Path(text_file).read_text(encoding="utf-8")


def make_batcher(data, batch_size, block_size, device):
    def get_batch():
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=device)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    return get_batch


def save_checkpoint(path, model, optimizer, vocab, config, losses, step):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "stoi": vocab.stoi,
        "itos": vocab.itos,
        "config": config,
        "losses": losses,
        "step": step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = Vocabulary(stoi=checkpoint["stoi"], itos=checkpoint["itos"])

    model = TinySelfAttentionLanguageModel(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint, model, vocab, config


def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy Loss")
    plt.show()


def plot_next_token_probs(model, vocab, context, device):
    context_for_model = context[-model.block_size :]
    inspect = vocab.encode(context_for_model).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, loss = model(inspect)
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1).detach().cpu()

    plt.figure()
    plt.bar(range(vocab.size), probs)
    plt.xticks(range(vocab.size), [repr(vocab.itos[i]) for i in range(vocab.size)], rotation=90)
    plt.title(f"Next character probabilities after {repr(context_for_model)}")
    plt.xlabel("Next character")
    plt.ylabel("Probability")
    plt.show()


def plot_attention_weights(model, vocab, context, device):
    context_for_model = context[-model.block_size :]
    inspect = vocab.encode(context_for_model).unsqueeze(0).to(device)

    with torch.no_grad():
        model(inspect)

    weights_by_head = model.self_attention.get_last_weights()
    labels = [repr(c) for c in context_for_model]
    n_head = len(weights_by_head)
    n_cols = math.ceil(math.sqrt(n_head))
    n_rows = math.ceil(n_head / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    last_image = None

    for head_index, weights in enumerate(weights_by_head):
        row = head_index // n_cols
        col = head_index % n_cols
        ax = axes[row][col]

        last_image = ax.imshow(weights[0])
        ax.set_title(f"Head {head_index}")
        ax.set_xlabel("Source position read from")
        ax.set_ylabel("Target position being updated")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    for empty_index in range(n_head, n_rows * n_cols):
        row = empty_index // n_cols
        col = empty_index % n_cols
        axes[row][col].axis("off")

    fig.suptitle("Attention Weights by Head")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.8)
    plt.show()


def run_train(args):
    device = get_device(args.device)
    print(f"using device: {device}")

    text = load_training_text(args.text_file)
    vocab = Vocabulary.from_text(text)
    data = vocab.encode(text).to(device)

    config = {
        "vocab_size": vocab.size,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "batch_size": args.batch_size,
    }

    model = TinySelfAttentionLanguageModel(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    get_batch = make_batcher(data, args.batch_size, args.block_size, device)

    losses = []

    model.train()
    for step in range(args.steps):
        xb, yb = get_batch()

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        if step % args.print_every == 0:
            print(step, float(loss.detach().cpu()))

    save_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=optimizer,
        vocab=vocab,
        config=config,
        losses=losses,
        step=args.steps,
    )
    print(f"saved checkpoint: {args.checkpoint}")

    model.eval()
    context = args.context if args.context else "\n"
    start = vocab.encode(context).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(start, max_new_tokens=args.generate_tokens)

    print()
    print(vocab.decode(out[0]))

    if not args.no_plots:
        plot_loss(losses)
        plot_next_token_probs(model, vocab, context, device)
        plot_attention_weights(model, vocab, context, device)


def run_infer(args):
    device = get_device(args.device)
    print(f"using device: {device}")

    checkpoint, model, vocab, config = load_checkpoint(args.checkpoint, device)
    model.eval()

    context = args.context if args.context else "\n"
    start = vocab.encode(context).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(start, max_new_tokens=args.generate_tokens)

    print()
    print(vocab.decode(out[0]))

    if not args.no_plots:
        if checkpoint.get("losses"):
            plot_loss(checkpoint["losses"])
        plot_next_token_probs(model, vocab, context, device)
        plot_attention_weights(model, vocab, context, device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run a tiny multi head self attention language model.")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--checkpoint", type=str, default="attention_checkpoint.pt")

    parser.add_argument("--text-file", type=str, default=None, help="Optional UTF-8 text corpus file for training.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--print-every", type=int, default=100)

    parser.add_argument("--generate-tokens", type=int, default=200)
    parser.add_argument("--context", type=str, default="\n")
    parser.add_argument("--no-plots", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
