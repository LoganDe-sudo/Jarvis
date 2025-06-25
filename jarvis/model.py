import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

def generate_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool()

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class ExpandableEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))

    def expand(self, new_vocab_size):
        if new_vocab_size > self.weight.size(0):
            new_weight = torch.randn(new_vocab_size, self.embed_dim)
            new_weight[:self.weight.size(0)] = self.weight.data
            self.weight = nn.Parameter(new_weight)

    def forward(self, x):
        return F.embedding(x, self.weight)

    def get_vocab_size(self):
        return self.weight.size(0)

class ExpandableLinearHead(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def expand(self, new_vocab_size):
        if new_vocab_size > self.weight.size(0):
            old_vocab_size = self.weight.size(0)
            new_weight = torch.randn(new_vocab_size, self.embed_dim)
            new_bias = torch.zeros(new_vocab_size)
            new_weight[:old_vocab_size] = self.weight.data
            new_bias[:old_vocab_size] = self.bias.data
            self.weight = nn.Parameter(new_weight)
            self.bias = nn.Parameter(new_bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def get_vocab_size(self):
        return self.weight.size(0)

class Jarvis(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        cfg = config or {
            "embed_dim": 128,
            "num_heads": 4,
            "ff_hidden": 512,
            "max_len": 512,
            "num_layers": 4,
            "dropout": 0.1
        }

        self.embedding = ExpandableEmbedding(vocab_size, cfg["embed_dim"])
        self.position = PositionalEncoding(cfg["embed_dim"], cfg["max_len"])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg["embed_dim"], cfg["num_heads"], cfg["ff_hidden"], cfg["dropout"])
            for _ in range(cfg["num_layers"])
        ])
        self.lm_head = ExpandableLinearHead(cfg["embed_dim"], vocab_size)
        print(f"🧫 Jarvis initialized with dynamic vocab size: {vocab_size} and config: {cfg}")

    def expand_vocab(self, new_vocab_size):
        self.embedding.expand(new_vocab_size)
        self.lm_head.expand(new_vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        seq_len = x.size(1)
        attn_mask = ~generate_causal_mask(seq_len).to(x.device)
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        return self.lm_head(x)

    def get_vocab_size(self):
        return self.lm_head.get_vocab_size()
