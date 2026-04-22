"""
model.py — Mini Transformer built from scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        return self.emb(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=20, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


def attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(F.softmax(scores, dim=-1), V)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.h  = num_heads
        self.dk = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x, pad_mask=None):
        B = x.size(0)
        Q, K, V = self.split(self.Wq(x)), self.split(self.Wk(x)), self.split(self.Wv(x))
        mask = pad_mask.unsqueeze(1).unsqueeze(2) if pad_mask is not None else None
        out  = attention(Q, K, V, mask)
        out  = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.dk)
        return self.drop(self.Wo(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff    = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, pad_mask=None):
        x = self.norm1(x + self.attn(x, pad_mask))
        x = self.norm2(x + self.ff(x))
        return x


def mean_pool(x, pad_mask):
    valid  = (~pad_mask).float().unsqueeze(-1)
    return (x * valid).sum(1) / valid.sum(1).clamp(min=1)


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, num_heads=4,
                 ff_dim=128, num_layers=1, dropout=0.1,
                 use_positional_encoding=True, max_len=20):
        super().__init__()
        self.emb     = TokenEmbedding(vocab_size, embed_dim)
        self.pe      = PositionalEncoding(embed_dim, max_len, dropout) if use_positional_encoding else None
        self.encoder = nn.ModuleList([EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.head    = nn.Linear(embed_dim, 2)

    def forward(self, ids, pad_mask):
        x = self.emb(ids)
        if self.pe:
            x = self.pe(x)
        for block in self.encoder:
            x = block(x, pad_mask)
        return self.head(mean_pool(x, pad_mask))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
