'''
Rotary Positional Embeddings
'''

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, C):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, C)   # (vocab_size x C) learnable params

    def forward(self, x):
        token_emb = self.embedding(x)     
        return token_emb                           


class RotaryEmbedding(nn.Module):
    def __init__(self, T, C, base=10000):
        super().__init__()
        assert C % 2 == 0, "RoPE needs an even head_size to form rotation pairs"
        self.T = T
        self.C = C
        inv_freq = 1 / (base ** (torch.arange(0, C, 2).float() / C))     # (C/2,)
        t = torch.arange(T, dtype=torch.float32)                         # (T,)
        freqs = torch.outer(t, inv_freq)                                 # (T, C/2)
        self.register_buffer('cos', freqs.cos(), persistent=False)                         # (T, C/2)
        self.register_buffer('sin', freqs.sin(), persistent=False)                         # (T, C/2)

    def forward(self, x):
        T = x.shape[1]
        cos = self.cos[:T, :].unsqueeze(0)                      # (1, T, C/2)
        sin = self.sin[:T, :].unsqueeze(0)                      # (1, T, C/2)
        x1, x2 = x[..., 0::2], x[..., 1::2]                     # (B, T, C/2), (B, T, C/2)
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = (x1 * cos) - (x2 * sin)
        rotated[...,1::2] = (x1 * sin) + (x2 * cos)
        return rotated


class Head(nn.Module):
    def __init__(self, T, C, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(C, head_size)    # (head_size x C) + head_size learnable params
        self.key = nn.Linear(C, head_size)      # (head_size x C) + head_size learnable params
        self.value = nn.Linear(C, head_size)    # (head_size x C) + head_size learnable params
        self.rope = RotaryEmbedding(T, head_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))  # (T, T)

    def forward(self, x):

        T = x.shape[1]

        q = self.rope(self.query(x))   # (B, T, head_size)
        k = self.rope(self.key(x))     # (B, T, head_size)
        v = self.value(x)              # (B, T, head_size)

        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)                # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))        # (B, T, T) Dynamic causal-mask slicing
        weights = torch.softmax(weights, dim=-1)                                    # (B, T, T)
        weights = self.dropout(weights)                                             # (B, T, T)

        out = weights @ v    # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, T, C, num_heads, dropout):
        super().__init__()
        assert C % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = C // num_heads
        self.heads = nn.ModuleList([Head(T, C, self.head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(C, C)            # (C^2) + C  learnable params
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, C, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4*C),         # (4C^2) + 4C learnable params
            nn.GELU(),
            nn.Linear(4*C, C),         # (4C^2) + C learnable params
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, T, C, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)                                     # 2C learnable params
        self.attn = MultiHeadAttention(T, C, num_heads, dropout)       # (num_heads x (3 x head_size) x (C+1)) + (C^2 + C) learnable params
        self.ln2 = nn.LayerNorm(C)                                     # 2C learnable params
        self.ff = FeedForward(C, dropout)                              # 8C^2 + 5C learnable params 

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, T, C, vocab_size, num_heads, n_layers, dropout):
        super().__init__()
        self.T = T
        self.token_embedding = TokenEmbedding(vocab_size, C)
        self.blocks = nn.Sequential(*[Block(T, C, num_heads, dropout) for _ in range(n_layers)])  
        self.ln_f = nn.LayerNorm(C)                 # 2C learnable params
        self.lm_head = nn.Linear(C, vocab_size)     # (vocab_size x C) + vocab_size learnable params
        # Weight Tying
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_tokens=100, temperature=1.0):
        
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= self.T else idx[:, -self.T:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx