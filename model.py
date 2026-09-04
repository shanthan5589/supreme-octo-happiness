'''
Fused QKV with Rotary Positional Embeddings
'''

import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:
    T: int
    C: int
    vocab_size: int
    num_heads: int
    n_layers: int
    dropout: float = 0.0

    def __post_init__(self):
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.C % self.num_heads != 0:
            raise ValueError("C must be divisible by num_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if (self.C // self.num_heads) % 2 != 0:
            raise ValueError("C // num_heads must be even for RoPE")


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
        inv_freq = 1 / (base ** (torch.arange(0, C, 2).float() / C))     # (C/2,)
        t = torch.arange(T, dtype=torch.float32)                         # (T,)
        freqs = torch.outer(t, inv_freq)                                 # (T, C/2)
        self.register_buffer('cos', freqs.cos(), persistent=False)       # (T, C/2)
        self.register_buffer('sin', freqs.sin(), persistent=False)       # (T, C/2)

    def forward(self, x):
        T = x.shape[-2]
        cos = self.cos[:T, :].view(1, 1, T, -1).to(dtype=x.dtype)   # (1, 1, T, C/2)
        sin = self.sin[:T, :].view(1, 1, T, -1).to(dtype=x.dtype)   # (1, 1, T, C/2)
        x1, x2 = x[..., 0::2], x[..., 1::2]                         # (B, num_heads, T, C/2), (B, num_heads, T, C/2)
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = (x1 * cos) - (x2 * sin)
        rotated[...,1::2] = (x1 * sin) + (x2 * cos)
        return rotated


class MultiHeadAttention(nn.Module):
    def __init__(self, T, C, num_heads, dropout):
        super().__init__()
        self.T = T
        self.num_heads = num_heads
        assert C % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = C // num_heads
        self.qkv = nn.Linear(C, 3 * C)      # (3C, C) 
        self.rope = RotaryEmbedding(T, self.head_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))  # (T, T)
        self.proj = nn.Linear(C, C)            # (C^2) + C  learnable params

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)                                                       # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_size)                 # (B, T, 3, num_heads, head_size)
        q, k, v = qkv.unbind(dim=2)                                             # (B, T, num_heads, head_size) x 3
        q = q.transpose(1, 2)                                                   # (B, num_heads, T, head_size)
        k = k.transpose(1, 2)                                                   # (B, num_heads, T, head_size)
        v = v.transpose(1, 2)                                                   # (B, num_heads, T, head_size)
        q, k = self.rope(q), self.rope(k)                                       # (B, num_heads, T, head_size) x 2     
        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)            # (B, num_heads, T, head_size) x (B, num_heads, head_size, T) -> (B, num_heads, T, T)      
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, num_heads, T, T) Dynamic causal-mask slicing
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v                                                       # (B, num_heads, T, head_size)
        out = out.transpose(1, 2).contiguous()                                  # (B, T, num_heads, head_size)
        out = out.view(B, T, C)                                                 # (B, T, C) because num_heads * head_size = C                  
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
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.C)
        self.blocks = nn.Sequential(*[Block(config.T, config.C, config.num_heads, config.dropout) for _ in range(config.n_layers)])  
        self.ln_f = nn.LayerNorm(config.C)                 # 2C learnable params
        self.lm_head = nn.Linear(config.C, config.vocab_size)     # (vocab_size x C) + vocab_size learnable params
        # Weight Tying
        self.lm_head.weight = self.token_embedding.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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