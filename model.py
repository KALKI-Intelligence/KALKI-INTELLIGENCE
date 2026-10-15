"""Kalki Mini: compact GPT-style decoder with a causal Compass context path."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class KalkiConfig:
    vocab_size: int = 50_257  # GPT-2 tokenizer vocabulary; no GPT-2 model weights
    context_length: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    dropout: float = 0.05
    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, x):
        length = x.size(-2)
        positions = torch.arange(length, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        cos, sin = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

class CompassBlock(nn.Module):
    """Causal self-attention plus a gated, prefix-summary route.

    Each position's summary sees only tokens at or before that position. This is
    Kalki's experimental architecture, not a claim of state-of-the-art novelty.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.norm1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_dropout = config.dropout
        self.norm2 = nn.LayerNorm(config.d_model)
        self.compass_value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.compass_gate = nn.Linear(config.d_model, config.d_model)
        self.ff = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model), nn.GELU(),
                                nn.Linear(4 * config.d_model, config.d_model), nn.Dropout(config.dropout))
    def forward(self, x):
        b, t, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        reshape = lambda a: a.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        q, k = self.rope(q), self.rope(k)
        a = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)
        x = x + self.out(a.transpose(1, 2).contiguous().view(b, t, width))
        normalized = self.norm2(x)
        positions = torch.arange(1, t + 1, device=x.device, dtype=normalized.dtype)
        summary = normalized.cumsum(dim=1) / positions[None, :, None]
        x = x + torch.sigmoid(self.compass_gate(normalized)) * self.compass_value(summary)
        return x + self.ff(normalized)

class KalkiMini(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or KalkiConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(CompassBlock(c) for _ in range(c.n_layers))
        self.final_norm = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self, token_ids, targets=None):
        if token_ids.size(1) > self.config.context_length:
            raise ValueError(f"Input length exceeds {self.config.context_length} tokens")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
    @torch.inference_mode()
    def generate(self, token_ids, max_new_tokens=160, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids[:, -self.config.context_length:])
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k > 0:
                threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            token_ids = torch.cat((token_ids, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return token_ids

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
