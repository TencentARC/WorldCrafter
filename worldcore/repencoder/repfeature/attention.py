from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops
except ImportError:  # pragma: no cover - exercised only in minimal CPU installs
    xops = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight.to(value.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention width must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        if kv is None:
            kv = query
        batch, query_tokens, channels = query.shape
        key_tokens = kv.shape[1]
        q = self.q_proj(query).reshape(
            batch, query_tokens, self.num_heads, self.head_dim
        )
        k = self.k_proj(kv).reshape(batch, key_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(kv).reshape(batch, key_tokens, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if xops is not None and query.is_cuda:
            value = xops.memory_efficient_attention(q, k, v, p=0.0, op=None)
        else:
            value = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
            ).transpose(1, 2)
        value = value.reshape(batch, query_tokens, channels)
        return self.proj(value)


__all__ = ["Attention", "RMSNorm"]
