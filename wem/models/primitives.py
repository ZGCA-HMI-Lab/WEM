"""Self-contained primitives inlined from the Wan codebase.

Only the pieces needed by WanARModel / mask head are kept, and attention is
implemented via ``torch.nn.functional.scaled_dot_product_attention`` so the
module has no dependency on flash-attention.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        weight = self.weight.float() if (self.elementwise_affine and self.weight is not None) else None
        bias = self.bias.float() if (self.elementwise_affine and self.bias is not None) else None
        out = F.layer_norm(x_fp32, self.normalized_shape, weight, bias, self.eps)
        return out.type_as(x)


class WanCrossAttention(nn.Module):
    """Cross attention with optional RMSNorm on Q/K and key-length masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_dtype = self.q.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        if context.dtype != target_dtype:
            context = context.to(dtype=target_dtype)

        b, lq, _ = x.shape
        lk = context.size(1)
        n, d = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, lq, n, d).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(b, lk, n, d).transpose(1, 2)
        v = self.v(context).view(b, lk, n, d).transpose(1, 2)

        attn_mask = None
        if context_lens is not None:
            key_idx = torch.arange(lk, device=context.device)
            valid = key_idx[None, None, None, :] < context_lens.view(-1, 1, 1, 1)
            attn_mask = torch.zeros(b, 1, 1, lk, device=context.device, dtype=q.dtype)
            attn_mask.masked_fill_(~valid, float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, lq, -1)
        if out.dtype != target_dtype:
            out = out.to(dtype=target_dtype)
        return self.o(out)


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        flat_out = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, flat_out)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e_mod = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(
                self.norm(x) * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2)
            )
        return x
