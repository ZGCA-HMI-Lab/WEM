"""DPT-style mask prediction head conditioned on ego_state via cross-attention.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import WanCrossAttention, WanLayerNorm


class Reassemble(nn.Module):
    """Reshape flat tokens into a 3D feature map with optional spatial upsample."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 1):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        # tokens may be padded beyond T*H*W — truncate to the valid prefix.
        B, _, C = tokens.shape
        x = tokens[:, : T * H * W, :].reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        x = self.proj(x)
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=(1, self.scale_factor, self.scale_factor),
                mode="trilinear",
                align_corners=False,
            )
        return x


class FusionBlock(nn.Module):
    """Residual conv block used to merge features across taps (RefineNet-style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, channels)
        self.gn2 = nn.GroupNorm(32, channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = x + skip
        res = x
        x = F.silu(self.gn1(x))
        x = self.conv1(x)
        x = F.silu(self.gn2(x))
        x = self.conv2(x)
        return x + res


class EgoCondBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.norm_q = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm_kv = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.attn = WanCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        nn.init.zeros_(self.attn.o.weight)
        if self.attn.o.bias is not None:
            nn.init.zeros_(self.attn.o.bias)

    def forward(
        self,
        tap: torch.Tensor,
        ego_state: torch.Tensor,
        ego_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return tap + self.attn(self.norm_q(tap), self.norm_kv(ego_state), ego_lens)


class DPTMaskHead(nn.Module):
    """DPT-style mask head operating on encoder taps, conditioned on ego_state."""

    def __init__(
        self,
        in_channels: int,
        fusion_channels: int = 256,
        num_taps: int = 4,
        num_heads: int = 8,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_taps = num_taps

        self.cond_blocks = nn.ModuleList([
            EgoCondBlock(in_channels, num_heads=num_heads, qk_norm=qk_norm, eps=eps)
            for _ in range(num_taps)
        ])
        self.tap_norms = nn.ModuleList([
            WanLayerNorm(in_channels, eps, elementwise_affine=True)
            for _ in range(num_taps)
        ])
        self.reassemble_blocks = nn.ModuleList([
            Reassemble(in_channels, fusion_channels, scale_factor=1)
            for _ in range(num_taps)
        ])
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(fusion_channels) for _ in range(num_taps)
        ])
        self.head = nn.Sequential(
            nn.Conv3d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, fusion_channels // 2),
            nn.SiLU(),
            nn.Conv3d(fusion_channels // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        enc_taps: List[torch.Tensor],
        ego_state: torch.Tensor,
        T: int,
        H: int,
        W: int,
        ego_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            enc_taps:  list of ``num_taps`` tensors, each of shape (B, L, C).
            ego_state: (B, N_m, C) ego-branch queries from the world model,
                       already projected to ``C`` by the caller.
            T, H, W:   latent spatiotemporal grid of the video.
            ego_lens:  optional (B,) lengths for cross-attn masking.

        Returns:
            (B, 1, T, H, W) mask logits.
        """
        assert len(enc_taps) == self.num_taps, (
            f"DPTMaskHead expects {self.num_taps} taps, got {len(enc_taps)}"
        )

        features = []
        for i, tap in enumerate(enc_taps):
            cond = self.cond_blocks[i](tap, ego_state, ego_lens)
            cond = self.tap_norms[i](cond)
            features.append(self.reassemble_blocks[i](cond, T, H, W))

        x = self.fusion_blocks[-1](features[-1])
        for i in range(self.num_taps - 2, -1, -1):
            x = self.fusion_blocks[i](features[i], skip=x)

        return self.head(x)
