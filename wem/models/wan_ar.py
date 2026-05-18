"""WanAR DiT with attention-level routing between world / ego branches.

Architecture:

    1. Shared ``enc_blocks`` process the full noisy latent. Intermediate
       features at a configurable set of tap layers are cached for the mask
       head.
    2. After the encoder, a DPT-style mask head predicts a soft mask from
       those enc taps, conditioned on the ego_state tokens via
       cross-attention (zero-initialised so at step 0 the mask head is a pure
       enc-feature DPT).
    3. The predicted mask is thresholded into a hard 0/1 partition. Each of
       two decoder stacks (``world_dec_blocks`` / ``ego_dec_blocks``) runs
       only over its partition: ``key_bias`` (``-inf`` outside the branch's
       region) restricts self-attention to same-partition keys, and
       ``query_mask`` zero-gates the residual updates on out-of-partition
       tokens. Optional spatial / temporal dilation (``routing_dilation``,
       ``routing_temporal_dilation``) softens the partition boundary by
       dilating each region; both the key set and the query set of each
       branch use the dilated region, but fusion still uses the un-dilated
       hard mask so every output token comes from exactly one branch. RoPE,
       causal mask, chunk structure, and batch shape are unchanged.
    4. Per-token hard fusion with the un-dilated partition:
       ``x = m · x_world + (1 - m) · x_ego`` (m in {0, 1}).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from wem.utils.masks import resize_soft_mask
from .mask_head import DPTMaskHead
from .primitives import (
    Head,
    WanCrossAttention,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)


def build_causal_mask(
    num_sink_frames: int,
    num_cond_frames: int,
    num_chunk_frames: int,
    num_chunks: int,
    h_patches: int,
    w_patches: int,
    device: torch.device,
) -> torch.Tensor:
    tokens_per_frame = h_patches * w_patches
    tokens_sink = num_sink_frames * tokens_per_frame
    tokens_cond = num_cond_frames * tokens_per_frame
    tokens_chunk = num_chunk_frames * tokens_per_frame
    total = tokens_sink + tokens_cond + tokens_chunk * num_chunks

    mask = torch.zeros(total, total, device=device, dtype=torch.bool)
    if tokens_sink > 0:
        mask[:tokens_sink, :tokens_sink] = 1

    cond_start = tokens_sink
    cond_end = cond_start + tokens_cond
    if tokens_cond > 0:
        mask[cond_start:cond_end, cond_start:cond_end] = 1

    for i in range(num_chunks):
        start = cond_end + i * tokens_chunk
        end = start + tokens_chunk
        if tokens_sink > 0:
            mask[start:end, :tokens_sink] = 1
        if tokens_cond > 0:
            mask[start:end, cond_start:cond_end] = 1
        mask[start:end, cond_end:end] = 1
    return mask


class WanARSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        num_sink_frames: int = 0,
        num_cond_frames: int = 1,
        num_chunk_frames: int = 20,
        num_chunks: int = 2,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.window_size = window_size

        self.num_sink_frames = num_sink_frames
        self.num_cond_frames = num_cond_frames
        self.num_chunk_frames = num_chunk_frames
        self.num_chunks = num_chunks

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self._causal_mask_cache: Dict[Tuple[int, int, str, int], torch.Tensor] = {}

    def _causal_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device.type, -1 if device.index is None else device.index)
        cached = self._causal_mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        mask = build_causal_mask(
            num_sink_frames=self.num_sink_frames,
            num_cond_frames=self.num_cond_frames,
            num_chunk_frames=self.num_chunk_frames,
            num_chunks=self.num_chunks,
            h_patches=h,
            w_patches=w,
            device=device,
        )
        self._causal_mask_cache[key] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        key_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, L, C)
            seq_lens:  (B,) valid token count per sample
            grid_sizes:(B, 3) = (F_patches, H_patches, W_patches)
            freqs:     RoPE freqs
            key_bias:  optional (B, L) additive per-key bias (in log-space).
                       ``log(m)`` for world branch, ``log(1 - m)`` for ego.
        """
        target_dtype = self.q.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        if q.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        _, h, w = grid_sizes[0].tolist()
        causal = self._causal_mask(h, w, x.device)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = torch.zeros_like(causal, dtype=q.dtype)
        attn_mask.masked_fill_(causal == 0, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        if key_bias is not None:
            if key_bias.dtype != q.dtype:
                key_bias = key_bias.to(dtype=q.dtype)
            attn_mask = attn_mask + key_bias[:, None, None, :]  # broadcast to (B, 1, L, L)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).flatten(2)
        return self.o(out)


class WanARAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        num_sink_frames: int = 0,
        num_cond_frames: int = 1,
        num_chunk_frames: int = 10,
        num_chunks: int = 2,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        state_attn_norm: bool = False,
        eps: float = 1e-6,
        layer_idx: int = -1,
        use_state_conditioning: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.layer_idx = layer_idx
        self.use_state_conditioning = use_state_conditioning

        self.num_sink_frames = num_sink_frames
        self.num_cond_frames = num_cond_frames
        self.num_chunk_frames = num_chunk_frames
        self.num_chunks = num_chunks

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanARSelfAttention(
            dim, num_heads, window_size, num_sink_frames, num_cond_frames, num_chunk_frames, num_chunks, qk_norm, eps
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        if use_state_conditioning:
            self.norm_world_state_1 = (
                WanLayerNorm(dim, eps, elementwise_affine=True) if state_attn_norm else nn.Identity()
            )
            self.world_state_cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
            self.world_state_ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_dim, dim),
            )
            self.norm_world_state_2 = WanLayerNorm(dim, eps)

            self.norm_ego_state_1 = (
                WanLayerNorm(dim, eps, elementwise_affine=True) if state_attn_norm else nn.Identity()
            )
            self.ego_state_cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
            self.ego_state_ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_dim, dim),
            )
            self.norm_ego_state_2 = WanLayerNorm(dim, eps)

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        context_lens: Optional[torch.Tensor],
        world_state: Optional[torch.Tensor],
        world_state_lens: Optional[torch.Tensor],
        ego_state: Optional[torch.Tensor],
        ego_state_lens: Optional[torch.Tensor],
        key_bias: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        branch_query_mask = None
        if query_mask is not None:
            branch_query_mask = query_mask.to(device=x.device, dtype=x.dtype)

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            key_bias=key_bias,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            self_attn_update = y * e[2].squeeze(2)
            if branch_query_mask is not None:
                self_attn_update = self_attn_update * branch_query_mask
            x = x + self_attn_update

        if self.use_state_conditioning and world_state is not None:
            z = self.world_state_cross_attn(self.norm_world_state_1(x), world_state, world_state_lens)
            world_state_ffn_out = self.world_state_ffn(self.norm_world_state_2(z))
            if branch_query_mask is not None:
                world_state_ffn_out = world_state_ffn_out * branch_query_mask
        else:
            world_state_ffn_out = 0.0

        if self.use_state_conditioning and ego_state is not None:
            z = self.ego_state_cross_attn(self.norm_ego_state_1(x), ego_state, ego_state_lens)
            ego_state_ffn_out = self.ego_state_ffn(self.norm_ego_state_2(z))
            if branch_query_mask is not None:
                ego_state_ffn_out = ego_state_ffn_out * branch_query_mask
        else:
            ego_state_ffn_out = 0.0

        x_norm = self.norm3(x)
        if isinstance(context, tuple) and len(context) == 2:
            context_prev, context_curr = context
            h_p = int(grid_sizes[0, 1].item())
            w_p = int(grid_sizes[0, 2].item())
            tokens_per_frame = h_p * w_p
            split_tokens = (self.num_sink_frames + self.num_cond_frames + self.num_chunk_frames) * tokens_per_frame
            split_tokens = max(1, min(split_tokens, x_norm.size(1) - 1))
            cross = torch.cat(
                [
                    self.cross_attn(x_norm[:, :split_tokens, :], context_prev, context_lens),
                    self.cross_attn(x_norm[:, split_tokens:, :], context_curr, context_lens),
                ],
                dim=1,
            )
        else:
            cross = self.cross_attn(x_norm, context, context_lens)
        if branch_query_mask is not None:
            cross = cross * branch_query_mask

        x = x + cross + world_state_ffn_out + ego_state_ffn_out

        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            ffn_update = y * e[5].squeeze(2)
            if branch_query_mask is not None:
                ffn_update = ffn_update * branch_query_mask
            x = x + ffn_update
        return x


class WanARModel(ModelMixin, ConfigMixin):
    ignore_for_config = ["dtype"]
    _no_split_modules = ["WanARAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type: str = "ti2v",
        enc_layers: int = 6,
        num_sink_frames: int = 0,
        num_cond_frames: int = 1,
        num_chunk_frames: int = 10,
        num_chunks: int = 2,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 48,
        dim: int = 3072,
        ffn_dim: int = 14336,
        freq_dim: int = 256,
        text_dim: int = 4096,
        state_dim: int = 2048,
        out_dim: int = 48,
        num_heads: int = 24,
        num_layers: int = 30,
        window_size: Tuple[int, int] = (-1, -1),
        mask_pred_layers: Optional[List[int]] = None,
        mask_fusion_channels: int = 256,
        mask_num_heads: int = 8,
        routing_threshold: float = 0.5,
        routing_dilation: int = 1,
        routing_temporal_dilation: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        state_attn_norm: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        use_dual_branch: bool = True,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        assert 1 <= enc_layers <= num_layers, (
            "This architecture requires at least one encoder layer to tap from."
        )

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.enc_layers = enc_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_dual_branch = use_dual_branch
        self.routing_threshold = float(routing_threshold)
        self.routing_dilation = int(max(0, routing_dilation))
        self.routing_temporal_dilation = int(max(0, routing_temporal_dilation))

        self.num_sink_frames = num_sink_frames
        self.num_cond_frames = num_cond_frames
        self.num_chunk_frames = num_chunk_frames
        self.num_chunks = num_chunks

        # Default tap indices: evenly spaced across the encoder stack, always
        # including the final encoder layer output.
        if mask_pred_layers is None:
            num_taps = min(4, enc_layers)
            mask_pred_layers = list(
                sorted(set(round((i + 1) * enc_layers / num_taps) - 1 for i in range(num_taps)))
            )
        assert all(0 <= i < enc_layers for i in mask_pred_layers), (
            f"mask_pred_layers {mask_pred_layers} out of range for enc_layers={enc_layers}"
        )
        self.mask_pred_layers = list(mask_pred_layers)

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        def _make_block(layer_idx: int, with_state: bool = True) -> WanARAttentionBlock:
            return WanARAttentionBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                num_sink_frames=num_sink_frames,
                num_cond_frames=num_cond_frames,
                num_chunk_frames=num_chunk_frames,
                num_chunks=num_chunks,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                state_attn_norm=state_attn_norm,
                eps=eps,
                layer_idx=layer_idx,
                use_state_conditioning=with_state,
            )

        num_dec = num_layers - enc_layers

        self.enc_blocks = nn.ModuleList(
            [_make_block(i, with_state=use_dual_branch) for i in range(enc_layers)]
        )

        if use_dual_branch:
            self.null_world_state_embed = nn.Parameter(torch.zeros(1, 1, state_dim))
            self.null_ego_state_embed = nn.Parameter(torch.zeros(1, 1, state_dim))
            self.world_state_embedding = nn.Sequential(
                nn.Linear(state_dim, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
            self.ego_state_embedding = nn.Sequential(
                nn.Linear(state_dim, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
            self.world_dec_blocks = nn.ModuleList([_make_block(i, with_state=True) for i in range(num_dec)])
            self.ego_dec_blocks = nn.ModuleList([_make_block(i, with_state=True) for i in range(num_dec)])
            self.mask_head = DPTMaskHead(
                in_channels=dim,
                fusion_channels=mask_fusion_channels,
                num_taps=len(self.mask_pred_layers),
                num_heads=mask_num_heads,
                qk_norm=qk_norm,
                eps=eps,
            )
        else:
            self.dec_blocks = nn.ModuleList([_make_block(i, with_state=False) for i in range(num_dec)])

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        self.init_weights()
        if dtype is not None:
            self.to(dtype=dtype)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        if self.use_dual_branch:
            dec_blocks_all = list(self.world_dec_blocks) + list(self.ego_dec_blocks)
        else:
            dec_blocks_all = list(self.dec_blocks)
        all_blocks = list(self.enc_blocks) + dec_blocks_all
        for block in all_blocks:
            if hasattr(block, "world_state_ffn"):
                nn.init.zeros_(block.world_state_ffn[-1].weight)
                if block.world_state_ffn[-1].bias is not None:
                    nn.init.zeros_(block.world_state_ffn[-1].bias)
            if hasattr(block, "ego_state_ffn"):
                nn.init.zeros_(block.ego_state_ffn[-1].weight)
                if block.ego_state_ffn[-1].bias is not None:
                    nn.init.zeros_(block.ego_state_ffn[-1].bias)
            if hasattr(block, "world_state_cross_attn"):
                nn.init.zeros_(block.world_state_cross_attn.o.weight)
                if block.world_state_cross_attn.o.bias is not None:
                    nn.init.zeros_(block.world_state_cross_attn.o.bias)
            if hasattr(block, "ego_state_cross_attn"):
                nn.init.zeros_(block.ego_state_cross_attn.o.weight)
                if block.ego_state_cross_attn.o.bias is not None:
                    nn.init.zeros_(block.ego_state_cross_attn.o.bias)

        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

        # Re-zero ego-conditioning output projections (init_weights xavier
        # would overwrite them otherwise).
        if self.use_dual_branch:
            for cond in self.mask_head.cond_blocks:
                nn.init.zeros_(cond.attn.o.weight)
                if cond.attn.o.bias is not None:
                    nn.init.zeros_(cond.attn.o.bias)

    def expand_token_timesteps(
        self,
        timesteps: torch.Tensor,
        grid_sizes: torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:
        expected_cols = 4 if self.num_sink_frames > 0 else 3
        assert timesteps.dim() == 2 and timesteps.size(1) == expected_cols
        B = timesteps.size(0)
        token_ts = []
        for b in range(B):
            Fp, Hp, Wp = grid_sizes[b].tolist()
            tokens_per_frame = Hp * Wp
            T = Fp * self.patch_size[0]
            expected = self.num_sink_frames + self.num_cond_frames + self.num_chunk_frames * self.num_chunks
            assert T == expected, f"Frame count mismatch: {T} vs {expected}"

            ts_b = []
            t_idx = 0
            if self.num_sink_frames > 0:
                for _ in range(self.num_sink_frames):
                    ts_b.append(timesteps[b, t_idx].expand(tokens_per_frame))
                t_idx += 1
            for _ in range(self.num_cond_frames):
                ts_b.append(timesteps[b, t_idx].expand(tokens_per_frame))
            t_idx += 1
            if self.num_chunks > 0:
                for _ in range(self.num_chunk_frames):
                    ts_b.append(timesteps[b, t_idx].expand(tokens_per_frame))
            t_idx += 1
            for _ in range(1, self.num_chunks):
                for _ in range(self.num_chunk_frames):
                    ts_b.append(timesteps[b, t_idx].expand(tokens_per_frame))
            token_ts.append(torch.cat(ts_b, dim=0))
        return torch.stack(token_ts, dim=0)

    def _dilate_region(self, region: torch.Tensor) -> torch.Tensor:
        """Apply configured spatial + temporal dilation to a 0/1 mask.

        ``region`` shape: (B, 1, Fp, Hp, Wp), float {0, 1}. Spatial dilation
        uses a (1, 3, 3) max-pool; temporal uses (3, 1, 1). Each is applied
        ``routing_dilation`` / ``routing_temporal_dilation`` times.
        """
        out = region
        for _ in range(self.routing_dilation):
            out = F.max_pool3d(out, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        for _ in range(self.routing_temporal_dilation):
            out = F.max_pool3d(out, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        return out

    def _build_key_bias(
        self,
        soft_mask: torch.Tensor,     # (B, 1, Fp, Hp, Wp)
        seq_len: int,
        seq_lens: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return five tensors used by the two decoder branches.

        Outputs:
            mask_tokens_strict  (B, L, 1) — un-dilated hard mask, used for
                                             final world/ego fusion.
            world_key_bias      (B, L)    — 0 on dilated world region, -inf
                                             elsewhere.
            ego_key_bias        (B, L)    — 0 on dilated ego region, -inf
                                             elsewhere.
            world_query_mask    (B, L, 1) — 1 on dilated world region, 0
                                             elsewhere (padding zeroed).
            ego_query_mask      (B, L, 1) — 1 on dilated ego region, 0
                                             elsewhere (padding zeroed).

        Dilation uses configured ``routing_dilation`` (spatial) and
        ``routing_temporal_dilation`` (temporal). Shared prefix frames
        (sink + cond) are forced valid in both branches.
        """
        B = soft_mask.shape[0]
        thr = self.routing_threshold
        mask_bin = (soft_mask >= thr).to(torch.float32)
        inv_mask_bin = 1.0 - mask_bin

        world_dilated = self._dilate_region(mask_bin)
        ego_dilated = self._dilate_region(inv_mask_bin)

        # Shared prefix frames are unsupervised context shared by both branches:
        #   - Both branches may read them as keys and update them as queries
        #     (world_dilated = ego_dilated = 1 on prefix positions), so each
        #     branch can condition on both the persistent sink and the latest
        #     local cond frame.
        #   - Strict mask_bin = 1 on prefix positions forces the hard fusion
        #     to pick x_world there. Those prefix outputs are not supervised and
        #     mainly serve as branch-specific context for later tokens.
        shared_patch_frames = (self.num_sink_frames + self.num_cond_frames) // self.patch_size[0]
        if shared_patch_frames > 0:
            mask_bin = mask_bin.clone()
            world_dilated = world_dilated.clone()
            ego_dilated = ego_dilated.clone()
            mask_bin[:, :, :shared_patch_frames, :, :] = 1.0
            world_dilated[:, :, :shared_patch_frames, :, :] = 1.0
            ego_dilated[:, :, :shared_patch_frames, :, :] = 1.0

        def _flatten_pad(m: torch.Tensor) -> torch.Tensor:
            m = m.flatten(2).transpose(1, 2)
            valid = m.size(1)
            if valid < seq_len:
                pad = m.new_zeros(B, seq_len - valid, 1)
                m = torch.cat([m, pad], dim=1)
            return m

        mask_strict_flat = _flatten_pad(mask_bin)
        world_flat = _flatten_pad(world_dilated)
        ego_flat = _flatten_pad(ego_dilated)

        neg_inf = torch.finfo(dtype).min
        zero_like = torch.zeros_like(world_flat)
        ninf_like = torch.full_like(world_flat, neg_inf)

        world_key_bias = torch.where(world_flat > 0.5, zero_like, ninf_like).squeeze(-1)
        ego_key_bias = torch.where(ego_flat > 0.5, zero_like, ninf_like).squeeze(-1)

        if seq_lens is not None:
            key_idx = torch.arange(seq_len, device=soft_mask.device)
            pad_mask = key_idx[None, :] >= seq_lens.to(soft_mask.device).view(-1, 1)  # (B, L)
            world_key_bias = world_key_bias.masked_fill(pad_mask, neg_inf)
            ego_key_bias = ego_key_bias.masked_fill(pad_mask, neg_inf)
            valid_queries = (~pad_mask).to(world_flat.dtype).unsqueeze(-1)
            world_flat = world_flat * valid_queries
            ego_flat = ego_flat * valid_queries

        return (
            mask_strict_flat.to(dtype=dtype),
            world_key_bias.to(dtype=dtype),
            ego_key_bias.to(dtype=dtype),
            world_flat.to(dtype=dtype),
            ego_flat.to(dtype=dtype),
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]],
        seq_len: int,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y: Optional[List[torch.Tensor]] = None,
        return_mask: bool = False,
        inspect_mode: bool = False,
        return_branches: bool = False,
        gt_mask: Optional[torch.Tensor] = None,
    ):
        device = self.patch_embedding.weight.device
        target_dtype = self.patch_embedding.weight.dtype
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.model_type == "i2v":
            assert y is not None
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        t = self.expand_token_timesteps(t, grid_sizes)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat)
                .unflatten(0, (bt, seq_len))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        def embed_context_list(ctx_list):
            return self.text_embedding(
                torch.stack(
                    [
                        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                        for u in ctx_list
                    ]
                )
            )

        context_lens = None
        if isinstance(context, dict):
            context = (embed_context_list(context["curr"]), embed_context_list(context["next"]))
        else:
            context = embed_context_list(context)

        batch_size = x.size(0)
        if self.use_dual_branch:
            if state is None:
                world_state_raw = self.null_world_state_embed.expand(batch_size, -1, -1).to(
                    device=device, dtype=target_dtype
                )
                ego_state_raw = self.null_ego_state_embed.expand(batch_size, -1, -1).to(
                    device=device, dtype=target_dtype
                )
            else:
                world_state_raw, ego_state_raw = state
            world_state = self.world_state_embedding(world_state_raw.to(device=device, dtype=target_dtype))
            ego_state = self.ego_state_embedding(ego_state_raw.to(device=device, dtype=target_dtype))
        else:
            world_state = None
            ego_state = None

        enc_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            world_state=world_state,
            world_state_lens=None,
            ego_state=ego_state,
            ego_state_lens=None,
            key_bias=None,
        )
        enc_taps: List[torch.Tensor] = []
        tap_set = set(self.mask_pred_layers) if self.use_dual_branch else set()
        for i, block in enumerate(self.enc_blocks):
            x = block(x, **enc_kwargs)
            if i in tap_set:
                enc_taps.append(x)

        if not self.use_dual_branch:
            dec_kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                world_state=None,
                world_state_lens=None,
                ego_state=None,
                ego_state_lens=None,
                key_bias=None,
            )
            for block in self.dec_blocks:
                x = block(x, **dec_kwargs)
            x = self.head(x, e)
            return self.unpatchify(x, grid_sizes)

        # Keep taps in the order specified by mask_pred_layers.
        order = {idx: pos for pos, idx in enumerate(sorted(self.mask_pred_layers))}
        enc_taps = [enc_taps[order[i]] for i in self.mask_pred_layers]

        Fp, Hp, Wp = [int(v) for v in grid_sizes[0].tolist()]
        mask_logits = self.mask_head(enc_taps, ego_state, Fp, Hp, Wp)

        if gt_mask is not None:
            fusion_mask = resize_soft_mask(gt_mask, (Fp, Hp, Wp))
        else:
            fusion_mask = torch.sigmoid(mask_logits)

        # Build per-branch attention key bias and query masks. Detached so
        # the mask is learned only from explicit mask supervision. key_bias
        # and query_mask both use the dilated region (symmetric boundary
        # overlap); mask_tokens stays un-dilated for hard fusion. Shared
        # prefix frames (sink + cond) are handled inside _build_key_bias
        # (valid in both branches, fusion still picks world).
        (
            mask_tokens,
            world_key_bias,
            ego_key_bias,
            world_query_mask,
            ego_query_mask,
        ) = self._build_key_bias(
            fusion_mask.detach(), seq_len, seq_lens, dtype=target_dtype
        )

        x_shared = x
        x_world = x_shared.clone()
        x_ego = x_shared.clone()

        world_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            world_state=world_state,
            world_state_lens=None,
            ego_state=ego_state,
            ego_state_lens=None,
            key_bias=world_key_bias,
            query_mask=world_query_mask,
        )
        ego_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            world_state=world_state,
            world_state_lens=None,
            ego_state=ego_state,
            ego_state_lens=None,
            key_bias=ego_key_bias,
            query_mask=ego_query_mask,
        )

        def _slice_batch(value, active_idx):
            if value is None:
                return None
            if isinstance(value, tuple):
                return tuple(
                    v.index_select(0, active_idx.to(v.device)) if torch.is_tensor(v) else v
                    for v in value
                )
            if torch.is_tensor(value):
                return value.index_select(0, active_idx.to(value.device))
            return value

        def _run_branch_blocks(
            x_branch: torch.Tensor,
            blocks: nn.ModuleList,
            branch_kwargs: Dict[str, Any],
        ) -> torch.Tensor:
            query_mask = branch_kwargs.get("query_mask")
            if query_mask is None:
                for block in blocks:
                    x_branch = block(x_branch, **branch_kwargs)
                return x_branch

            active_mask = (query_mask.squeeze(-1) > 0).any(dim=1)
            if not bool(active_mask.any().item()):
                return x_branch
            if bool(active_mask.all().item()):
                for block in blocks:
                    x_branch = block(x_branch, **branch_kwargs)
                return x_branch

            active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
            x_active = x_branch.index_select(0, active_idx)
            active_kwargs = {
                key: (_slice_batch(value, active_idx) if key != "freqs" else value)
                for key, value in branch_kwargs.items()
            }
            for block in blocks:
                x_active = block(x_active, **active_kwargs)

            x_out = x_branch.clone()
            x_out.index_copy_(0, active_idx, x_active)
            return x_out

        x_world = _run_branch_blocks(x_world, self.world_dec_blocks, world_kwargs)
        x_ego = _run_branch_blocks(x_ego, self.ego_dec_blocks, ego_kwargs)

        x = mask_tokens * x_world + (1.0 - mask_tokens) * x_ego
        x = self.head(x, e)
        out = self.unpatchify(x, grid_sizes)

        branch_outputs: Optional[Dict[str, List[torch.Tensor]]] = None
        if return_branches:
            branch_outputs = {
                "world": self.unpatchify(self.head(x_world, e), grid_sizes),
                "ego": self.unpatchify(self.head(x_ego, e), grid_sizes),
            }

        inspect_features: Optional[Dict[str, Any]] = None
        if inspect_mode:
            inspect_features = {
                "enc": x_shared,
                "enc_taps": enc_taps,
                "grid_sizes": grid_sizes,
                "world_state": world_state,
                "ego_state": ego_state,
            }

        extras = []
        if return_mask:
            extras.append(mask_logits)
        if inspect_mode:
            extras.append(inspect_features)
        if return_branches:
            extras.append(branch_outputs)
        return (out, *extras) if extras else out

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> List[torch.Tensor]:
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    @classmethod
    def init_from_wan(
        cls,
        checkpoint_dir: str,
        num_sink_frames: int = 0,
        num_cond_frames: int = 1,
        num_chunk_frames: int = 20,
        num_chunks: int = 2,
        enc_layers: int = 6,
        dtype: torch.dtype = torch.bfloat16,
        use_dual_branch: bool = False,
    ) -> "WanARModel":
        """Load Wan's DiT weights, remapping flat ``blocks.i`` to the split layout.

        Wan's ``blocks.i.*`` are mapped to:
          ``i < enc_layers``  →  ``enc_blocks.i.*``
          ``i >= enc_layers`` →  ``dec_blocks.(i - enc_layers).*``
        All other keys are matched by name. Shape mismatches are skipped.
        WEM-specific parameters (mask head, state embeddings, dual decoder) keep
        their random init and appear in missing_keys.
        """
        import json
        import os
        from safetensors import safe_open

        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            config = json.load(f)

        model = cls(
            model_type=config.get("model_type", "t2v"),
            enc_layers=enc_layers,
            num_sink_frames=num_sink_frames,
            num_cond_frames=num_cond_frames,
            num_chunk_frames=num_chunk_frames,
            num_chunks=num_chunks,
            patch_size=tuple(config.get("patch_size", (1, 2, 2))),
            text_len=config.get("text_len", 512),
            in_dim=config.get("in_dim", 48),
            dim=config.get("dim", 2048),
            ffn_dim=config.get("ffn_dim", 14336),
            freq_dim=config.get("freq_dim", 256),
            text_dim=config.get("text_dim", 4096),
            out_dim=config.get("out_dim", 48),
            num_heads=config.get("num_heads", 24),
            num_layers=config.get("num_layers", 30),
            qk_norm=config.get("qk_norm", True),
            cross_attn_norm=config.get("cross_attn_norm", True),
            eps=config.get("eps", 1e-6),
            dtype=dtype,
            use_dual_branch=use_dual_branch,
        )

        def _remap(key: str) -> str:
            parts = key.split(".")
            if parts[0] == "blocks" and len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                suffix = ".".join(parts[2:])
                if idx < enc_layers:
                    return f"enc_blocks.{idx}.{suffix}"
                else:
                    return f"dec_blocks.{idx - enc_layers}.{suffix}"
            return key

        index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        model_sd = model.state_dict()
        shard_to_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for wan_key, shard_name in weight_map.items():
            wem_key = _remap(wan_key)
            if wem_key in model_sd:
                shard_to_pairs[shard_name].append((wan_key, wem_key))

        load_sd: dict[str, torch.Tensor] = {}
        for shard_name, pairs in shard_to_pairs.items():
            shard_path = os.path.join(checkpoint_dir, shard_name)
            if not os.path.exists(shard_path):
                continue
            with safe_open(shard_path, framework="pt") as sf:
                for wan_key, wem_key in pairs:
                    tensor = sf.get_tensor(wan_key)
                    if model_sd[wem_key].shape == tensor.shape:
                        load_sd[wem_key] = tensor
                    else:
                        print(
                            f"Skipping {wan_key} → {wem_key}: "
                            f"shape {tuple(tensor.shape)} vs {tuple(model_sd[wem_key].shape)}"
                        )

        incompatible = model.load_state_dict(load_sd, strict=False)
        print(f"Loaded {len(load_sd)} / {len(weight_map)} Wan tensors")
        if incompatible.missing_keys:
            print(f"Missing keys ({len(incompatible.missing_keys)}):", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print(f"Unexpected keys ({len(incompatible.unexpected_keys)}):", incompatible.unexpected_keys)
        return model
