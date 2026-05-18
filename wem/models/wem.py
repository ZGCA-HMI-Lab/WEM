"""Top-level WEMModel = Qwen3 world model + attention-routed WanAR DiT."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from easydict import EasyDict

from .wan_ar import WanARModel
from .world_model import Qwen3WorldModel


class WEMModel(nn.Module):
    def __init__(self, config: EasyDict):
        super().__init__()
        wm_cfg = config.world_model
        dec_cfg = config.wan_decoder

        self.world_model = Qwen3WorldModel(
            model_name=wm_cfg.model_name,
            num_query_tokens=wm_cfg.num_query_tokens,
            num_world_query_tokens=getattr(wm_cfg, "num_world_query_tokens", None),
            num_ego_query_tokens=getattr(wm_cfg, "num_ego_query_tokens", None),
            ego_recent_turns=getattr(wm_cfg, "ego_recent_turns", 3),
            freeze_backbone=wm_cfg.freeze_backbone,
            device=wm_cfg.device,
            dtype=wm_cfg.dtype,
            cache_dir=config.cache_dir,
            use_device_map=False,
            use_lora=wm_cfg.use_lora,
        )
        self.wan_decoder = WanARModel(
            model_type=dec_cfg.model_type,
            enc_layers=getattr(dec_cfg, "enc_layers", 6),
            num_sink_frames=getattr(dec_cfg, "num_sink_frames", 0),
            num_cond_frames=dec_cfg.num_cond_frames,
            num_chunk_frames=dec_cfg.num_chunk_frames,
            num_chunks=dec_cfg.num_chunks,
            patch_size=dec_cfg.patch_size,
            text_len=dec_cfg.text_len,
            in_dim=dec_cfg.in_dim,
            dim=dec_cfg.dim,
            ffn_dim=dec_cfg.ffn_dim,
            freq_dim=dec_cfg.freq_dim,
            text_dim=dec_cfg.text_dim,
            state_dim=dec_cfg.state_dim,
            out_dim=dec_cfg.out_dim,
            num_heads=dec_cfg.num_heads,
            num_layers=dec_cfg.num_layers,
            window_size=getattr(dec_cfg, "window_size", (-1, -1)),
            mask_pred_layers=getattr(dec_cfg, "mask_pred_layers", None),
            mask_fusion_channels=getattr(dec_cfg, "mask_fusion_channels", 256),
            mask_num_heads=getattr(dec_cfg, "mask_num_heads", 8),
            routing_threshold=getattr(dec_cfg, "routing_threshold", 0.5),
            routing_dilation=getattr(dec_cfg, "routing_dilation", 1),
            routing_temporal_dilation=getattr(dec_cfg, "routing_temporal_dilation", 0),
            qk_norm=dec_cfg.qk_norm,
            cross_attn_norm=dec_cfg.cross_attn_norm,
            state_attn_norm=dec_cfg.state_attn_norm,
            eps=dec_cfg.eps,
            dtype=dec_cfg.dtype,
        )

    def forward(
        self,
        ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        visual_embeds: Optional[torch.Tensor] = None,
        token_turn_ids: Optional[torch.Tensor] = None,
        x: Optional[List[torch.Tensor]] = None,
        t: Optional[torch.Tensor] = None,
        context=None,
        seq_len: Optional[int] = None,
        y: Optional[List[torch.Tensor]] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode: str = "full",
        inspect_mode: bool = False,
        return_mask: bool = False,
        return_branches: bool = False,
        gt_mask: Optional[torch.Tensor] = None,
    ):
        if mode == "wm":
            return self.world_model(ids, mask, visual_embeds, token_turn_ids)

        if mode == "decoder":
            return self.wan_decoder(
                x=x,
                t=t,
                context=context,
                seq_len=seq_len,
                state=state,
                y=y,
                return_mask=return_mask,
                inspect_mode=inspect_mode,
                return_branches=return_branches,
                gt_mask=gt_mask,
            )

        world_state, ego_state = self.world_model(ids, mask, visual_embeds, token_turn_ids)
        return self.wan_decoder(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            state=(world_state, ego_state),
            y=y,
            return_mask=return_mask,
            inspect_mode=inspect_mode,
            return_branches=return_branches,
            gt_mask=gt_mask,
        )
