# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
from easydict import EasyDict
from .wan import wan_cfg as wan_base_cfg

_WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "checkpoints/Wan2.2-TI2V-5B")
_QWEN_CKPT_DIR = os.environ.get("QWEN_CKPT_DIR", "checkpoints/Qwen3-VL-2B-Instruct")

wem_cfg = EasyDict(wan_base_cfg)
wem_cfg.__name__ = 'Config: WEM Model'

wem_cfg.world_model = EasyDict()
wem_cfg.world_model.model_name = _QWEN_CKPT_DIR
wem_cfg.world_model.num_query_tokens = 256
wem_cfg.world_model.num_world_query_tokens = 192
wem_cfg.world_model.num_ego_query_tokens = 64
wem_cfg.world_model.ego_recent_turns = 1
wem_cfg.world_model.freeze_backbone = True
wem_cfg.world_model.dtype = torch.bfloat16
wem_cfg.world_model.device = "cuda"
wem_cfg.world_model.ckpt_path = None
wem_cfg.world_model.use_lora = False

wem_cfg.wan_decoder = EasyDict()
wem_cfg.wan_decoder.model_type = 'ti2v'
wem_cfg.wan_decoder.num_sink_frames = 1
wem_cfg.wan_decoder.num_cond_frames = 1
wem_cfg.wan_decoder.num_chunk_frames = 10
wem_cfg.wan_decoder.num_chunks = 2
wem_cfg.wan_decoder.patch_size = wem_cfg.patch_size
wem_cfg.wan_decoder.text_len = wem_cfg.text_len
wem_cfg.wan_decoder.in_dim = 48
wem_cfg.wan_decoder.dim = wem_cfg.dim
wem_cfg.wan_decoder.ffn_dim = wem_cfg.ffn_dim
wem_cfg.wan_decoder.freq_dim = wem_cfg.freq_dim
wem_cfg.wan_decoder.text_dim = 4096
wem_cfg.wan_decoder.state_dim = 2048
wem_cfg.wan_decoder.trd_align_dim = 768
wem_cfg.wan_decoder.trd_projector_dim = 2048
wem_cfg.wan_decoder.out_dim = 48
wem_cfg.wan_decoder.num_heads = wem_cfg.num_heads
wem_cfg.wan_decoder.num_layers = wem_cfg.num_layers
wem_cfg.wan_decoder.enc_layers = 24

wem_cfg.wan_decoder.mask_pred_layers = [5, 9, 13, 17, 21, 23]
wem_cfg.wan_decoder.mask_fusion_channels = 384
wem_cfg.wan_decoder.mask_num_heads = 12
wem_cfg.wan_decoder.routing_threshold = 0.6
# Spatial / temporal dilation (in rings) applied to both world and ego
# branch regions so each branch's key set and query set span a thin overlap
# zone around the predicted boundary. Final fusion still uses the un-dilated
# hard mask.
wem_cfg.wan_decoder.routing_dilation = 1
wem_cfg.wan_decoder.routing_temporal_dilation = 0
wem_cfg.wan_decoder.window_size = wem_cfg.window_size
wem_cfg.wan_decoder.qk_norm = wem_cfg.qk_norm
wem_cfg.wan_decoder.cross_attn_norm = wem_cfg.cross_attn_norm
wem_cfg.wan_decoder.state_attn_norm = True
wem_cfg.wan_decoder.eps = wem_cfg.eps
wem_cfg.wan_decoder.dtype = wem_cfg.param_dtype
wem_cfg.wan_decoder.param_dtype = wem_cfg.param_dtype
wem_cfg.wan_decoder.num_train_timesteps = wem_cfg.num_train_timesteps
wem_cfg.wan_decoder.vae_stride = wem_cfg.vae_stride
wem_cfg.wan_decoder.sample_neg_prompt = wem_cfg.sample_neg_prompt
wem_cfg.wan_decoder.sink_noise_sigma = 0.0

wem_cfg.dtype = torch.bfloat16
wem_cfg.device = "cuda"
wem_cfg.cache_dir = None
