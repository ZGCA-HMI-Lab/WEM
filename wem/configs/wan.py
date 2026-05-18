# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
from easydict import EasyDict

_WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "checkpoints/Wan2.2-TI2V-5B")

wan_cfg = EasyDict(__name__='Config: Wan TI2V 5B')

wan_cfg.t5_model = 'umt5_xxl'
wan_cfg.t5_dtype = torch.bfloat16
wan_cfg.text_len = 512
wan_cfg.t5_checkpoint = os.path.join(_WAN_CKPT_DIR, 'models_t5_umt5-xxl-enc-bf16.pth')
wan_cfg.t5_tokenizer = os.path.join(_WAN_CKPT_DIR, 'google/umt5-xxl')

wan_cfg.vae_checkpoint = os.path.join(_WAN_CKPT_DIR, 'Wan2.2_VAE.pth')
wan_cfg.vae_stride = (4, 16, 16)

wan_cfg.param_dtype = torch.bfloat16
wan_cfg.patch_size = (1, 2, 2)
wan_cfg.dim = 3072
wan_cfg.ffn_dim = 14336
wan_cfg.freq_dim = 256
wan_cfg.num_heads = 24
wan_cfg.num_layers = 30
wan_cfg.window_size = (-1, -1)
wan_cfg.qk_norm = True
wan_cfg.cross_attn_norm = True
wan_cfg.eps = 1e-6

wan_cfg.num_train_timesteps = 1000
wan_cfg.sample_fps = 24
wan_cfg.sample_neg_prompt = (
    'Vivid colors, overexposed, static, blurry details, subtitles, style, '
    'artwork, painting, still, overall gray, worst quality, low quality, '
    'JPEG compression artifacts, ugly, incomplete, extra fingers, '
    'poorly drawn hands, poorly drawn faces, deformed, disfigured, '
    'deformed limbs, fused fingers, static frame, cluttered background, '
    'three legs, many people in background, walking backwards'
)
wan_cfg.frame_num = 121
wan_cfg.sample_shift = 5.0
wan_cfg.sample_steps = 50
wan_cfg.sample_guide_scale = 5.0
