# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import math
import numpy as np
import os
from pathlib import Path
import random
import sys
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
from PIL import Image
from tqdm import tqdm

from third_party.wan.distributed.fsdp import shard_model
from third_party.wan.distributed.util import get_world_size
from third_party.wan.modules.t5 import T5EncoderModel
from third_party.wan.modules.vae2_2 import Wan2_2_VAE
from third_party.wan.utils.utils import best_output_size, save_video

from wem.models.wan_ar import WanARModel
from wem.models.world_model import Qwen3WorldModel
from wem.models.wem import WEMModel
from transformers import AutoProcessor
from wem.utils.masks import resize_soft_mask
from wem.utils.chat_template import QWEN_CHAT_TOKENS


def _remap_legacy_branch_keys(state_dict: dict) -> dict:
    """Rename old nav/manip branch keys to world/ego."""
    return {
        k.replace("nav_", "world_").replace("manip_", "ego_"): v
        for k, v in state_dict.items()
    }


def _noise_with_sigma(
    clean: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Flow-matching noise: x = (1 - sigma) * clean + sigma * noise."""
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    return (1 - sigma) * clean + sigma * noise


class WanARGenerator:
    """Stage-1 AR video generator (no world model, single decoder branch)."""

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=config.t5_checkpoint,
            tokenizer_path=config.t5_tokenizer,
            shard_fn=shard_fn if t5_fsdp else None,
        )
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(vae_pth=config.vae_checkpoint, device=self.device)

        print(f"Creating WanARModel from {checkpoint_dir}")
        self.model = WanARModel(
            model_type=getattr(config, "model_type", "ti2v"),
            enc_layers=getattr(config, "enc_layers", 24),
            num_sink_frames=getattr(config, "num_sink_frames", 0),
            num_cond_frames=getattr(config, "num_cond_frames", 1),
            num_chunk_frames=getattr(config, "num_chunk_frames", 10),
            num_chunks=getattr(config, "num_chunks", 2),
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=getattr(config, "in_dim", 48),
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=getattr(config, "text_dim", 4096),
            out_dim=getattr(config, "out_dim", 48),
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            dtype=config.param_dtype,
            use_dual_branch=False,
        )

        if os.path.isdir(checkpoint_dir):
            safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                print(f"Loading WanARModel weights from {safetensors_path}")
                state_dict = load_file(safetensors_path)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {missing}")
                print(f"Unexpected keys: {unexpected}")

        self.model.eval().requires_grad_(False)
        if not self.init_on_cpu:
            self.model.to(self.device)
        if dit_fsdp:
            self.model = shard_fn(self.model)
        elif convert_model_dtype:
            self.model.to(self.param_dtype)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        input_prompts: Union[str, List[str]],
        img,
        max_area: int = 480 * 480,
        window_size: int = 21,
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ) -> Optional[torch.Tensor]:
        if isinstance(input_prompts, str):
            input_prompts = [input_prompts]

        ih, iw = img.height, img.width
        dh = self.patch_size[1] * self.vae_stride[1]
        dw = self.patch_size[2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1, y1 = (img.width - ow) // 2, (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        sink_frames = int(getattr(self.model, "num_sink_frames", 0))
        cond_frames = int(getattr(self.model, "num_cond_frames", 1))
        chunk_latents = (window_size - cond_frames) // 2
        prefix_frames = sink_frames + cond_frames
        latent_h = oh // self.vae_stride[1]
        latent_w = ow // self.vae_stride[2]
        latent_t = sink_frames + window_size
        seq_len = latent_t * latent_h * latent_w // (self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            contexts = [self.text_encoder([p], self.device) for p in input_prompts]
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_null = [t.to(self.device) for t in self.text_encoder([n_prompt], torch.device("cpu"))]
            contexts = [
                [t.to(self.device) for t in self.text_encoder([p], torch.device("cpu"))]
                for p in input_prompts
            ]

        z_first = self.vae.encode([img_tensor])[0]
        sink_latent = z_first[:, :sink_frames] if sink_frames > 0 else None
        cond_latent = z_first[:, -cond_frames:]
        cond_latent_noised = _noise_with_sigma(cond_latent, sigma=0.055, generator=seed_g)

        sigmas_prev = torch.linspace(0.5, 0.0, sampling_steps + 1, device=self.device)
        sigmas_curr = torch.linspace(1.0, 0.5, sampling_steps + 1, device=self.device)
        timesteps_prev = sigmas_prev * 1000
        timesteps_curr = sigmas_curr * 1000

        def _make_timestep(ts_prev, ts_curr):
            cols = 4 if sink_frames > 0 else 3
            ts = torch.zeros(1, cols, device=self.device)
            if sink_frames > 0:
                ts[0, 0] = 1.0
                ts[0, 1] = 55.0
                ts[0, 2] = ts_prev
                ts[0, 3] = ts_curr
            else:
                ts[0, 0] = 55.0
                ts[0, 1] = ts_prev
                ts[0, 2] = ts_curr
            return ts

        def _build_x(prev, curr):
            parts = ([sink_latent] if sink_frames > 0 else []) + [cond_latent_noised, prev, curr]
            return torch.cat(parts, dim=1)

        all_decoded = []
        chunk0 = torch.randn(
            self.vae.model.z_dim, chunk_latents, latent_h, latent_w,
            dtype=torch.float32, generator=seed_g, device=self.device,
        )

        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            if self.rank == 0:
                all_decoded.append(img_tensor)

            # Bootstrap: denoise chunk0 from sigma=1.0 to 0.5
            ctx_boot = contexts[0]
            for t in tqdm(range(sampling_steps), desc="bootstrap"):
                ts = _make_timestep(timesteps_curr[t], timesteps_curr[t])
                x_full = _build_x(chunk0, chunk0)
                ctx_d = {"curr": ctx_boot, "next": ctx_boot}
                null_d = {"curr": context_null, "next": context_null}
                pred_cond = self.model(x=[x_full], t=ts, context=ctx_d, seq_len=seq_len)[0]
                pred_null = self.model(x=[x_full], t=ts, context=null_d, seq_len=seq_len)[0]
                pred = pred_null + guide_scale * (pred_cond - pred_null)
                chunk0 = chunk0 + (sigmas_curr[t + 1] - sigmas_curr[t]) * pred[
                    :, prefix_frames:prefix_frames + chunk_latents
                ]

            prev = chunk0
            for action_idx in range(len(input_prompts)):
                ctx_curr = contexts[action_idx]
                ctx_next = contexts[action_idx + 1] if action_idx + 1 < len(contexts) else ctx_curr
                curr = torch.randn(
                    self.vae.model.z_dim, chunk_latents, latent_h, latent_w,
                    dtype=torch.float32, generator=seed_g, device=self.device,
                )
                for t in tqdm(range(sampling_steps), desc=f"Chunk {action_idx}"):
                    ts = _make_timestep(timesteps_prev[t], timesteps_curr[t])
                    x_full = _build_x(prev, curr)
                    ctx_d = {"curr": ctx_curr, "next": ctx_next}
                    null_d = {"curr": context_null, "next": context_null}
                    pred_cond = self.model(x=[x_full], t=ts, context=ctx_d, seq_len=seq_len)[0]
                    pred_null = self.model(x=[x_full], t=ts, context=null_d, seq_len=seq_len)[0]
                    pred = pred_null + guide_scale * (pred_cond - pred_null)
                    prev = prev + (sigmas_prev[t + 1] - sigmas_prev[t]) * pred[
                        :, prefix_frames:prefix_frames + chunk_latents
                    ]
                    curr = curr + (sigmas_curr[t + 1] - sigmas_curr[t]) * pred[
                        :, prefix_frames + chunk_latents:prefix_frames + 2 * chunk_latents
                    ]

                if self.rank == 0:
                    all_decoded.append(self.vae.decode([prev])[0])
                cond_latent = prev[:, -cond_frames:]
                cond_latent_noised = _noise_with_sigma(cond_latent, sigma=0.055, generator=seed_g)
                prev = curr

        if offload_model:
            self.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()
        if self.rank != 0:
            return None
        return torch.cat(all_decoded, dim=1)


class WEMGenerator:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.wan_decoder.num_train_timesteps
        self.param_dtype = config.wan_decoder.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=config.t5_checkpoint,
            tokenizer_path=config.t5_tokenizer,
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.wan_decoder.vae_stride
        self.patch_size = config.wan_decoder.patch_size
        self.vae = Wan2_2_VAE(vae_pth=config.vae_checkpoint, device=self.device)

        print(f"Creating WEMModel from {checkpoint_dir}")
        self.model = WEMModel(config)

        if os.path.isdir(checkpoint_dir):
            safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                print(f"Loading WEMModel weights from {safetensors_path}")
                wem_state_dict = _remap_legacy_branch_keys(load_file(safetensors_path))
                missing, unexpected = self.model.load_state_dict(wem_state_dict, strict=False)
                print(f"Missing keys: {missing}")
                print(f"Unexpected keys: {unexpected}")
                self.model.eval().requires_grad_(False)

        if not self.init_on_cpu:
            self.model.to(self.device)

        self.model.wan_decoder = self._configure_wan_decoder(
            model=self.model.wan_decoder,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
        )
        self.model.world_model = self._configure_world_model(
            model=self.model.world_model,
            merge_lora=True,
        )

        self.sp_size = get_world_size() if use_sp else 1
        self.sample_neg_prompt = config.wan_decoder.sample_neg_prompt
        self.processor = AutoProcessor.from_pretrained(config.world_model.model_name)

    def _configure_world_model(
        self,
        model: Qwen3WorldModel,
        merge_lora: bool = True,
    ) -> Qwen3WorldModel:
        model.eval().requires_grad_(False)
        if model.use_lora and merge_lora:
            lm = model.backbone.model.language_model
            model.backbone.model.language_model = lm.merge_and_unload()
            model.use_lora = False
            print("LoRA weights merged into base model.")
        if dist.is_initialized():
            dist.barrier()
        if not self.init_on_cpu:
            model.to(self.device)
        return model

    def _configure_wan_decoder(self, model, dit_fsdp, shard_fn, convert_model_dtype):
        model.eval().requires_grad_(False)
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)
        return model

    def _encode_visual(
        self,
        visual: torch.Tensor,
        is_image: bool,
        cache: Optional[Dict[int, torch.Tensor]],
        cache_key: int,
    ) -> torch.Tensor:
        """Run the ViT on a single image or video, using cache to skip recomputation."""
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        visual_dtype = self.model.world_model.backbone.visual.dtype
        if is_image:
            inputs = self.processor.image_processor(images=[visual], return_tensors="pt")
            pvals = inputs.pixel_values.to(self.device, dtype=visual_dtype)
            gthw = inputs.image_grid_thw.to(self.device) if hasattr(inputs, "image_grid_thw") else None
            with torch.no_grad():
                emb = self.model.world_model.backbone.visual(pvals, grid_thw=gthw)
        else:
            if visual.ndim == 4 and visual.shape[0] == 3:
                visual = visual.permute(1, 0, 2, 3)
            inputs = self.processor.video_processor(
                videos=[visual],
                return_tensors="pt",
                video_metadata=[{"fps": 16.0, "total_num_frames": visual.shape[0]}],
            )
            if "pixel_values_videos" in inputs:
                vpvals = inputs.pixel_values_videos
            elif "pixel_values" in inputs:
                vpvals = inputs.pixel_values
            else:
                raise ValueError(f"Unexpected video processor keys: {list(inputs.keys())}")
            vpvals = vpvals.to(self.device, dtype=visual_dtype)
            vgthw = inputs.video_grid_thw.to(self.device) if "video_grid_thw" in inputs else None
            with torch.no_grad():
                emb = self.model.world_model.backbone.visual(vpvals, grid_thw=vgthw)

        if isinstance(emb, tuple):
            emb = emb[0]
        if emb.dim() == 3:
            emb = emb.squeeze(0)

        if cache is not None:
            cache[cache_key] = emb
        return emb

    def _get_current_state(
        self,
        visuals: List[torch.Tensor],
        texts: List[str],
        round_limit: int,
        visual_cache: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build world-model input from visual/text history and return (world_state, ego_state).

        visual_cache maps absolute visual index → cached ViT embedding.  Pass the same dict
        across successive calls within one generate() invocation to avoid re-encoding prior
        frames.  The first frame (index 0) and all decoded videos (index 1..k) are stable once
        produced, so their embeddings are cached on first access and reused on every later call.
        """
        T = QWEN_CHAT_TOKENS
        num_texts = len(texts)
        start_idx = max(0, num_texts - round_limit)

        input_ids_list = []
        turn_ids_list = []
        visual_embeds_list = []

        input_ids_list.append(torch.tensor(
            [T.im_start_token_id, T.user_token_id, T.newline_token_id],
            device=self.device, dtype=torch.long,
        ))
        turn_ids_list.append(torch.full((3,), start_idx, device=self.device, dtype=torch.long))

        v_emb = self._encode_visual(visuals[0], is_image=True, cache=visual_cache, cache_key=0)
        n_img_tokens = v_emb.shape[0]
        visual_embeds_list.append(v_emb)

        for tok, n in [
            (T.vision_start_token_id, 1),
            (T.video_token_id, n_img_tokens),
            (T.vision_end_token_id, 1),
        ]:
            input_ids_list.append(torch.full((n,), tok, device=self.device, dtype=torch.long))
            turn_ids_list.append(torch.full((n,), start_idx, device=self.device, dtype=torch.long))

        for k in range(start_idx, num_texts - 1):
            text_ids = self.processor.tokenizer(
                texts[k], return_tensors="pt", add_special_tokens=False
            ).input_ids[0].to(self.device)
            input_ids_list.append(text_ids)
            turn_ids_list.append(torch.full((text_ids.shape[0],), k, device=self.device, dtype=torch.long))

            v_emb = self._encode_visual(
                visuals[k + 1], is_image=False, cache=visual_cache, cache_key=k + 1
            )
            n_vid_tokens = v_emb.shape[0]
            visual_embeds_list.append(v_emb)

            for tok, n in [
                (T.vision_start_token_id, 1),
                (T.video_token_id, n_vid_tokens),
                (T.vision_end_token_id, 1),
            ]:
                input_ids_list.append(torch.full((n,), tok, device=self.device, dtype=torch.long))
                turn_ids_list.append(torch.full((n,), k + 1, device=self.device, dtype=torch.long))

        target_ids = self.processor.tokenizer(
            texts[-1], return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(self.device)
        input_ids_list.append(target_ids)
        turn_ids_list.append(torch.full((target_ids.shape[0],), num_texts - 1, device=self.device, dtype=torch.long))

        input_ids_list.append(torch.tensor(
            [T.im_end_token_id, T.newline_token_id],
            device=self.device, dtype=torch.long,
        ))
        turn_ids_list.append(torch.full((2,), num_texts - 1, device=self.device, dtype=torch.long))

        input_ids_list.append(torch.tensor(
            [T.im_start_token_id, T.assistant_token_id, T.newline_token_id],
            device=self.device, dtype=torch.long,
        ))
        turn_ids_list.append(torch.full((3,), num_texts - 1, device=self.device, dtype=torch.long))

        input_ids = torch.cat(input_ids_list).unsqueeze(0)
        token_turn_ids = torch.cat(turn_ids_list).unsqueeze(0)
        visual_embeds = torch.cat(visual_embeds_list).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        return self.model(
            ids=input_ids,
            mask=attention_mask,
            visual_embeds=visual_embeds,
            token_turn_ids=token_turn_ids,
            mode="wm",
        )

    def generate(
        self,
        input_prompts: List[str],
        img,
        max_area: int = 480 * 480,
        window_size: int = 21,
        sampling_steps: int = 50,
        action_guide_scale: Optional[float] = None,
        state_guide_scale: Optional[float] = None,
        ego_state_guide_scale: Optional[float] = None,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        round_limit: int = 32,
        return_mask_video: bool = False,
        save_mask_steps_path: Optional[str] = None,
        mask_threshold: float = 0.5,
        gt_mask_paths: Optional[List[str]] = None,
        return_branch_videos: bool = False,
        mask_video_save_path: Optional[str] = None,
        mask_video_fps: int = 16,
        intermediate_save_dir: Optional[str] = None,
        intermediate_fps: int = 16,
    ):
        missing_cfg = [
            name for name, val in (
                ("action_guide_scale", action_guide_scale),
                ("state_guide_scale", state_guide_scale),
                ("ego_state_guide_scale", ego_state_guide_scale),
            )
            if val is None
        ]
        if missing_cfg:
            raise ValueError(f"Missing required CFG scale(s): {', '.join(missing_cfg)}")
        action_guide_scale = float(action_guide_scale)
        state_guide_scale = float(state_guide_scale)
        ego_state_guide_scale = float(ego_state_guide_scale)

        explicit_return_mask_video = bool(return_mask_video)
        if mask_video_save_path is not None:
            return_mask_video = True
        if gt_mask_paths is not None and len(gt_mask_paths) != len(input_prompts):
            raise ValueError("gt_mask_paths must have the same length as input_prompts.")

        ih, iw = img.height, img.width
        dh = self.patch_size[1] * self.vae_stride[1]
        dw = self.patch_size[2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1, y1 = (img.width - ow) // 2, (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        sink_frames = int(getattr(self.model.wan_decoder, "num_sink_frames", 0))
        cond_frames = int(getattr(self.model.wan_decoder, "num_cond_frames", 1))
        if cond_frames != 1:
            raise ValueError(f"WEMGenerator expects num_cond_frames=1, got {cond_frames}")
        if sink_frames not in (0, 1):
            raise ValueError(f"WEMGenerator expects num_sink_frames in {{0, 1}}, got {sink_frames}")
        if (window_size - cond_frames) % 2 != 0:
            raise ValueError(
                f"window_size={window_size} must equal cond_frames({cond_frames}) + 2 * chunk_latents."
            )

        chunk_latents = (window_size - cond_frames) // 2
        prefix_frames = sink_frames + cond_frames
        latent_h = oh // self.vae_stride[1]
        latent_w = ow // self.vae_stride[2]
        latent_t = sink_frames + window_size
        seq_len = latent_t * latent_h * latent_w // (self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            contexts = [self.text_encoder([p], self.device) for p in input_prompts]
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_null = [t.to(self.device) for t in self.text_encoder([n_prompt], torch.device("cpu"))]
            contexts = [
                [t.to(self.device) for t in self.text_encoder([p], torch.device("cpu"))]
                for p in input_prompts
            ]

        z_first = self.vae.encode([img_tensor])[0]
        sink_latent = z_first[:, :sink_frames] if sink_frames > 0 else None

        sigmas_prev = torch.linspace(0.5, 0.0, sampling_steps + 1, device=self.device)
        sigmas_curr = torch.linspace(1.0, 0.5, sampling_steps + 1, device=self.device)
        timesteps_prev = sigmas_prev * 1000
        timesteps_curr = sigmas_curr * 1000

        def _load_gt_mask(mask_path: str) -> torch.Tensor:
            suffix = Path(mask_path).suffix.lower()
            if suffix == ".npz":
                raw = np.load(mask_path)["mask"].astype(np.float32)
                m = torch.from_numpy(raw)
            elif suffix in {".pt", ".pth"}:
                m = torch.load(mask_path, map_location="cpu")
                if not isinstance(m, torch.Tensor):
                    raise TypeError(f"Expected tensor in {mask_path}, got {type(m)!r}")
                m = m.float()
            else:
                raise ValueError(f"Unsupported GT mask format: {mask_path}")
            while m.dim() > 3:
                m = m.squeeze(0)
            if m.dim() != 3:
                raise ValueError(f"GT mask must be (T,H,W) after squeeze, got {tuple(m.shape)}")
            m = m.clamp(0.0, 1.0)
            return resize_soft_mask(m, (chunk_latents, latent_h, latent_w)).to(
                device=self.device, dtype=self.param_dtype
            )

        gt_masks = [_load_gt_mask(p) for p in gt_mask_paths] if gt_mask_paths is not None else None

        def _build_gt_mask_for_decoder(action_idx: int, bootstrap: bool = False) -> Optional[torch.Tensor]:
            if gt_masks is None:
                return None
            ones_cond = torch.ones((cond_frames, latent_h, latent_w), device=self.device, dtype=self.param_dtype)
            prev_mask = gt_masks[action_idx]
            next_mask = prev_mask if bootstrap else gt_masks[min(action_idx + 1, len(gt_masks) - 1)]
            parts = []
            if sink_frames > 0:
                parts.append(torch.ones((sink_frames, latent_h, latent_w), device=self.device, dtype=self.param_dtype))
            parts.extend([ones_cond, prev_mask, next_mask])
            return torch.cat(parts, dim=0).unsqueeze(0).unsqueeze(0)

        def _make_step_mask(mask_logits: torch.Tensor, gt_mask: Optional[torch.Tensor]) -> torch.Tensor:
            if gt_mask is not None:
                return (
                    F.interpolate(
                        gt_mask.to(device=self.device, dtype=torch.float32),
                        size=tuple(int(v) for v in mask_logits.shape[2:]),
                        mode="trilinear",
                        align_corners=False,
                    )[0, 0].clamp(0, 1) * 255
                ).to(torch.uint8).cpu()
            return (torch.sigmoid(mask_logits[0, 0]).detach().clamp(0, 1) * 255).to(torch.uint8).cpu()

        def _null_state_like(
            world_state: torch.Tensor,
            ego_state: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            decoder = self.model.wan_decoder
            if hasattr(decoder, "_fsdp_wrapped_module"):
                decoder = decoder._fsdp_wrapped_module
            elif hasattr(decoder, "module"):
                decoder = decoder.module
            world_tok = decoder.null_world_state_embed.to(device=world_state.device, dtype=world_state.dtype)
            ego_tok = decoder.null_ego_state_embed.to(device=ego_state.device, dtype=ego_state.dtype)
            null_world = world_tok.reshape(-1, world_state.shape[-1])[0].view(1, 1, -1).expand_as(world_state).contiguous()
            null_ego = ego_tok.reshape(-1, ego_state.shape[-1])[0].view(1, 1, -1).expand_as(ego_state).contiguous()
            return null_world, null_ego

        def _make_timestep(ts_prev, ts_curr):
            cols = 4 if sink_frames > 0 else 3
            ts = torch.zeros(1, cols, device=self.device)
            if sink_frames > 0:
                ts[0, 0] = 1.0
                ts[0, 1] = 55.0
                ts[0, 2] = ts_prev
                ts[0, 3] = ts_curr
            else:
                ts[0, 0] = 55.0
                ts[0, 1] = ts_prev
                ts[0, 2] = ts_curr
            return ts

        num_chunks = len(input_prompts)
        all_decoded_videos = []
        all_step_masks = []
        final_prev_masks = []
        per_chunk_phase1: Dict[int, List[torch.Tensor]] = {k: [] for k in range(num_chunks)}
        per_chunk_phase2: Dict[int, List[torch.Tensor]] = {k: [] for k in range(num_chunks)}
        all_world_videos: List[torch.Tensor] = []
        all_ego_videos: List[torch.Tensor] = []

        cond_latent = z_first[:, -cond_frames:]
        cond_latent_noised = _noise_with_sigma(cond_latent, sigma=0.055, generator=seed_g)

        img_uint8 = (img_tensor.squeeze(1) * 0.5 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
        wm_history_visuals = [img_uint8]
        wm_history_texts = []
        visual_cache: Dict[int, torch.Tensor] = {}

        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            if self.rank == 0:
                all_decoded_videos.append(img_tensor)
                if return_branch_videos:
                    all_world_videos.append(img_tensor)
                    all_ego_videos.append(img_tensor)

            wm_history_texts.append(input_prompts[0])
            world_state, ego_state = self._get_current_state(wm_history_visuals, wm_history_texts, round_limit, visual_cache)
            null_world_state, null_ego_state = _null_state_like(world_state, ego_state)

            chunk0 = torch.randn(
                self.vae.model.z_dim, chunk_latents, latent_h, latent_w,
                dtype=torch.float32, generator=seed_g, device=self.device,
            )
            if return_branch_videos:
                chunk0_world = chunk0.clone()
                chunk0_ego = chunk0.clone()

            gt_mask_boot = _build_gt_mask_for_decoder(0, bootstrap=True)
            arg_c_boot = {
                "context": {"curr": contexts[0], "next": contexts[0]},
                "seq_len": seq_len,
                "state": (world_state, ego_state),
            }
            arg_state_boot = {
                "context": {"curr": context_null, "next": context_null},
                "seq_len": seq_len,
                "state": (world_state, null_ego_state),
            }
            arg_ego_state_boot = {
                "context": {"curr": context_null, "next": context_null},
                "seq_len": seq_len,
                "state": (world_state, ego_state),
            }
            arg_null_boot = {
                "context": {"curr": context_null, "next": context_null},
                "seq_len": seq_len,
                "state": (null_world_state, null_ego_state),
            }
            for t in tqdm(range(sampling_steps), desc="bootstrap"):
                ts = _make_timestep(timesteps_curr[t], timesteps_curr[t])
                x_parts = ([sink_latent] if sink_frames > 0 else []) + [cond_latent_noised, chunk0, chunk0]
                x_full = torch.cat(x_parts, dim=1)

                if return_branch_videos:
                    _out_c, mask_logits, _branch_boot = self.model(
                        x=[x_full], t=ts, mode="decoder",
                        return_mask=True, return_branches=True, gt_mask=gt_mask_boot, **arg_c_boot,
                    )
                    world_noise_boot = _branch_boot["world"][0]
                    ego_noise_boot = _branch_boot["ego"][0]
                else:
                    _out_c, mask_logits = self.model(
                        x=[x_full], t=ts, mode="decoder",
                        return_mask=True, gt_mask=gt_mask_boot, **arg_c_boot,
                    )
                noise_pred_cond = _out_c[0]

                step_mask_boot = _make_step_mask(mask_logits, gt_mask_boot)
                all_step_masks.append(step_mask_boot)
                per_chunk_phase1[0].append(
                    step_mask_boot[prefix_frames:prefix_frames + chunk_latents].clone()
                )

                noise_pred_uncond = self.model(x=[x_full], t=ts, mode="decoder", **arg_null_boot)[0]
                noise_pred_state = self.model(x=[x_full], t=ts, mode="decoder", **arg_state_boot)[0]
                noise_pred_ego_state = self.model(x=[x_full], t=ts, mode="decoder", **arg_ego_state_boot)[0]

                noise_pred = (
                    noise_pred_uncond
                    + state_guide_scale * (noise_pred_state - noise_pred_uncond)
                    + ego_state_guide_scale * (noise_pred_ego_state - noise_pred_state)
                    + action_guide_scale * (noise_pred_cond - noise_pred_ego_state)
                )
                pred_chunk = noise_pred[:, prefix_frames:prefix_frames + chunk_latents]
                chunk0 = chunk0 + (sigmas_curr[t + 1] - sigmas_curr[t]) * pred_chunk

                if return_branch_videos:
                    _cfg_base_boot = (
                        noise_pred_uncond
                        + state_guide_scale * (noise_pred_state - noise_pred_uncond)
                        + ego_state_guide_scale * (noise_pred_ego_state - noise_pred_state)
                    )
                    pred_world_boot = _cfg_base_boot + action_guide_scale * (world_noise_boot - noise_pred_ego_state)
                    pred_ego_boot = _cfg_base_boot + action_guide_scale * (ego_noise_boot - noise_pred_ego_state)
                    _ds = sigmas_curr[t + 1] - sigmas_curr[t]
                    chunk0_world = chunk0_world + _ds * pred_world_boot[:, prefix_frames:prefix_frames + chunk_latents]
                    chunk0_ego = chunk0_ego + _ds * pred_ego_boot[:, prefix_frames:prefix_frames + chunk_latents]

            prev = chunk0
            if return_branch_videos:
                prev_world = chunk0_world
                prev_ego = chunk0_ego

            for action_idx in range(num_chunks):
                print(f"Generating chunk {action_idx + 1}/{num_chunks}")
                ctx_curr = contexts[action_idx]
                ctx_next = contexts[action_idx + 1] if action_idx + 1 < len(contexts) else ctx_curr

                if action_idx > 0:
                    wm_history_texts.append(input_prompts[action_idx])
                world_state, ego_state = self._get_current_state(wm_history_visuals, wm_history_texts, round_limit, visual_cache)
                null_world_state, null_ego_state = _null_state_like(world_state, ego_state)

                curr = torch.randn(
                    self.vae.model.z_dim, chunk_latents, latent_h, latent_w,
                    dtype=torch.float32, generator=seed_g, device=self.device,
                )
                if return_branch_videos:
                    curr_world = curr.clone()
                    curr_ego = curr.clone()
                gt_mask_action = _build_gt_mask_for_decoder(action_idx, bootstrap=False)
                arg_c = {
                    "context": {"curr": ctx_curr, "next": ctx_next},
                    "seq_len": seq_len,
                    "state": (world_state, ego_state),
                }
                arg_state = {
                    "context": {"curr": context_null, "next": context_null},
                    "seq_len": seq_len,
                    "state": (world_state, null_ego_state),
                }
                arg_ego_state = {
                    "context": {"curr": context_null, "next": context_null},
                    "seq_len": seq_len,
                    "state": (world_state, ego_state),
                }
                arg_null = {
                    "context": {"curr": context_null, "next": context_null},
                    "seq_len": seq_len,
                    "state": (null_world_state, null_ego_state),
                }

                for t in tqdm(range(sampling_steps), desc=f"Chunk {action_idx}"):
                    ts = _make_timestep(timesteps_prev[t], timesteps_curr[t])
                    x_parts = ([sink_latent] if sink_frames > 0 else []) + [cond_latent_noised, prev, curr]
                    x_full = torch.cat(x_parts, dim=1)

                    if return_branch_videos:
                        _out_c, mask_logits, _branch = self.model(
                            x=[x_full], t=ts, mode="decoder",
                            return_mask=True, return_branches=True, gt_mask=gt_mask_action, **arg_c,
                        )
                        world_noise_cond = _branch["world"][0]
                        ego_noise_cond = _branch["ego"][0]
                    else:
                        _out_c, mask_logits = self.model(
                            x=[x_full], t=ts, mode="decoder",
                            return_mask=True, gt_mask=gt_mask_action, **arg_c,
                        )
                    noise_pred_cond = _out_c[0]

                    step_mask = _make_step_mask(mask_logits, gt_mask_action)
                    all_step_masks.append(step_mask)
                    if t == sampling_steps - 1:
                        final_prev_masks.append(step_mask[prefix_frames:prefix_frames + chunk_latents])
                    per_chunk_phase2[action_idx].append(
                        step_mask[prefix_frames:prefix_frames + chunk_latents].clone()
                    )
                    if action_idx + 1 < num_chunks:
                        per_chunk_phase1[action_idx + 1].append(
                            step_mask[prefix_frames + chunk_latents:prefix_frames + 2 * chunk_latents].clone()
                        )

                    noise_pred_uncond = self.model(x=[x_full], t=ts, mode="decoder", **arg_null)[0]
                    noise_pred_state = self.model(x=[x_full], t=ts, mode="decoder", **arg_state)[0]
                    noise_pred_ego_state = self.model(x=[x_full], t=ts, mode="decoder", **arg_ego_state)[0]

                    noise_pred = (
                        noise_pred_uncond
                        + state_guide_scale * (noise_pred_state - noise_pred_uncond)
                        + ego_state_guide_scale * (noise_pred_ego_state - noise_pred_state)
                        + action_guide_scale * (noise_pred_cond - noise_pred_ego_state)
                    )

                    pred_prev = noise_pred[:, prefix_frames:prefix_frames + chunk_latents]
                    pred_curr = noise_pred[:, prefix_frames + chunk_latents:prefix_frames + 2 * chunk_latents]
                    prev = prev + (sigmas_prev[t + 1] - sigmas_prev[t]) * pred_prev
                    curr = curr + (sigmas_curr[t + 1] - sigmas_curr[t]) * pred_curr

                    if return_branch_videos:
                        _cfg_base = (
                            noise_pred_uncond
                            + state_guide_scale * (noise_pred_state - noise_pred_uncond)
                            + ego_state_guide_scale * (noise_pred_ego_state - noise_pred_state)
                        )
                        pred_world = _cfg_base + action_guide_scale * (world_noise_cond - noise_pred_ego_state)
                        pred_ego = _cfg_base + action_guide_scale * (ego_noise_cond - noise_pred_ego_state)
                        _ds_prev = sigmas_prev[t + 1] - sigmas_prev[t]
                        _ds_curr = sigmas_curr[t + 1] - sigmas_curr[t]
                        prev_world = prev_world + _ds_prev * pred_world[:, prefix_frames:prefix_frames + chunk_latents]
                        curr_world = curr_world + _ds_curr * pred_world[:, prefix_frames + chunk_latents:prefix_frames + 2 * chunk_latents]
                        prev_ego = prev_ego + _ds_prev * pred_ego[:, prefix_frames:prefix_frames + chunk_latents]
                        curr_ego = curr_ego + _ds_curr * pred_ego[:, prefix_frames + chunk_latents:prefix_frames + 2 * chunk_latents]

                if self.rank == 0:
                    decoded_prev = self.vae.decode([prev])[0]
                    all_decoded_videos.append(decoded_prev)
                    decoded_uint8 = (decoded_prev / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
                    wm_history_visuals.append(decoded_uint8)
                    if return_branch_videos:
                        all_world_videos.append(self.vae.decode([prev_world])[0])
                        all_ego_videos.append(self.vae.decode([prev_ego])[0])

                    if intermediate_save_dir is not None:
                        os.makedirs(intermediate_save_dir, exist_ok=True)
                        _tag = f"chunk_{action_idx:02d}"
                        save_video(
                            tensor=decoded_prev[None],
                            save_file=os.path.join(intermediate_save_dir, f"{_tag}.mp4"),
                            fps=intermediate_fps, nrow=1, normalize=True, value_range=(-1, 1),
                        )
                        if return_branch_videos and all_world_videos:
                            save_video(
                                tensor=all_world_videos[-1][None],
                                save_file=os.path.join(intermediate_save_dir, f"{_tag}_world.mp4"),
                                fps=intermediate_fps, nrow=1, normalize=True, value_range=(-1, 1),
                            )
                            save_video(
                                tensor=all_ego_videos[-1][None],
                                save_file=os.path.join(intermediate_save_dir, f"{_tag}_ego.mp4"),
                                fps=intermediate_fps, nrow=1, normalize=True, value_range=(-1, 1),
                            )
                        print(f"[intermediate] saved {_tag} to {intermediate_save_dir}")

                cond_latent = prev[:, -cond_frames:]
                cond_latent_noised = _noise_with_sigma(cond_latent, sigma=0.055, generator=seed_g)
                prev = curr
                if return_branch_videos:
                    prev_world = curr_world
                    prev_ego = curr_ego

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                final_video = torch.cat(all_decoded_videos, dim=1)
                world_video = torch.cat(all_world_videos, dim=1) if all_world_videos else None
                ego_video = torch.cat(all_ego_videos, dim=1) if all_ego_videos else None

                if save_mask_steps_path and all_step_masks:
                    per_chunk_tensors = []
                    for k in range(num_chunks):
                        p1, p2 = per_chunk_phase1.get(k, []), per_chunk_phase2.get(k, [])
                        if len(p1) != sampling_steps or len(p2) != sampling_steps:
                            print(
                                f"[warn] chunk {k} incomplete mask trace: "
                                f"phase1={len(p1)}/{sampling_steps}, phase2={len(p2)}/{sampling_steps}; skipping."
                            )
                            continue
                        per_chunk_tensors.append(torch.stack(p1 + p2, dim=0))
                    per_chunk_stacked = (
                        torch.stack(per_chunk_tensors, dim=0) if per_chunk_tensors else torch.empty(0)
                    )
                    torch.save(
                        {
                            "step_masks": torch.stack(all_step_masks, dim=0),
                            "step_masks_per_chunk": per_chunk_stacked,
                            "sampling_steps": sampling_steps,
                            "steps_per_chunk": 2 * sampling_steps,
                            "phase_offset": -sampling_steps,
                            "num_chunks": num_chunks,
                            "chunk_latents": chunk_latents,
                            "prefix_frames": prefix_frames,
                            "num_sink_frames": sink_frames,
                            "num_cond_frames": cond_frames,
                            "mask_threshold": float(mask_threshold),
                        },
                        save_mask_steps_path,
                    )

                mask_video = None
                if return_mask_video:
                    video01 = (final_video * 0.5 + 0.5).clamp(0, 1)
                    if final_prev_masks:
                        full_mask_patch = torch.cat(final_prev_masks, dim=0)
                        mask_prob = (
                            full_mask_patch.unsqueeze(0).unsqueeze(0)
                            .to(video01.device, dtype=torch.float32) / 255.0
                        )
                        mask_prob = F.interpolate(
                            mask_prob,
                            size=(video01.shape[1], video01.shape[2], video01.shape[3]),
                            mode="trilinear",
                            align_corners=False,
                        )[0, 0]
                        ego_region = mask_prob < float(mask_threshold)
                        video01 = video01 * (~ego_region).unsqueeze(0).to(video01.dtype)
                    mask_video = (video01 * 2.0 - 1.0).clamp(-1, 1)
                    if mask_video_save_path is not None:
                        mask_video_dir = os.path.dirname(mask_video_save_path)
                        if mask_video_dir:
                            os.makedirs(mask_video_dir, exist_ok=True)
                        save_video(
                            tensor=mask_video[None],
                            save_file=mask_video_save_path,
                            fps=mask_video_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                        )

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        if self.rank != 0:
            return None
        if mask_video_save_path is not None and not explicit_return_mask_video:
            if return_branch_videos:
                return final_video, world_video, ego_video
            return final_video
        if return_mask_video:
            if return_branch_videos:
                return final_video, mask_video, world_video, ego_video
            return final_video, mask_video
        if return_branch_videos:
            return final_video, world_video, ego_video
        return final_video
