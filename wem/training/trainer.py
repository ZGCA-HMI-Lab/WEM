import os
import math
from typing import Dict, Any, Optional, List, Union, Tuple
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from wem.models.wan_ar import WanARModel
from wem.models.wem import WEMModel
from wem.utils.masks import resize_soft_mask
from wem.training import distributed
from wem.training import fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from wem.training.scheduler import CausalSwinFlowMatchScheduler

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import time

class FSDPShardedEMA:
    # Under FSDP with use_orig_params=True, param.data is only a view of the
    # FlatParameter. In-place copy_ must be used to actually modify the
    # underlying storage — Python-level rebinding has no effect.
    def __init__(self, model: torch.nn.Module, target_decay: float = 0.9999):
        self.target_decay = target_decay
        self.decay = target_decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.decay = self.target_decay
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            shadow = self.shadow[name]
            if shadow.shape != param.data.shape:
                # Shard shape changed (rare, e.g. after re-wrap); rebuild shadow.
                self.shadow[name] = param.data.detach().clone()
                continue
            shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            shadow = self.shadow[name]
            if shadow.shape != param.data.shape:
                continue
            if name not in self.backup:
                self.backup[name] = param.data.detach().clone()
            else:
                self.backup[name].copy_(param.data)
            param.data.copy_(shadow)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.backup:
                continue
            backup = self.backup[name]
            if backup.shape != param.data.shape:
                continue
            param.data.copy_(backup)
        self.backup.clear()

class BaseTrainer:
    def __init__(
        self,
        model: Union[WanARModel, torch.nn.Module],
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        grad_clip: Optional[float] = None,
        grad_accum_steps: int = 1,
        log_memory: bool = False,
        lr_scheduler_type: str = "cosine",
        num_warmup_steps: int = 100,
        max_steps: int = 100000,
        adam_betas: tuple = (0.95, 0.99),
        adam_epsilon: float = 1e-8,
        use_fused_optimizer: bool = True,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.log_memory = log_memory

        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.max_steps = max_steps
        self.adam_betas = adam_betas
        self.adam_epsilon = adam_epsilon
        self.use_fused_optimizer = use_fused_optimizer

        self.global_step = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[Any] = None
        self.trainable_params: List[torch.nn.Parameter] = []
        self.extra_params: List[torch.nn.Parameter] = []

        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )
        self._accum_step = 0
        self._last_grad_norm = 0.0

        self._timings = {
            "forward": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
            "total": 0.0,
        }

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _create_optimizer(self, params, lr, weight_decay):
        return AdamW(
            params,
            lr=lr,
            betas=self.adam_betas,
            eps=self.adam_epsilon,
            weight_decay=weight_decay,
            fused=self.use_fused_optimizer and torch.cuda.is_available(),
        )

    def _create_lr_scheduler(self):
        if self.optimizer is None:
            return None

        if self.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                self.num_warmup_steps,
                self.max_steps
            )
        elif self.lr_scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                self.num_warmup_steps,
                self.max_steps
            )
        elif self.lr_scheduler_type == "constant":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                self.num_warmup_steps,
                10000000000
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.lr_scheduler_type}")

    def _clip_gradients(self) -> float:
        max_norm = self.grad_clip if self.grad_clip is not None else float("inf")
        if fsdp.is_fsdp_model(self.model):
            grad_norm = FSDP.clip_grad_norm_(self.model, max_norm)
            return float(grad_norm)
        params = self.trainable_params
        total_norm = torch.nn.utils.clip_grad_norm_(
            params,
            max_norm,
            foreach=True
        )
        return total_norm.item()

    def _step_optimizer(self) -> tuple[float, Dict[str, float]]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")

        optim_start = time.time()

        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        self._sync_extra_grads()

        grad_norm = self._clip_gradients()
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError(f"Non-finite grad norm detected: {grad_norm}")

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        self._last_grad_norm = grad_norm
        self.global_step += 1

        self._timings["optimizer"] = time.time() - optim_start

        return grad_norm, {}

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.amp_dtype)

    def _sync_extra_grads(self):
        if not self.extra_params or not distributed.is_distributed():
            return
        world_size = distributed.get_world_size()
        if world_size <= 1:
            return
        for p in self.extra_params:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)

    def _prepare_backward(self):
        if self._accum_step == 0 and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def _backward(self, loss: torch.Tensor):
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _maybe_no_sync(self, is_last: bool):
        if not is_last and fsdp.is_fsdp_model(self.model) and distributed.is_distributed():
            return self.model.no_sync()
        return nullcontext()

    def _get_model_dtype(self) -> torch.dtype:
        """Return model dtype, using amp_dtype directly for FSDP to avoid parameter access."""
        if hasattr(self, '_cached_model_dtype'):
            return self._cached_model_dtype

        if fsdp.is_fsdp_model(self.model) and self.use_amp:
            self._cached_model_dtype = self.amp_dtype
            return self._cached_model_dtype

        model = self.model
        if hasattr(model, '_fsdp_wrapped_module'):
            model = model._fsdp_wrapped_module
        elif hasattr(model, 'module'):
            model = model.module

        for param in model.parameters():
            self._cached_model_dtype = param.dtype
            return self._cached_model_dtype

        self._cached_model_dtype = self.dtype
        return self._cached_model_dtype

    def _reduce_loss_for_logging(self, loss: torch.Tensor) -> float:
        reduced = distributed.all_reduce_mean(loss)
        return reduced.item()

    def _reduce_float_for_logging(self, value: float) -> float:
        try:
            device = torch.device(self.device)
        except Exception:
            device = torch.device("cpu")
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
        reduced = distributed.all_reduce_mean(tensor)
        return float(reduced.item())

    def _compute_seq_len(self, frame_count: int, height: int, width: int) -> int:
        model = self.model
        if hasattr(model, '_fsdp_wrapped_module'):
            model = model._fsdp_wrapped_module
        elif hasattr(model, 'module'):
            model = model.module
        patch_size = getattr(model, 'patch_size', (1, 2, 2))
        patch_t, patch_h, patch_w = patch_size
        return (frame_count // patch_t) * (height // patch_h) * (width // patch_w)

    def save_checkpoint(self, step_dir: str):
        os.makedirs(step_dir, exist_ok=True)
        fsdp.save_model_checkpoint(self.model, step_dir=step_dir)

class WanARTrainer(BaseTrainer):
    def __init__(
        self,
        model: WanARModel,
        cond_size: int,
        chunk_size: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        grad_clip: Optional[float] = None,
        grad_accum_steps: int = 1,
        log_memory: bool = False,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_min: float = 0.003 / 1.002,
        use_loss_weighting: bool = False,
        lr_scheduler_type: str = "cosine",
        num_warmup_steps: int = 1000,
        max_steps: int = 100000,
        first_chunk_training_prob: float = 0.1,
        cfg_dropout_prob: float = 0.1,
    ):
        super().__init__(
            model=model,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            log_memory=log_memory,
            lr_scheduler_type=lr_scheduler_type,
            num_warmup_steps=num_warmup_steps,
            max_steps=max_steps,
        )
        self.cond_size = cond_size
        self.chunk_size = chunk_size

        if not fsdp.is_fsdp_model(self.model):
            self.model.to(self.device, dtype=self.dtype)

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._create_optimizer(
            self.trainable_params,
            learning_rate,
            weight_decay
        )
        self.lr_scheduler = self._create_lr_scheduler()

        self.scheduler = CausalSwinFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(num_inference_steps=1000, training=True)
        self.use_loss_weighting = use_loss_weighting
        self.first_chunk_training_prob = first_chunk_training_prob

        self.cfg_dropout_prob = cfg_dropout_prob

        self.target_ema_decay = 0.9
        self.ema_model_weights = FSDPShardedEMA(self.model, target_decay=self.target_ema_decay)

    def save_checkpoint(self, step_dir: str):
        super().save_checkpoint(step_dir)
        if hasattr(self, 'ema_model_weights'):
            self.ema_model_weights.swap(self.model)
            super().save_checkpoint(f"{step_dir}_ema")
            self.ema_model_weights.restore(self.model)

    def train_one_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.model.train()
        latents_clean = batch["latents"].to(self.device, dtype=self._get_model_dtype()).clone()
        is_last_chunk = batch["is_last"].to(self.device)

        B, C, T, H, W = latents_clean.shape

        drop_mask = torch.rand(B, device=self.device) < self.cfg_dropout_prob
        context_curr = []
        context_next = []
        for i, (c_curr, c_next, c_null) in enumerate(zip(batch["context_curr"], batch["context_next"], batch["context_null"])):
            c_curr_tensor = c_curr.to(self.device, dtype=self._get_model_dtype())
            c_next_tensor = c_next.to(self.device, dtype=self._get_model_dtype())
            c_null_tensor = c_null.to(self.device, dtype=self._get_model_dtype())
            if c_null_tensor.dim() == 1:
                c_null_tensor = c_null_tensor.unsqueeze(0)
            if bool(drop_mask[i].item()):
                c_curr_tensor = c_null_tensor
                c_next_tensor = c_null_tensor
            context_curr.append(c_curr_tensor)
            context_next.append(c_next_tensor)
        context = {
            "curr": context_curr,
            "next": context_next,
        }

        cond_frames = int(max(1, self.cond_size))
        if T < cond_frames + 2 * self.chunk_size:
            raise ValueError(
                f"Invalid latent temporal length T={T}; expected at least cond_size({cond_frames}) + 2*chunk_size({self.chunk_size})."
            )

        rand_probs = torch.rand(B, device=latents_clean.device)
        use_first_chunk_mode = rand_probs < self.first_chunk_training_prob
        timesteps_ref = self.scheduler.timesteps.to(latents_clean.device)
        num_timesteps = len(timesteps_ref)
        half_timesteps = num_timesteps // 2

        ids_high = torch.randint(0, half_timesteps, (B,), device=latents_clean.device)
        ids_low = ids_high + half_timesteps
        final_next_ids = ids_high
        final_curr_ids = torch.where(use_first_chunk_mode, ids_high, ids_low)
        curr_timesteps = timesteps_ref[final_curr_ids]
        next_timesteps = timesteps_ref[final_next_ids]

        noise = torch.randn_like(latents_clean)
        x_t = self.scheduler.add_noise(
            latents_clean,
            noise,
            curr_timesteps,
            next_timesteps,
            self.chunk_size,
            cond_size=cond_frames,
        )
        v_target = self.scheduler.training_target(latents_clean, noise)
        x_list = [x_t[i] for i in range(B)]

        seq_len = self._compute_seq_len(T, H, W)

        t_cf = torch.full((B, 1), 55.0, device=latents_clean.device, dtype=latents_clean.dtype)
        t_curr_reshaped = curr_timesteps.unsqueeze(1).to(dtype=latents_clean.dtype)
        t_next_reshaped = next_timesteps.unsqueeze(1).to(dtype=latents_clean.dtype)
        t_combined = torch.cat([t_cf, t_curr_reshaped, t_next_reshaped], dim=1)

        is_last = (self._accum_step + 1) % self.grad_accum_steps == 0
        if self._accum_step == 0:
            self._prepare_backward()

        with self._maybe_no_sync(is_last):
            with self._autocast():
                v_pred_list = self.model(
                    x=x_list,
                    t=t_combined,
                    context=context,
                    seq_len=seq_len,
                    y=None,
                )
                v_pred = torch.stack(v_pred_list, dim=0)
                raw_loss_flow = ((v_pred[:, :, cond_frames:, :, :] - v_target[:, :, cond_frames:, :, :]) ** 2).mean()

                if self.use_loss_weighting:
                    loss_weight = self.scheduler.get_loss_mask(
                        self.chunk_size, curr_timesteps, next_timesteps,
                        self.device, v_target.dtype, cond_size=cond_frames
                    )
                    temporal_mask = torch.zeros((1, 1, T, 1, 1), device=self.device, dtype=loss_weight.dtype)
                    temporal_mask[:, :, cond_frames:cond_frames+self.chunk_size, :, :] = 1.0
                    should_mask_curr = is_last_chunk | use_first_chunk_mode
                    mask_curr_broadcast = should_mask_curr.view(B, 1, 1, 1, 1).to(dtype=loss_weight.dtype)
                    final_mask = (1.0 - mask_curr_broadcast) + mask_curr_broadcast * temporal_mask
                    loss_flow = ((v_pred - v_target) ** 2 * loss_weight * final_mask).mean()
                else:
                    mask = torch.ones_like(v_pred)
                    mask[:, :, :cond_frames, :, :] = 0.0
                    temporal_mask = torch.ones_like(mask)
                    temporal_mask[:, :, cond_frames+self.chunk_size:, :, :] = 0.0
                    should_mask_curr = is_last_chunk | use_first_chunk_mode
                    mask_curr_broadcast = should_mask_curr.view(B, 1, 1, 1, 1).to(dtype=mask.dtype)
                    batch_mask = (1.0 - mask_curr_broadcast) + mask_curr_broadcast * temporal_mask
                    loss_flow = ((v_pred - v_target) ** 2 * mask * batch_mask).mean()
                loss = loss_flow

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected in WanARTrainer: {loss.item()}")
            self._backward(loss / self.grad_accum_steps)

        if is_last:
            grad_norm, _ = self._step_optimizer()
            self._accum_step = 0
            did_step = True
            self.ema_model_weights.update(self.model)
        else:
            self._accum_step += 1
            grad_norm = self._last_grad_norm
            did_step = False

        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0

        raw_loss_flow_val = self._reduce_loss_for_logging(raw_loss_flow.detach())
        loss_flow_val = self._reduce_loss_for_logging(loss_flow.detach())

        metrics = {
            "loss": loss_flow_val,
            "raw_loss": raw_loss_flow_val,
            "grad_norm": grad_norm,
            "lr": lr,
            "did_step": did_step,
            "avg_timestep": (curr_timesteps.mean().item() + next_timesteps.mean().item()) / 2,
        }
        return metrics

class WEMTrainer(BaseTrainer):
    def __init__(
        self,
        model: WEMModel,
        chunk_size: int = 10,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        grad_clip: float = 0.5,
        grad_accum_steps: int = 1,
        max_steps: int = 20000,
        num_warmup_steps: int = 1000,
        log_memory: bool = False,
        shift: float = 5.0,
        sigma_min: float = 0.003 / 1.002,
        first_chunk_training_prob: float = 0.1,
        use_loss_weighting: bool = False,
        action_dropout_prob: float = 0.1,
        world_state_dropout_prob: float = 0.1,
        ego_state_dropout_prob: float = 0.1,
        mask_loss_weight: float = 0.2,
        mask_loss_weight_floor_ratio: float = 0.2,
        ego_flow_emphasis: float = 4.0,
    ):
        super().__init__(
            model=model,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            log_memory=log_memory,
            max_steps=max_steps,
            num_warmup_steps=num_warmup_steps,
        )
        self.chunk_size = chunk_size
        self.first_chunk_training_prob = first_chunk_training_prob
        self.use_loss_weighting = use_loss_weighting

        trainable_params, extra_params = self._get_trainable_params(model)
        self.trainable_params = trainable_params
        self.extra_params = extra_params
        self.optimizer = self._create_optimizer(self.trainable_params, learning_rate, weight_decay)
        self.lr_scheduler = self._create_lr_scheduler()

        self.target_ema_decay = 0.99
        self.ema_model_weights = FSDPShardedEMA(self.model, target_decay=self.target_ema_decay)

        self.scheduler = CausalSwinFlowMatchScheduler(
            num_train_timesteps=1000,
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(num_inference_steps=1000, training=True)
        model_ref = self._get_model_ref()
        wan_decoder = getattr(model_ref, "wan_decoder", None)
        self.sink_size = int(getattr(wan_decoder, "num_sink_frames", 0)) if wan_decoder is not None else 0
        self.sink_noise_sigma = float(getattr(wan_decoder, "sink_noise_sigma", 0.0)) if wan_decoder is not None else 0.0

        self.action_dropout_prob = action_dropout_prob
        self.world_state_dropout_prob = world_state_dropout_prob
        self.ego_state_dropout_prob = ego_state_dropout_prob
        self.mask_loss_weight_init = mask_loss_weight
        self.mask_loss_weight_floor = mask_loss_weight * mask_loss_weight_floor_ratio
        self.ego_flow_emphasis = ego_flow_emphasis

        # Cache null state embeds to avoid FSDP.summon_full_params(recurse=True) every step.
        # Null embeds change slowly (lr=1e-5), so resyncing every N steps is sufficient.
        self._null_embed_sync_interval = 100
        self._null_world_state_embed: Optional[torch.Tensor] = None
        self._null_ego_state_embed: Optional[torch.Tensor] = None
        self._sync_null_embeds(force=True)

    def _get_trainable_params(self, model):
        trainable_params = []
        extra_params = []
        for _, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(p)
        return trainable_params, extra_params

    def _get_model_ref(self):
        model_ref = self.model
        if hasattr(model_ref, '_fsdp_wrapped_module'):
            model_ref = model_ref._fsdp_wrapped_module
        elif hasattr(model_ref, 'module'):
            model_ref = model_ref.module
        return model_ref

    def _sync_null_embeds(self, force: bool = False) -> None:
        """Gather null state embeds from the (possibly FSDP-sharded) model.

        Avoids summon_full_params(recurse=True) on every training step by
        caching the result and only re-syncing every _null_embed_sync_interval
        steps. Null embeds evolve slowly at lr=1e-5, so stale values are fine.
        """
        if not force and (self.global_step % self._null_embed_sync_interval != 0):
            return
        if fsdp.is_fsdp_model(self.model):
            with FSDP.summon_full_params(self.model, recurse=True, writeback=False):
                self._null_world_state_embed = (
                    self.model.module.wan_decoder.null_world_state_embed.detach().clone()
                )
                self._null_ego_state_embed = (
                    self.model.module.wan_decoder.null_ego_state_embed.detach().clone()
                )
        else:
            self._null_world_state_embed = (
                self.model.wan_decoder.null_world_state_embed.detach().clone()
            )
            self._null_ego_state_embed = (
                self.model.wan_decoder.null_ego_state_embed.detach().clone()
            )

    @staticmethod
    def _masked_dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        eps: float = 1.0,
    ) -> torch.Tensor:
        pred = pred * valid_mask
        target = target * valid_mask
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        return 1.0 - (2.0 * intersection + eps) / (denominator + eps)

    @staticmethod
    def _resize_world_mask(
        world_mask: torch.Tensor,
        target_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        return resize_soft_mask(world_mask, target_size)

    @staticmethod
    def _cosine_annealed_weight(init_weight: float, floor_weight: float, progress: float) -> float:
        progress = min(max(float(progress), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(floor_weight + (init_weight - floor_weight) * cosine)

    def save_checkpoint(self, step_dir: str):
        super().save_checkpoint(step_dir)
        if hasattr(self, 'ema_model_weights'):
            self.ema_model_weights.swap(self.model)
            super().save_checkpoint(f"{step_dir}_ema")
            self.ema_model_weights.restore(self.model)

    def train_wm_step(self, batch: Dict[str, Any]):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        token_turn_ids = batch["token_turn_ids"].to(self.device, non_blocking=True)
        visual_embeds = batch["visual_embeds"].to(self.device, self._get_model_dtype(), non_blocking=True)
        world_state, ego_state = self.model(
            ids=input_ids,
            mask=attention_mask,
            visual_embeds=visual_embeds,
            token_turn_ids=token_turn_ids,
            mode="wm",
        )
        return world_state, ego_state

    def train_decoder_step(
        self,
        batch: Dict[str, Any],
        states: Tuple[torch.Tensor, torch.Tensor],
    ):
        latents_clean = batch["latents"].to(self.device, dtype=self._get_model_dtype()).clone()
        is_last_chunk = batch["is_last"].to(self.device)
        world_masks = batch["world_masks"].to(self.device)

        B, _, T, H, W = latents_clean.shape
        sink_frames = int(max(0, self.sink_size))
        cond_frames = int(max(1, T - sink_frames - 2 * self.chunk_size))
        prefix_frames = sink_frames + cond_frames
        if T < sink_frames + cond_frames + 2 * self.chunk_size:
            raise ValueError(
                f"Invalid latent temporal length T={T}; expected at least sink_size({sink_frames}) + cond_size({cond_frames}) + 2*chunk_size({self.chunk_size})."
            )

        action_drop_mask = torch.rand(B, device=self.device) < self.action_dropout_prob
        world_state_drop_mask = torch.rand(B, device=self.device) < self.world_state_dropout_prob
        ego_state_drop_mask = torch.rand(B, device=self.device) < self.ego_state_dropout_prob
        action_drop_flags = action_drop_mask.tolist()
        context_curr = []
        context_next = []
        for i, (c_curr, c_next, c_null) in enumerate(zip(batch["context_curr"], batch["context_next"], batch["context_null"])):
            c_curr_tensor = c_curr.to(self.device, dtype=self._get_model_dtype())
            c_next_tensor = c_next.to(self.device, dtype=self._get_model_dtype())
            c_null_tensor = c_null.to(self.device, dtype=self._get_model_dtype())
            if c_null_tensor.dim() == 1:
                c_null_tensor = c_null_tensor.unsqueeze(0)
            if action_drop_flags[i]:
                c_curr_tensor = c_null_tensor
                c_next_tensor = c_null_tensor
            context_curr.append(c_curr_tensor)
            context_next.append(c_next_tensor)
        context = {"curr": context_curr, "next": context_next}

        state_world, state_ego = states
        state_world_cfg = state_world.clone()
        state_ego_cfg = state_ego.clone()

        local_world_has_drop = bool(world_state_drop_mask.any().item())
        local_ego_has_drop = bool(ego_state_drop_mask.any().item())
        global_world_has_drop = local_world_has_drop
        global_ego_has_drop = local_ego_has_drop
        if fsdp.is_fsdp_model(self.model) and distributed.is_distributed():
            drop_flags = torch.tensor([
                int(local_world_has_drop),
                int(local_ego_has_drop),
            ], device=state_world_cfg.device, dtype=torch.int32)
            dist.all_reduce(drop_flags, op=dist.ReduceOp.MAX)
            global_world_has_drop = bool(drop_flags[0].item())
            global_ego_has_drop = bool(drop_flags[1].item())

        if global_world_has_drop or global_ego_has_drop:
            null_world_state_embed = self._null_world_state_embed
            null_ego_state_embed = self._null_ego_state_embed

            if global_world_has_drop:
                null_world_state_embed = null_world_state_embed.to(device=state_world_cfg.device, dtype=state_world_cfg.dtype).reshape(-1, state_world_cfg.shape[-1])
                if null_world_state_embed.shape[0] < 1:
                    raise RuntimeError("wan_decoder.null_world_state_embed is empty after FSDP full-param gather.")
                null_world_state_token = null_world_state_embed[0]
                if local_world_has_drop:
                    state_world_cfg[world_state_drop_mask] = null_world_state_token.to(device=state_world_cfg.device, dtype=state_world_cfg.dtype).view(1, 1, -1).expand(
                        int(world_state_drop_mask.sum().item()),
                        state_world_cfg.shape[1],
                        state_world_cfg.shape[2],
                    )
            if global_ego_has_drop:
                null_ego_state_embed = null_ego_state_embed.to(device=state_ego_cfg.device, dtype=state_ego_cfg.dtype).reshape(-1, state_ego_cfg.shape[-1])
                if null_ego_state_embed.shape[0] < 1:
                    raise RuntimeError("wan_decoder.null_ego_state_embed is empty after FSDP full-param gather.")
                null_ego_state_token = null_ego_state_embed[0]
                if local_ego_has_drop:
                    state_ego_cfg[ego_state_drop_mask] = null_ego_state_token.to(device=state_ego_cfg.device, dtype=state_ego_cfg.dtype).view(1, 1, -1).expand(
                        int(ego_state_drop_mask.sum().item()),
                        state_ego_cfg.shape[1],
                        state_ego_cfg.shape[2],
                    )

        states_cfg = (state_world_cfg, state_ego_cfg)

        use_first_chunk_mode = bool(
            (torch.rand(1, device=latents_clean.device) < self.first_chunk_training_prob).item()
        )
        is_first_chunk = torch.full((B,), use_first_chunk_mode, device=latents_clean.device, dtype=torch.bool)

        timesteps_ref = self.scheduler.timesteps.to(latents_clean.device)
        half_timesteps = len(timesteps_ref) // 2
        shared_high_id = int(torch.randint(0, half_timesteps, (1,), device=latents_clean.device).item())
        ids_high = torch.full((B,), shared_high_id, device=latents_clean.device, dtype=torch.long)
        ids_low = ids_high + half_timesteps
        final_next_ids = ids_high
        final_curr_ids = torch.where(is_first_chunk, ids_high, ids_low)
        curr_timesteps = timesteps_ref[final_curr_ids]
        next_timesteps = timesteps_ref[final_next_ids]

        noise = torch.randn_like(latents_clean)
        x_t = self.scheduler.add_noise(
            latents_clean,
            noise,
            curr_timesteps,
            next_timesteps,
            self.chunk_size,
            cond_size=cond_frames,
            sink_size=sink_frames,
            sink_noise_sigma=self.sink_noise_sigma,
        )
        v_target = self.scheduler.training_target(latents_clean, noise)
        x_list = [x_t[i] for i in range(B)]

        seq_len = self._compute_seq_len(T, H, W)
        t_sink = torch.full((B, 1), 1.0, device=latents_clean.device, dtype=latents_clean.dtype)
        t_cf = torch.full((B, 1), 55.0, device=latents_clean.device, dtype=latents_clean.dtype)
        t_curr_reshaped = curr_timesteps.unsqueeze(1).to(dtype=latents_clean.dtype)
        t_next_reshaped = next_timesteps.unsqueeze(1).to(dtype=latents_clean.dtype)
        if sink_frames > 0:
            t_combined = torch.cat([t_sink, t_cf, t_curr_reshaped, t_next_reshaped], dim=1)
        else:
            t_combined = torch.cat([t_cf, t_curr_reshaped, t_next_reshaped], dim=1)

        is_last = (self._accum_step + 1) % self.grad_accum_steps == 0
        if self._accum_step == 0:
            self._prepare_backward()

        gt_mask_5d = world_masks.unsqueeze(1).to(dtype=self._get_model_dtype())
        anneal_progress = min(self.global_step / max(float(self.max_steps), 1.0), 1.0)
        cur_mask_loss_weight = self._cosine_annealed_weight(
            self.mask_loss_weight_init,
            self.mask_loss_weight_floor,
            anneal_progress,
        )

        with self._maybe_no_sync(is_last):
            with self._autocast():
                v_pred_list, masks_pred = self.model(
                    x=x_list,
                    t=t_combined,
                    context=context,
                    seq_len=seq_len,
                    state=states_cfg,
                    y=None,
                    return_mask=True,
                    gt_mask=gt_mask_5d,
                    mode="decoder",
                )
                v_pred = torch.stack(v_pred_list, dim=0)
                ego_spatial_w = 1.0 + self.ego_flow_emphasis * (1.0 - world_masks.unsqueeze(1).to(dtype=v_pred.dtype))

                if self.use_loss_weighting:
                    loss_weight = self.scheduler.get_loss_mask(
                        self.chunk_size,
                        curr_timesteps,
                        next_timesteps,
                        self.device,
                        v_target.dtype,
                        cond_size=cond_frames,
                        sink_size=sink_frames,
                    )
                    temporal_mask = torch.zeros((1, 1, T, 1, 1), device=self.device, dtype=loss_weight.dtype)
                    temporal_mask[:, :, prefix_frames:prefix_frames + self.chunk_size, :, :] = 1.0
                    should_mask_curr = is_last_chunk | is_first_chunk
                    mask_curr_broadcast = should_mask_curr.view(B, 1, 1, 1, 1).to(dtype=loss_weight.dtype)
                    flow_valid = loss_weight * ((1.0 - mask_curr_broadcast) + mask_curr_broadcast * temporal_mask)
                else:
                    flow_valid = torch.ones_like(v_pred)
                    flow_valid[:, :, :prefix_frames, :, :] = 0.0
                    temporal_mask = torch.ones_like(flow_valid)
                    temporal_mask[:, :, prefix_frames + self.chunk_size:, :, :] = 0.0
                    should_mask_curr = is_last_chunk | is_first_chunk
                    mask_curr_broadcast = should_mask_curr.view(B, 1, 1, 1, 1).to(dtype=flow_valid.dtype)
                    flow_valid = flow_valid * ((1.0 - mask_curr_broadcast) + mask_curr_broadcast * temporal_mask)

                loss_flow = ((v_pred - v_target) ** 2 * flow_valid * ego_spatial_w).mean()

                mask_valid = torch.ones((B, 1, T, 1, 1), device=self.device, dtype=masks_pred.dtype)
                mask_valid[:, :, :prefix_frames, :, :] = 0.0
                mask_temporal = torch.ones_like(mask_valid)
                mask_temporal[:, :, prefix_frames + self.chunk_size:, :, :] = 0.0
                should_mask_curr = is_last_chunk | is_first_chunk
                mask_curr_bc = should_mask_curr.view(B, 1, 1, 1, 1).to(dtype=mask_valid.dtype)
                mask_valid = mask_valid * ((1.0 - mask_curr_bc) + mask_curr_bc * mask_temporal)
                mask_valid = F.interpolate(mask_valid, size=masks_pred.shape[2:], mode='nearest')

                gt_mask_target = world_masks.unsqueeze(1).to(dtype=masks_pred.dtype)
                gt_mask_target = self._resize_world_mask(
                    gt_mask_target,
                    masks_pred.shape[2:],
                )
                valid_gt = gt_mask_target[mask_valid > 0.5]
                if valid_gt.numel() > 0:
                    pos_ratio = valid_gt.mean().clamp(min=0.01, max=0.99)
                else:
                    pos_ratio = torch.tensor(0.5, device=self.device, dtype=masks_pred.dtype)
                bce_weight = torch.where(gt_mask_target > 0.5, 1.0 - pos_ratio, pos_ratio) * mask_valid
                per_pixel_loss = F.binary_cross_entropy_with_logits(
                    masks_pred,
                    gt_mask_target,
                    weight=bce_weight,
                    reduction="sum",
                )
                mask_bce_loss = per_pixel_loss / bce_weight.sum().clamp(min=1.0)
                pred_manip = 1.0 - torch.sigmoid(masks_pred)
                target_manip = 1.0 - gt_mask_target
                mask_dice_loss = self._masked_dice_loss(pred_manip, target_manip, mask_valid)
                mask_loss = 0.5 * (mask_bce_loss + mask_dice_loss)

                valid_pred = torch.sigmoid(masks_pred)[mask_valid > 0.5]
                mask_pred_mean = valid_pred.mean().item() if valid_pred.numel() > 0 else 0.0
                loss = loss_flow + cur_mask_loss_weight * mask_loss

            self._backward(loss / self.grad_accum_steps)

        if is_last:
            grad_norm, _ = self._step_optimizer()
            self._accum_step = 0
            did_step = True
            self.ema_model_weights.update(self.model)
        else:
            self._accum_step += 1
            grad_norm = self._last_grad_norm
            did_step = False

        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
        loss_val = self._reduce_loss_for_logging(loss.detach())
        loss_flow_val = self._reduce_loss_for_logging(loss_flow.detach())
        loss_mask_val = self._reduce_loss_for_logging(mask_loss.detach())
        loss_mask_bce_val = self._reduce_loss_for_logging(mask_bce_loss.detach())
        loss_mask_dice_val = self._reduce_loss_for_logging(mask_dice_loss.detach())

        metrics = {
            "loss": loss_val,
            "loss_flow": loss_flow_val,
            "loss_mask": loss_mask_val,
            "loss_mask_bce": loss_mask_bce_val,
            "loss_mask_dice": loss_mask_dice_val,
            "mask_loss_weight": cur_mask_loss_weight,
            "mask_pred_mean": mask_pred_mean,
            "grad_norm": grad_norm,
            "lr": lr,
            "did_step": did_step,
            "avg_timestep": (curr_timesteps.mean().item() + next_timesteps.mean().item()) / 2,
        }
        return metrics

    def train_one_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.model.train()
        self._sync_null_embeds()
        with self._autocast():
            states = self.train_wm_step(batch)
        return self.train_decoder_step(batch, states)
