from __future__ import annotations
import os

from typing import Any, Dict, Optional
import functools

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from safetensors.torch import save_file
try:
    from torch.distributed.fsdp.wrap import apply_activation_checkpointing, checkpoint_wrapper
except Exception:  # pragma: no cover - older torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from wem.models.wan_ar import WanARAttentionBlock, WanARSelfAttention

try:
    from third_party.wan.modules.model import WanAttentionBlock
except Exception:  # pragma: no cover - optional dependency
    WanAttentionBlock = None

from wem.training.distributed import get_rank, is_distributed, is_main_process, barrier

_QWEN_CLASS_NAMES = {"Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"}


def _get_cfg(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        value = cfg.get(name, default)
        return default if value is None else value
    value = getattr(cfg, name, default)
    return default if value is None else value


def is_fsdp_model(model: torch.nn.Module) -> bool:
    return isinstance(model, FSDP)


def _get_auto_wrap_policy(cfg: Any):
    policy_type = _get_cfg(cfg, "fsdp_auto_wrap_policy", "transformer")
    if policy_type == "size":
        min_params = _get_cfg(cfg, "fsdp_min_num_params", int(1e7))
        return size_based_auto_wrap_policy(min_num_params=min_params)

    layer_cls = {WanARAttentionBlock, WanARSelfAttention}
    if WanAttentionBlock is not None:
        layer_cls.add(WanAttentionBlock)
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
            Qwen3VLVisionBlock,
        )
        layer_cls.update({Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock})
    except Exception:  # pragma: no cover - optional dependency
        pass

    def _policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        if isinstance(module, tuple(layer_cls)):
            return True
        return module.__class__.__name__ in _QWEN_CLASS_NAMES

    return _policy


def _get_mixed_precision(cfg: Any) -> Optional[MixedPrecision]:
    precision = _get_cfg(cfg, "fsdp_mixed_precision", "bf16")
    if precision == "none":
        return None
    if precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _get_sharding_strategy(cfg: Any) -> ShardingStrategy:
    mode = _get_cfg(cfg, "fsdp_sharding", "full")
    if mode == "hybrid":
        return ShardingStrategy.HYBRID_SHARD
    return ShardingStrategy.FULL_SHARD


def _cast_floating_model_tensors(model: torch.nn.Module, target_dtype: torch.dtype) -> None:
    for param in model.parameters():
        if torch.is_floating_point(param.data) and param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)
    for buffer in model.buffers():
        if torch.is_floating_point(buffer.data) and buffer.dtype != target_dtype:
            buffer.data = buffer.data.to(dtype=target_dtype)


def _apply_activation_checkpointing(model: torch.nn.Module, cfg: Any):
    if not _get_cfg(cfg, "fsdp_activation_checkpointing", False):
        return
    layer_cls = (WanARAttentionBlock,)
    if WanAttentionBlock is not None:
        layer_cls = layer_cls + (WanAttentionBlock,)

    def check_fn(module: torch.nn.Module) -> bool:
        if isinstance(module, layer_cls):
            return True
        return module.__class__.__name__ in _QWEN_CLASS_NAMES

    def wrapper(module: torch.nn.Module) -> torch.nn.Module:
        impl = CheckpointImpl.NO_REENTRANT
        if module.__class__.__name__ in _QWEN_CLASS_NAMES:
            impl = CheckpointImpl.REENTRANT
        return checkpoint_wrapper(module, checkpoint_impl=impl)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


def build_fsdp_model(model: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    if not _get_cfg(cfg, "fsdp", False):
        return model

    _apply_activation_checkpointing(model, cfg)

    mixed_precision = _get_mixed_precision(cfg)
    if mixed_precision is not None:
        _cast_floating_model_tensors(model, mixed_precision.param_dtype)
    cpu_offload = CPUOffload(offload_params=True) if _get_cfg(cfg, "fsdp_cpu_offload", False) else None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = local_rank
    else:
        device_id = None

    return FSDP(
        model,
        auto_wrap_policy=_get_auto_wrap_policy(cfg),
        mixed_precision=mixed_precision,
        sharding_strategy=_get_sharding_strategy(cfg),
        cpu_offload=cpu_offload,
        device_id=device_id,
        sync_module_states=_get_cfg(cfg, "fsdp_sync_module_states", True),
        forward_prefetch=_get_cfg(cfg, "fsdp_forward_prefetch", True),
        use_orig_params=True,
        limit_all_gathers=_get_cfg(cfg, "fsdp_limit_all_gathers", True),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )


def save_model_checkpoint(model: torch.nn.Module, step_dir: str):
    if is_fsdp_model(model):
        model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=model_cfg
        ):
            model_sd = model.state_dict()
    else:
        model_sd = model.state_dict()

    if is_main_process():
        save_file(model_sd, os.path.join(step_dir, "model.safetensors"))

    barrier()
