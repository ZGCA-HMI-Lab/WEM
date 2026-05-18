import os
from typing import Optional

import torch
import torch.distributed as dist


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    force: bool = False,
) -> bool:
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return True
    world_size = _get_env_int("WORLD_SIZE", 1)
    if world_size <= 1 and not force:
        return False
    if world_size <= 1 and force:
        rank = 0
        world_size = 1
        if init_method is None:
            init_method = "tcp://127.0.0.1:29500"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method=init_method,
        )
        return True
    rank = _get_env_int("RANK", 0)
    local_rank = _get_env_int("LOCAL_RANK", 0)

    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)

    import datetime
    timeout = datetime.timedelta(minutes=30)

    print(f"[Rank {rank}] Initializing process group: backend={backend}, rank={rank}/{world_size}, local_rank={local_rank}")
    print(f"[Rank {rank}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")

    device_id = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() and backend == "nccl" else None

    init_kwargs = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "timeout": timeout,
    }
    if device_id is not None:
        init_kwargs["device_id"] = device_id
    if init_method is not None:
        init_kwargs["init_method"] = init_method

    dist.init_process_group(**init_kwargs)
    print(f"[Rank {rank}] Process group initialized successfully")
    return True


def setup_device(default_device: str = "cuda") -> str:
    local_rank = _get_env_int("LOCAL_RANK", 0)
    if torch.cuda.is_available() and "cuda" in default_device:
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return _get_env_int("RANK", 0)


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return _get_env_int("WORLD_SIZE", 1)


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / get_world_size()
