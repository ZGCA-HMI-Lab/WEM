from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def resize_soft_mask(
    mask: torch.Tensor,
    target_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Resize a soft 3D mask while preserving occupancy semantics."""
    orig_dim = mask.dim()
    if orig_dim == 3:
        mask_5d = mask.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 5:
        mask_5d = mask
    else:
        raise ValueError(
            f"Expected mask with shape [T,H,W] or [B,1,T,H,W], got {tuple(mask.shape)}"
        )

    target_size = tuple(int(v) for v in target_size)
    orig_dtype = mask_5d.dtype
    mask_5d = mask_5d.to(dtype=torch.float32).clamp_(0.0, 1.0)
    in_size = tuple(int(v) for v in mask_5d.shape[-3:])

    if in_size == target_size:
        resized = mask_5d
    elif all(src >= dst for src, dst in zip(in_size, target_size)):
        if all(src % dst == 0 for src, dst in zip(in_size, target_size)):
            kernel = tuple(src // dst for src, dst in zip(in_size, target_size))
            resized = F.avg_pool3d(mask_5d, kernel_size=kernel, stride=kernel)
        else:
            resized = F.adaptive_avg_pool3d(mask_5d, output_size=target_size)
    else:
        resized = F.interpolate(
            mask_5d,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

    resized = resized.clamp_(0.0, 1.0).to(dtype=orig_dtype)
    if orig_dim == 3:
        return resized[0, 0]
    return resized
