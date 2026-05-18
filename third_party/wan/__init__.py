# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Minimal Wan package shim used by WEM.

Keep this module intentionally lightweight. Importing third_party.wan is part
of normal submodule resolution, so eager imports of the original Wan entrypoints
would pull optional dependencies such as decord even when WEM only needs
modules.t5, modules.vae2_2, distributed.fsdp, or utils.utils.
"""

__all__ = ["configs", "distributed", "modules", "utils"]
