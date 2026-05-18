# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Minimal utils package for WEM vendored Wan components.

The original Wan package exports flow-matching solver helpers here. This
vendored subset only needs utils.py; importing missing solver files here would
break WEM generation before save_video/best_output_size can be loaded.
"""

__all__ = [
    'utils',
]
