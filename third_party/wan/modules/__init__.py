# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Minimal module package for WEM vendored Wan components.

Do not import heavy submodules here. WEM imports the required implementations
directly from their files, and eager imports can initialize CUDA or optional
backends before the caller is ready.
"""

__all__ = [
    'Wan2_1_VAE',
    'Wan2_2_VAE',
    'WanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'flash_attention',
]
