"""CLIP 特征抽取模块（P1）。"""

from .clip import TAKE_CHOICES, build_transform, encode_batch, extract_features, load_clip_vision

__all__ = [
    "TAKE_CHOICES",
    "build_transform",
    "encode_batch",
    "extract_features",
    "load_clip_vision",
]
