"""冻结的 CLIP ViT-B/32 视觉塔：预处理、单批编码、全池特征抽取（P1）。

口径（与 configs/data.yaml: features 一致，三方法必须同一种）：
  - 预处理：Resize(224, 224) + ToTensor，不做像素 Normalize（CLIP 内部自带）；
  - 取法：take=cls_last_hidden → final [CLS]（论文口径）；pooler_output → 官方 get_cifar_feature.py 口径；
  - 工程约定：默认对输出特征做 L2 归一化（否则 σ 失去意义），可用 l2_normalize=False 关闭；
  - 行序：输出第 r 行 = 索引文件第 r 个元素对应的图片（选择器输出的 r 即训练池内位置）。

`extract_features` 直接丢弃 DataLoader 的标签（loader 可返回 (img, label) 或 img），
保证抽特征阶段不可能使用标签信息。
"""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn.functional as F
from torchvision import transforms

TAKE_CHOICES = ("cls_last_hidden", "pooler_output")
DeviceLike = Union[str, torch.device]


def build_transform(resize: int = 224) -> transforms.Compose:
    """Resize + ToTensor；不加 Normalize（CLIP 的 processor 已在内部归一化）。"""
    return transforms.Compose([transforms.Resize((int(resize), int(resize))), transforms.ToTensor()])


def load_clip_vision(
    model_name: str = "openai/clip-vit-base-patch32",
    device: DeviceLike = "cpu",
) -> torch.nn.Module:
    """加载 CLIP 视觉塔并冻结：eval 模式 + 全部参数 requires_grad=False。"""
    from transformers import CLIPVisionModel

    model = CLIPVisionModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def encode_batch(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    take: str = "cls_last_hidden",
) -> torch.Tensor:
    """单批像素 [B, 3, H, W] → 特征 [B, 768]（float32）。"""
    if take not in TAKE_CHOICES:
        raise ValueError(f"take 必须是 {TAKE_CHOICES} 之一，收到 {take!r}")
    pixel_values = pixel_values.to(next(model.parameters()).device)
    outputs = model(pixel_values=pixel_values)
    if take == "cls_last_hidden":
        feats = outputs.last_hidden_state[:, 0, :]
    else:
        feats = outputs.pooler_output
    return feats.float()


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: Iterable,
    device: DeviceLike = "cpu",
    take: str = "cls_last_hidden",
    l2_normalize: bool = True,
    amp: bool = True,
) -> torch.Tensor:
    """遍历 loader 抽取全部特征，返回 [N, 768] float32（可选 L2 归一化）。"""
    if take not in TAKE_CHOICES:
        raise ValueError(f"take 必须是 {TAKE_CHOICES} 之一，收到 {take!r}")
    dev = torch.device(device)
    chunks = []
    for batch in loader:
        # DataLoader 可能返回 (img, label) 或 img；标签一律丢弃
        pixel_values = batch[0] if isinstance(batch, (tuple, list)) else batch
        pixel_values = pixel_values.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, enabled=bool(amp) and dev.type == "cuda"):
            chunks.append(encode_batch(model, pixel_values, take=take))
    if not chunks:
        raise ValueError("loader 为空，无法抽取特征")

    feats = torch.cat(chunks, dim=0)
    if l2_normalize:
        feats = F.normalize(feats, p=2, dim=1)
    return feats
