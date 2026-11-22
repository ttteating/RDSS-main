"""CIFAR-10 数据管线（P3）：有标签 / 无标签 / 评估数据集，以及 FixMatch 的弱-强增强。

口径（与 configs/fixmatch.yaml、论文 §A5.1 一致）：
  - 弱增强：RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize；
  - 强增强：RandAugment(num_ops=2, magnitude=10) → 同上；
  - 无标签集只返回 (弱增强, 强增强)，**不返回标签**（选择/训练阶段避免标签泄漏）。

索引语义：本模块所有 indices 都是**原始 CIFAR-10 split 内的位置**；
训练池内行号 → 原始索引的映射由 `splits/trainpool_indices_seed0.json` 在脚本层完成。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def weak_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def strong_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    """确定性预处理：验证/测试集只做 ToTensor + Normalize，不做任何随机增强。"""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


class _SubsetDataset(Dataset):
    """按索引取 base[i]（base 需返回 (PIL Image, label)）的只读视图。"""

    def __init__(self, base: Dataset, indices) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError(f"indices 必须是一维，收到 shape={self.indices.shape}")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(base)):
            raise ValueError(f"indices 越界：范围 [{self.indices.min()}, {self.indices.max()}]，base 大小 {len(base)}")

    def __len__(self) -> int:
        return int(self.indices.size)

    def _load(self, i: int) -> Tuple[Image.Image, int]:
        img, label = self.base[int(self.indices[i])]
        return img, int(label)


class LabeledDataset(_SubsetDataset):
    """有标签训练集：返回 (transform(img), label)。"""

    def __init__(self, base: Dataset, indices, transform) -> None:
        super().__init__(base, indices)
        self.transform = transform

    def __getitem__(self, i: int):
        img, label = self._load(i)
        return self.transform(img), label


class EvalDataset(_SubsetDataset):
    """验证/测试集：返回 (eval_transform(img), label)。"""

    def __init__(self, base: Dataset, indices, transform) -> None:
        super().__init__(base, indices)
        self.transform = transform

    def __getitem__(self, i: int):
        img, label = self._load(i)
        return self.transform(img), label


class UnlabeledDataset(_SubsetDataset):
    """无标签训练池：返回 (弱增强, 强增强)，刻意不返回标签。"""

    def __init__(self, base: Dataset, indices, weak, strong) -> None:
        super().__init__(base, indices)
        self.weak = weak
        self.strong = strong

    def __getitem__(self, i: int):
        img, _ = self._load(i)
        return self.weak(img), self.strong(img)


class FakeCIFAR(Dataset):
    """随机像素烟测数据集：接口与 torchvision CIFAR10 对齐（返回 PIL Image, label）。

    仅用于 `--fake-data` 验证训练循环/断点，产物不得写入正式结果。
    """

    classes = CIFAR10_CLASSES

    def __init__(self, n: int = 800, seed: int = 0, num_classes: int = 10) -> None:
        rng = np.random.RandomState(int(seed))
        self.images = rng.randint(0, 256, size=(int(n), 32, 32, 3), dtype=np.uint8)
        self.targets = rng.randint(0, int(num_classes), size=int(n)).astype(np.int64)

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, i: int) -> Tuple[Image.Image, int]:
        return Image.fromarray(self.images[i]), int(self.targets[i])


def split_labeled_unlabeled(pool: np.ndarray, labeled_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把训练池拆成有标签 / 无标签的**原始索引**。

    参数：
        pool        —— 训练池的原始 CIFAR 索引（长度 n）
        labeled_pos —— 选择器输出的**池内行号** r（0..n-1），可有重复则报错

    返回：
        (labeled_orig, unlabeled_orig) = (pool[labeled_pos], pool[其余行号])
        有标签集合保持选择顺序；无标签集合保持池内原顺序。
    """
    pool = np.asarray(pool, dtype=np.int64)
    pos = np.asarray(labeled_pos, dtype=np.int64)
    if pool.ndim != 1:
        raise ValueError(f"pool 必须是一维，收到 shape={pool.shape}")
    if pos.ndim != 1:
        raise ValueError(f"labeled_pos 必须是一维，收到 shape={pos.shape}")
    if pos.size and (pos.min() < 0 or pos.max() >= pool.size):
        raise ValueError(f"labeled_pos 越界：必须在 [0, {pool.size})，收到 [{pos.min()}, {pos.max()}]")
    if np.unique(pos).size != pos.size:
        raise ValueError("labeled_pos 含重复行号：选择结果不允许重复")

    mask = np.ones(pool.size, dtype=bool)
    mask[pos] = False
    return pool[pos], pool[mask]
