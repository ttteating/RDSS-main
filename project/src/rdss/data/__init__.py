"""CIFAR-10 数据与划分模块。

只在此处 re-export 划分相关符号：`rdss.data.split` 仅依赖 numpy，
因此 `scripts/split_val.py` / `tests/test_p1_kernel_split.py` 可在无 torch 的环境运行。
数据管线（`rdss.data.cifar`）需要 torchvision，按需显式导入。
"""

from .split import (
    CIFAR10_TEST_SIZE,
    CIFAR10_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    make_splits,
    test_indices,
)

__all__ = [
    "CIFAR10_TRAIN_SIZE",
    "CIFAR10_TEST_SIZE",
    "DEFAULT_VAL_SIZE",
    "make_splits",
    "test_indices",
]
