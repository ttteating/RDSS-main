"""CIFAR-10 固定划分（§A5.2）：50000 train → 5000 val + 45000 train pool；test 固定为 0..9999。

要点：
  - 只用 `np.random.RandomState(seed)`（legacy 随机流，跨 numpy 版本可复现）；
  - 索引语义：**原始 CIFAR-10 train split 内的位置**（0..49999），不是训练池内行号；
  - 划分只做一次，三方法共享（random / k-center / RDSS 禁止各自重划）；
  - 第 r 行特征对应 `trainpool_indices_seed0.json` 的第 r 个元素（由 scripts/split_val.py 落盘固定）。

本模块只依赖 numpy，保证在无 torch 的机器上也能生成/复算划分索引。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

CIFAR10_TRAIN_SIZE = 50000
CIFAR10_TEST_SIZE = 10000
DEFAULT_VAL_SIZE = 5000


def make_splits(
    n_total: int = CIFAR10_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """随机置换后切出验证集与训练池，两两不交且并集为全集。

    返回 {"val": [val_size], "train_pool": [n_total - val_size]}，均为升序 int64。
    """
    n_total = int(n_total)
    val_size = int(val_size)
    if not 0 < val_size < n_total:
        raise ValueError(f"val_size 必须在 (0, n_total) 内，收到 val_size={val_size}, n_total={n_total}")

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n_total)
    # 排序只为让索引文件可读、sha1 稳定；划分本身仍由 seed 完全决定
    val = np.sort(perm[:val_size]).astype(np.int64, copy=False)
    train_pool = np.sort(perm[val_size:]).astype(np.int64, copy=False)
    return {"val": val, "train_pool": train_pool}


def test_indices(n_total: int = CIFAR10_TEST_SIZE) -> np.ndarray:
    """官方测试集索引（CIFAR-10 test split 0..9999，不做任何随机划分）。"""
    return np.arange(int(n_total), dtype=np.int64)
