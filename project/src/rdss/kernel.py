"""高斯核与 median heuristic（§A2）。

要点：
  - σ = 全池 pairwise **距离**的中位数；γ = 1/σ；k(x,y) = exp(-||x-y||²/σ)。
  - 特征先做 L2 归一化再算距离（工程约定，写入 configs/kernel.yaml）。
  - n=45000 时全核矩阵约 8.1GB，故中位数用**分块 + 直方图**统计，不落地全矩阵：
    距离的平方落在 [0, 4]（归一化后），用 32768 个等宽 bin 累积计数，
    再取累积分布 50% 分位点；σ = sqrt(中位数平方距离)（开方单调，不改变中位数位置）。
  - 直方图引入的误差 ≤ 一个 bin 宽（默认 4/32768 ≈ 1.2e-4），报告里记为中位数估计精度。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, eps)


def sq_dists(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """欧氏距离平方矩阵；Y=None 时为 X 的自距离矩阵。小规模用（P2/GKHR 逐轮调用）。"""
    X = np.asarray(X, dtype=np.float32)
    Y = X if Y is None else np.asarray(Y, dtype=np.float32)
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    y2 = np.einsum("ij,ij->i", Y, Y)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)  # 数值误差可能给出 -1e-7
    return d2


def gaussian_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, gamma: float = 1.0) -> np.ndarray:
    """k(x,y) = exp(-gamma * ||x-y||²)，与官方 RDSS.gaussiankernel 语义一致（但默认 γ 不同）。"""
    if gamma <= 0:
        raise ValueError(f"gamma 必须 > 0，收到 {gamma}")
    return np.exp(-gamma * sq_dists(X, Y)).astype(np.float32)


def median_sq_dist(
    X: np.ndarray,
    block_rows: int = 2000,
    bins: int = 32768,
    max_sq: Optional[float] = None,
    sample_size: int = 0,
    sample_seed: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """分块统计 pairwise 平方距离的中位数（排除自身距离 0）。

    sample_size > 0 且小于 n 时先随机下采样（降级方案，必须记录 seed）。
    返回 (中位数平方距离, 统计信息)。
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"特征必须是二维 [N, d]，收到 {X.shape}")
    n_orig = X.shape[0]
    n = n_orig
    if sample_size and sample_size < n:
        idx = np.sort(np.random.RandomState(sample_seed).choice(n, size=sample_size, replace=False))
        X = X[idx]
        n = X.shape[0]
    if n < 2:
        raise ValueError("至少需要 2 个样本才能估计中位数")

    if max_sq is None:
        # ||x-y||² ≤ 4·max||x||²（Cauchy-Schwarz），保证不截断
        max_sq = float(4.0 * np.max(np.einsum("ij,ij->i", X, X)))
        max_sq = max(max_sq, 1e-12)

    counts = np.zeros(bins, dtype=np.int64)
    for start in range(0, n, block_rows):
        block = X[start : start + block_rows]
        d2 = sq_dists(block, X)
        np.clip(d2, 0.0, max_sq, out=d2)
        hist, _ = np.histogram(d2, bins=bins, range=(0.0, max_sq))
        counts += hist
        counts[0] -= block.shape[0]  # 去掉 block 与自身的距离（恰为 0，落在第 0 个 bin）

    total = int(counts.sum())
    target = total / 2.0
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, bins - 1)
    before = int(cum[i - 1]) if i > 0 else 0
    width = max_sq / bins
    frac = (target - before) / max(int(counts[i]), 1)
    median_sq = float(i * width + frac * width)
    meta = {
        "median_sq_dist": median_sq,
        "n": float(n),
        "n_orig": float(n_orig),
        "pairs": float(total),
        "bins": float(bins),
        "max_sq_dist": float(max_sq),
        "bin_width": float(width),
        "sampled": float(1 if n < n_orig else 0),
    }
    return median_sq, meta


def sigma_from_features(
    X: np.ndarray,
    *,
    normalize: bool = True,
    block_rows: int = 2000,
    bins: int = 32768,
    sample_size: int = 0,
    sample_seed: int = 0,
) -> Dict[str, float]:
    """计算 σ（median distance）与 γ=1/σ；返回可直接写入 kernel.yaml 的统计。"""
    feats = l2_normalize(X) if normalize else np.asarray(X, dtype=np.float32)
    median_sq, meta = median_sq_dist(
        feats, block_rows=block_rows, bins=bins, sample_size=sample_size, sample_seed=sample_seed
    )
    sigma = float(np.sqrt(median_sq))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"σ 估计失败：median_sq={median_sq}")
    return {
        "sigma": sigma,
        "gamma": 1.0 / sigma,
        "median_sq_dist": median_sq,
        "n": meta["n"],
        "pairs": meta["pairs"],
        "bins": meta["bins"],
        "bin_width": meta["bin_width"],
        "max_sq_dist": meta["max_sq_dist"],
        "l2_normalized": 1.0 if normalize else 0.0,
        "sample_size": float(sample_size),
        "sample_seed": float(sample_seed),
    }
