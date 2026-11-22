"""三种样本选择器：random / k-center（core-set 贪心）/ RDSS（GKHR）。

共同约束（写入报告，任何脚本不得绕过）：
  - 输入只有**特征矩阵**与预算 m，函数签名里没有标签参数，选择过程无法读取标签；
  - 输出为训练池内行号（0..n-1），不重复、数量恰为 m；
  - random / k-center 依赖选择种子；RDSS 完全确定性（同特征同 α 同 γ 必得同一集合）。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..kernel import gaussian_kernel

MuDiagonal = Literal["include", "exclude"]


def alpha_auto(m: int) -> float:
    """论文默认 α = 1 − 1/√m（m=250 → 0.9368，m=1000 → 0.9684）。"""
    if m <= 0:
        raise ValueError(f"预算 m 必须为正，收到 {m}")
    return float(1.0 - 1.0 / np.sqrt(m))


def random_select(n_pool: int, m: int, seed: int) -> np.ndarray:
    """无放回均匀随机采样（固定种子可复现）。"""
    if not 0 < m <= n_pool:
        raise ValueError(f"预算非法: m={m}, n_pool={n_pool}")
    idx = np.random.RandomState(seed).choice(n_pool, size=m, replace=False)
    return np.sort(idx).astype(np.int64)


def kcenter_greedy(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    """core-set 贪心（课程新增基线，A7-3）：每次选离已选集合最远的点。

    第 1 个中心由 seed 随机决定，故结果依赖选择种子。
    复杂度 O(m·n·d)，用 float32 + 单向量内积实现，避免每次重算全距离矩阵。
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if not 0 < m <= n:
        raise ValueError(f"预算非法: m={m}, n={n}")

    rng = np.random.RandomState(seed)
    first = int(rng.randint(n))
    selected = np.empty(m, dtype=np.int64)
    selected[0] = first

    # 到已选集合的最近距离平方；先全部置 inf，再按已选点更新
    min_d2 = np.full(n, np.inf, dtype=np.float32)
    min_d2 = _update_min_d2(X, first, min_d2)
    min_d2[first] = -np.inf  # 已选点不再参与 argmax

    for p in range(1, m):
        j = int(np.argmax(min_d2))
        selected[p] = j
        min_d2 = _update_min_d2(X, j, min_d2)
        min_d2[j] = -np.inf
    return selected


def _update_min_d2(X: np.ndarray, j: int, min_d2: np.ndarray) -> np.ndarray:
    """d2_i ← min(d2_i, ||x_i − x_j||²)，用内积展开，float32 单遍。"""
    xj = X[j]
    d2 = np.einsum("ij,ij->i", X, X) - 2.0 * (X @ xj) + float(xj @ xj)
    np.maximum(d2, 0.0, out=d2)
    return np.minimum(min_d2, d2, out=min_d2)


def compute_mu(
    X: np.ndarray,
    gamma: float,
    block_rows: int = 2000,
    diagonal: MuDiagonal = "include",
) -> np.ndarray:
    """μ(x_i) = 与全池的平均核相似度，分块计算，不落地 n×n 核矩阵。

    diagonal="include"：按论文 Algorithm 1 含自身 k(x_i,x_i)=1；
    diagonal="exclude"：按官方 RDSS.py L22 去掉对角线，分母相应变为 n−1。
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    mu = np.zeros(n, dtype=np.float64)
    for start in range(0, n, block_rows):
        block = X[start : start + block_rows]
        mu[start : start + block_rows] = gaussian_kernel(block, X, gamma).sum(axis=1, dtype=np.float64)
    mu /= float(n)
    if diagonal == "exclude":
        # 去掉自身的 1.0，分母 n-1：μ_ex = (n·μ_in − 1) / (n − 1)
        mu = (mu * n - 1.0) / float(n - 1)
    elif diagonal != "include":
        raise ValueError(f"未知 diagonal 口径: {diagonal}")
    return mu


def gkhr(
    X: np.ndarray,
    m: int,
    alpha: float,
    gamma: float,
    diagonal: MuDiagonal = "include",
    block_rows: int = 2000,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """论文 Algorithm 1（GKHR）的 O(m·n) 增量实现（§A4 骨架）。

    读法：S 是"与已选集合的平均相似度"，μ 是"与全池的平均相似度"；
    min(S − αμ) 同时压低 S（别与已选重复）并抬高 μ（仍在分布中心）；
    p=1 时 S_0=0，第一步等价于选 μ 最大者（全池最"中心"的样本）。
    第 5 行是增量更新：β_p = 1/p，S_p = (1−β)S_{p−1} + β·k(x*, ·)，避免每轮重算。
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if not 0 < m <= n:
        raise ValueError(f"预算非法: m={m}, n={n}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1]，收到 {alpha}")
    if gamma <= 0:
        raise ValueError(f"gamma 必须 > 0，收到 {gamma}")

    if mu is None:
        mu = compute_mu(X, gamma, block_rows=block_rows, diagonal=diagonal)

    S = np.zeros(n, dtype=np.float32)
    selected = np.zeros(n, dtype=bool)
    out = np.empty(m, dtype=np.int64)

    for p in range(1, m + 1):
        cand = np.where(~selected)[0]
        j = int(cand[np.argmin(S[cand] - alpha * mu[cand])])
        selected[j] = True
        out[p - 1] = j
        if p == m:
            break
        beta = 1.0 / p  # p 轮后新集合规模为 p+1，但增量系数按行 5 写作 1/p
        S[cand] = (1.0 - beta) * S[cand] + beta * gaussian_kernel(X[j : j + 1], X[cand], gamma).ravel()
    return out
