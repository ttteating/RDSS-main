"""P1 单元测试：数据划分与核带宽。

本地无数据集也可运行（纯 numpy）：
    python tests/test_p1_kernel_split.py        # 无 pytest 时走 __main__
    pytest tests/test_p1_kernel_split.py -q     # 有 pytest 时
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdss.data.split import CIFAR10_TEST_SIZE, make_splits  # noqa: E402
from rdss.data.split import test_indices as make_test_indices  # noqa: E402
from rdss.kernel import gaussian_kernel, l2_normalize, median_sq_dist, sigma_from_features, sq_dists  # noqa: E402


def test_split_sizes_disjoint() -> None:
    s = make_splits(n_total=50000, val_size=5000, seed=0)
    val, pool = s["val"], s["train_pool"]
    assert val.size == 5000 and pool.size == 45000
    assert np.intersect1d(val, pool).size == 0
    assert np.array_equal(np.sort(np.concatenate([val, pool])), np.arange(50000))
    assert np.unique(val).size == val.size and np.unique(pool).size == pool.size
    assert make_test_indices().size == CIFAR10_TEST_SIZE


def test_split_deterministic() -> None:
    a = make_splits(seed=0)
    b = make_splits(seed=0)
    c = make_splits(seed=1)
    assert np.array_equal(a["val"], b["val"])
    assert np.array_equal(a["train_pool"], b["train_pool"])
    assert not np.array_equal(a["val"], c["val"]), "不同 seed 必须给出不同划分"


def test_gaussian_kernel_properties() -> None:
    rng = np.random.RandomState(0)
    X = l2_normalize(rng.randn(64, 32).astype(np.float32))
    K = gaussian_kernel(X, gamma=1.7)
    assert K.shape == (64, 64)
    assert np.allclose(K, K.T, atol=1e-6), "核矩阵必须对称"
    assert np.allclose(np.diag(K), 1.0, atol=1e-6), "k(x,x)=1"
    assert K.min() > 0.0 and K.max() <= 1.0 + 1e-6


def test_median_sq_matches_bruteforce() -> None:
    rng = np.random.RandomState(42)
    X = l2_normalize(rng.randn(300, 16).astype(np.float32))
    D2 = sq_dists(X)
    np.fill_diagonal(D2, np.nan)
    brute = float(np.nanmedian(D2))
    ours, meta = median_sq_dist(X, block_rows=64, bins=32768)
    tol = 3.0 * meta["bin_width"]
    assert abs(ours - brute) <= tol, f"中位数估计偏差过大: ours={ours:.6f} brute={brute:.6f} tol={tol:.2e}"
    assert meta["pairs"] == 300 * 299, "必须排除自身距离"


def test_median_block_consistency_and_sampling() -> None:
    rng = np.random.RandomState(7)
    X = l2_normalize(rng.randn(500, 24).astype(np.float32))
    a, _ = median_sq_dist(X, block_rows=37, bins=8192)
    b, _ = median_sq_dist(X, block_rows=500, bins=8192)
    assert abs(a - b) < 1e-6, "分块大小不应改变结果（同一 bin 数下）"

    s1 = sigma_from_features(X, block_rows=100, bins=8192)
    s2 = sigma_from_features(X, block_rows=100, bins=8192)
    assert s1["sigma"] == s2["sigma"], "同输入两次 σ 必须完全相同"
    assert abs(s1["gamma"] - 1.0 / s1["sigma"]) < 1e-9
    assert s1["sigma"] > 0

    sub = sigma_from_features(X, bins=8192, sample_size=200, sample_seed=0)
    assert sub["sample_size"] == 200 and sub["n"] == 200
    sub2 = sigma_from_features(X, bins=8192, sample_size=200, sample_seed=0)
    assert sub["sigma"] == sub2["sigma"], "下采样必须由 seed 完全决定"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
