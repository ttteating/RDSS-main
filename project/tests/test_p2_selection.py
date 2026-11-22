"""P2 单元测试：三种选择器的正确性、确定性与索引合法性。

无需数据集 / CLIP：
    python tests/test_p2_selection.py
    pytest tests/test_p2_selection.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdss.kernel import gaussian_kernel  # noqa: E402
from rdss.selection import alpha_auto, compute_mu, gkhr, kcenter_greedy, random_select  # noqa: E402
from rdss.selection.record import validate_indices  # noqa: E402


def _toy(n: int = 200, d: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(n, d).astype(np.float32)


def test_random_contract() -> None:
    n, m = 500, 50
    a = random_select(n, m, seed=42)
    b = random_select(n, m, seed=42)
    c = random_select(n, m, seed=7)
    assert a.size == m and np.unique(a).size == m
    assert a.min() >= 0 and a.max() < n
    assert np.array_equal(a, b), "同种子必须可复现"
    assert not np.array_equal(a, c), "不同种子应给出不同集合"


def test_kcenter_contract_and_greedy() -> None:
    X = _toy(n=300, d=4, seed=1)
    m, seed = 20, 42
    idx = kcenter_greedy(X, m, seed)
    assert idx.size == m and np.unique(idx).size == m
    assert idx[0] == np.random.RandomState(seed).randint(X.shape[0]), "第一个中心必须由种子决定"
    assert np.array_equal(idx, kcenter_greedy(X, m, seed)), "同种子必须可复现"

    # 贪心性质：每一步新中心到已选集合的距离，等于当前全池最大最近距离
    chosen = idx[:1].tolist()
    for p in range(1, m):
        d_all = np.min(
            np.linalg.norm(X[:, None, :] - X[np.array(chosen)[None, :], :], axis=2), axis=1
        )
        expect = int(np.argmax(np.where(np.isin(np.arange(X.shape[0]), chosen), -np.inf, d_all)))
        assert idx[p] == expect, f"第 {p} 步不是最远点"
        chosen.append(int(idx[p]))


def test_gkhr_contract_and_determinism() -> None:
    X = _toy(n=150, d=8, seed=2)
    a = gkhr(X, 30, alpha=alpha_auto(30), gamma=1.0)
    b = gkhr(X, 30, alpha=alpha_auto(30), gamma=1.0)
    assert a.size == 30 and np.unique(a).size == 30
    assert np.array_equal(a, b), "RDSS 必须完全确定"


def test_gkhr_matches_naive_algorithm1() -> None:
    """与"每轮从零重算 S 与 μ"的朴素 Algorithm 1 比对，验证增量更新等价。"""
    X = _toy(n=120, d=16, seed=3)
    m, alpha, gamma = 40, 0.9, 1.3
    mu = compute_mu(X, gamma, diagonal="include")
    fast = gkhr(X, m, alpha, gamma, mu=mu)

    selected: list[int] = []
    naive = np.empty(m, dtype=np.int64)
    K = gaussian_kernel(X, gamma=gamma)
    for p in range(1, m + 1):
        cand = np.array([i for i in range(X.shape[0]) if i not in selected], dtype=np.int64)
        S = K[np.ix_(cand, selected)].mean(axis=1) if selected else np.zeros(cand.size)
        j = int(cand[np.argmin(S - alpha * mu[cand])])
        naive[p - 1] = j
        selected.append(j)

    assert np.array_equal(fast, naive), f"增量实现与朴素实现不一致: {fast[:8]} vs {naive[:8]}"


def test_first_pick_is_most_central() -> None:
    """论文读法：p=1 时 S_0=0，第一步选 μ 最大者。"""
    X = _toy(n=400, d=12, seed=4)
    gamma = 1.0
    mu = compute_mu(X, gamma, diagonal="include")
    assert gkhr(X, 5, alpha=0.9, gamma=gamma)[0] == int(np.argmax(mu))


def test_mu_diagonal_variants() -> None:
    X = _toy(n=200, d=6, seed=5)
    gamma = 0.8
    inc = compute_mu(X, gamma, diagonal="include")
    exc = compute_mu(X, gamma, diagonal="exclude")
    n = X.shape[0]
    assert np.allclose(exc, (inc * n - 1.0) / (n - 1), atol=1e-6)


def test_validate_indices_rejects_bad_input() -> None:
    ok = validate_indices([0, 5, 9], n_pool=10, m=3)
    assert ok.tolist() == [0, 5, 9]
    for bad, why in (([0, 0, 9], "重复"), ([0, 5, 10], "越界"), ([0, 5], "数量不符")):
        try:
            validate_indices(bad, n_pool=10, m=3)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f"未拦截非法索引（{why}）: {bad}")


def test_alpha_auto_table() -> None:
    assert abs(alpha_auto(40) - 0.8419) < 5e-4
    assert abs(alpha_auto(250) - 0.9368) < 5e-4
    assert abs(alpha_auto(1000) - 0.9684) < 5e-4
    assert abs(alpha_auto(4000) - 0.9842) < 5e-4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
