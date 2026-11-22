"""P2-5：二维 toy 直觉图（复现官方 generate_visualization_data.py + visualization.py 的思路）。

用法：
    python scripts/viz_toy_selection.py --budget 100 --output figures/toy_selection.png

做三件事：
    1. 用与官方 generate_visualization_data.py 完全相同的生成过程（np.random.seed(42)，
       10 个高斯团 × 500 点，二维）重新生成 toy 数据（并存一份 pkl 便于复核）；
    2. 在 toy 数据上跑三种选择器（m=100，α=1−1/√100=0.9，σ 用 median heuristic）；
    3. 画 3 联图：灰点=全池，黑点=入选。用于验证"代表性/多样性"直觉是否成立。

标签只用于**画图区分 10 个团**，不参与任何选择（选择器函数里没有标签参数）。
不依赖 CLIP 特征与数据集，可在本地直接运行。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rdss.kernel import sigma_from_features  # noqa: E402
from rdss.selection import alpha_auto, gkhr, kcenter_greedy, random_select  # noqa: E402


def generate_toy(seed: int = 42, n_points: int = 5000, n_distributions: int = 10):
    """与官方 generate_visualization_data.py 逐行等价的生成逻辑（seed 固定）。"""
    rng = np.random.RandomState(seed)
    data, labels = [], []
    for i in range(n_distributions):
        mean_x, mean_y = rng.uniform(-12, 12), rng.uniform(-12, 12)
        std_x, std_y = rng.uniform(0.5, 2), rng.uniform(0.5, 2)
        x = rng.normal(mean_x, std_x, n_points // n_distributions)
        y = rng.normal(mean_y, std_y, n_points // n_distributions)
        data.extend(list(zip(x, y)))
        labels.extend([i] * (n_points // n_distributions))
    return np.asarray(data, dtype=np.float32), np.asarray(labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="random / k-center 的选择种子")
    parser.add_argument("--output", default="figures/toy_selection.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, labels = generate_toy()
    m = args.budget
    alpha = alpha_auto(m)
    stats = sigma_from_features(X, normalize=False)
    gamma = stats["gamma"]

    out = Path(args.output)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    toy_dir = out.parent / "toy"
    toy_dir.mkdir(parents=True, exist_ok=True)
    with (toy_dir / "visualization.pkl").open("wb") as f:
        pickle.dump((X, labels), f)

    picks = {
        "RDSS / GKHR (ours)": gkhr(X, m, alpha, gamma),
        "Random": random_select(X.shape[0], m, args.seed),
        "k-center": kcenter_greedy(X, m, args.seed),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, idx) in zip(axes, picks.items()):
        ax.scatter(X[:, 0], X[:, 1], s=4, c="0.75", linewidths=0, label="pool (n=5000)")
        ax.scatter(X[idx, 0], X[idx, 1], s=12, c="k", linewidths=0, label=f"selected (m={m})")
        covered = np.unique(labels[idx]).size
        ax.set_title(f"{name}\nclusters covered: {covered}/10", fontsize=10)
        ax.set_xlabel("x (toy coordinate, a.u.)")
        ax.set_ylabel("y (toy coordinate, a.u.)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"2D toy selection (m={m}, alpha={alpha:.3f}, gamma={gamma:.3f}, seed={args.seed})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[OK] toy 图已保存: {out}")
    for name, idx in picks.items():
        print(f"     {name:<20} 覆盖团数 {np.unique(labels[idx]).size}/10  索引数 {idx.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
