"""P2-3：RDSS / GKHR（论文 Algorithm 1，O(m·n) 增量实现）。

用法：
    python scripts/selection_rdss.py --budget 250 \
        --feature-path features/cifar10_trainpool_clip.npy \
        --gamma-path configs/kernel.yaml --alpha auto \
        --output selections/rdss_m250.json
    # α 消融（P5）
    python scripts/selection_rdss.py --budget 250 --alpha 0 ...

口径：
    - α 默认 1 − 1/√m；α=1 → 只剩代表性，α=0 → 只剩多样性；
    - γ 取 configs/kernel.yaml 的 median heuristic（P1 回填）；未回填时必须显式 --gamma；
    - μ 的对角线口径默认按论文（include），可 --mu-diagonal exclude 复现官方行为；
    - 确定性：无选择种子，同输入必得同一索引集合。
只读特征矩阵，不读图像/标签。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rdss.selection import alpha_auto, append_timing, gkhr, save_selection  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import sha1_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--gamma-path", default="configs/kernel.yaml")
    parser.add_argument("--alpha", default="auto", help="'auto' = 1-1/sqrt(m)，或显式浮点数（P5 消融）")
    parser.add_argument("--gamma", type=float, default=None, help="覆盖 kernel.yaml 的 γ（复现官方 0.5 时用）")
    parser.add_argument("--mu-diagonal", choices=["include", "exclude"], default=None)
    parser.add_argument("--block-rows", type=int, default=0, help="0 = 取 kernel.yaml: kernel.block_rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-csv", default="results/p2/selection_timing.csv")
    args = parser.parse_args()

    feat_path = Path(args.feature_path)
    feat_path = feat_path if feat_path.is_absolute() else PROJECT_ROOT / feat_path
    if not feat_path.exists():
        print(f"[FAIL] 特征文件不存在: {feat_path}")
        print("       请先运行: python scripts/extract_clip_features.py --pool trainpool ...")
        return 1

    X = np.load(feat_path)
    n_pool = X.shape[0]
    if args.budget > n_pool:
        print(f"[FAIL] 预算 m={args.budget} 超过池大小 n={n_pool}")
        return 1

    kernel_cfg = load_yaml(args.gamma_path)
    kcfg = kernel_cfg["kernel"]
    gamma = args.gamma if args.gamma is not None else kcfg.get("gamma")
    if gamma is None:
        print("[FAIL] configs/kernel.yaml 的 gamma 尚未回填（P1 未跑 compute_sigma.py）")
        print("       先跑: python scripts/compute_sigma.py --feature-path <特征> --output configs/kernel.yaml")
        print("       或用 --gamma 0.5 复现官方默认行为（需在报告标注为差异实验）")
        return 1
    gamma = float(gamma)

    alpha = alpha_auto(args.budget) if args.alpha == "auto" else float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        print(f"[FAIL] alpha 必须在 [0, 1]，收到 {alpha}")
        return 1

    diagonal = args.mu_diagonal or kernel_cfg.get("mu", {}).get("diagonal", "include")
    block_rows = args.block_rows or int(kcfg.get("block_rows", 2000))

    print(f"RDSS/GKHR  n={n_pool}  d={X.shape[1]}  m={args.budget}  α={alpha:.6f}  γ={gamma:.6f}  μ对角线={diagonal}")

    t0 = time.perf_counter()
    indices = gkhr(X, args.budget, alpha, gamma, diagonal=diagonal, block_rows=block_rows)
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    sigma = kcfg.get("sigma")
    meta = save_selection(
        out,
        indices,
        method="rdss",
        budget=args.budget,
        n_pool=n_pool,
        elapsed_sec=elapsed,
        extra={
            "selector_seed": None,
            "alpha": alpha,
            "alpha_rule": "1-1/sqrt(m)" if args.alpha == "auto" else "explicit",
            "gamma": gamma,
            "sigma": None if sigma is None else float(sigma),
            "mu_diagonal": diagonal,
            "feature_file": feat_path.name,
            "feature_sha1": sha1_file(feat_path),
            "implementation": "GKHR Algorithm 1, incremental O(m·n), float32",
            "label_used": False,
        },
    )
    append_timing(
        PROJECT_ROOT / args.timing_csv,
        {
            "created": meta["created"],
            "method": "rdss",
            "budget": args.budget,
            "n_pool": n_pool,
            "selector_seed": "",
            "alpha": f"{alpha:.6f}",
            "gamma": f"{gamma:.6f}",
            "elapsed_sec": meta["elapsed_sec"],
            "output": out.name,
        },
    )

    print(f"[OK] rdss  α={alpha:.4f}  γ={gamma:.4f}  用时 {elapsed:.2f}s")
    print(f"     {out.name}  sha1={meta['file_sha1'][:12]}…")
    print(f"     索引校验通过：数量={meta['n_selected']}，无重复，范围 [0, {n_pool})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
