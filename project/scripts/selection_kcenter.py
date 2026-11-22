"""P2-2：k-center（core-set 贪心）基线 —— 课程新增，非论文原基线（A7-3）。

用法：
    python scripts/selection_kcenter.py --budget 250 --seed 42 \
        --feature-path features/cifar10_trainpool_clip.npy \
        --output selections/kcenter_m250_s42.json

结果依赖选择种子（第一个中心随机初始化），必须报 3 个种子的均值±标准差。
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

from rdss.selection import append_timing, kcenter_greedy, save_selection  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import sha1_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-csv", default="results/p2/selection_timing.csv")
    parser.add_argument("--l2-normalize", action="store_true", help="k-center 按欧氏距离工作，默认不对特征再归一化")
    args = parser.parse_args()

    feat_path = Path(args.feature_path)
    feat_path = feat_path if feat_path.is_absolute() else PROJECT_ROOT / feat_path
    if not feat_path.exists():
        print(f"[FAIL] 特征文件不存在: {feat_path}")
        print("       请先运行: python scripts/extract_clip_features.py --pool trainpool ...")
        return 1

    X = np.load(feat_path)
    if X.ndim != 2:
        print(f"[FAIL] 特征必须是二维 [N, d]，收到 {X.shape}")
        return 1
    if args.l2_normalize:
        from rdss.kernel import l2_normalize

        X = l2_normalize(X)
    n_pool = X.shape[0]

    seeds = load_yaml("configs/seeds.yaml")
    if args.seed not in seeds["SELECTION_SEEDS"]:
        print(f"[WARN] seed={args.seed} 不在 SELECTION_SEEDS={seeds['SELECTION_SEEDS']} 中，报告须说明")

    t0 = time.perf_counter()
    indices = kcenter_greedy(X, args.budget, args.seed)
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    meta = save_selection(
        out,
        indices,
        method="kcenter",
        budget=args.budget,
        n_pool=n_pool,
        elapsed_sec=elapsed,
        extra={
            "selector_seed": args.seed,
            "init": "random_one_point",
            "feature_file": feat_path.name,
            "feature_sha1": sha1_file(feat_path),
            "l2_normalize": bool(args.l2_normalize),
            "implementation": "core-set greedy, O(m·n·d), float32",
            "label_used": False,
            "note": "课程新增基线（非论文 k-Means）",
        },
    )
    append_timing(
        PROJECT_ROOT / args.timing_csv,
        {
            "created": meta["created"],
            "method": "kcenter",
            "budget": args.budget,
            "n_pool": n_pool,
            "selector_seed": args.seed,
            "elapsed_sec": meta["elapsed_sec"],
            "output": out.name,
        },
    )

    print(f"[OK] k-center  m={args.budget}  seed={args.seed}  n_pool={n_pool}  d={X.shape[1]}")
    print(f"     {out.name}  sha1={meta['file_sha1'][:12]}…  用时 {meta['elapsed_sec']:.2f}s")
    print(f"     索引校验通过：数量={meta['n_selected']}，无重复，范围 [0, {n_pool})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
