"""P2-4：与官方 RDSS.py 的一致性测试（§A4 保真要求）。

用法（服务器，真实特征）：
    python scripts/check_rdss_consistency.py --feature-path features/cifar10_trainpool_clip.npy \
        --budget 250 --n-subset 5000
本地无特征时可用合成数据先验逻辑自洽：
    python scripts/check_rdss_consistency.py --synthetic 2000 --dim 768 --budget 250

为什么用子集：官方 khsample 先算 `kernel(data)` 的全 n×n 矩阵（float64），
n=45000 时约 16GB，普通机器必 OOM；故一致性测试在 n=5000 子集上做（同特征、同 m、同 α、同 γ）。

理论预期（写进报告）：官方把 μ 去掉对角线，等价于 μ_ex = (n·μ_in − 1)/(n − 1)，
即 α 被缩放为 α·n/(n−1) 并叠加常数偏移，因此 include / exclude 两种口径的选择几乎总相同，
只在近似并列时可能分叉。本脚本同时跑两种口径，输出哪一版与官方逐位相同。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rdss.selection import alpha_auto, gkhr  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import save_json, sha1_file  # noqa: E402


def _load_official(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到官方实现: {path}")
    spec = importlib.util.spec_from_file_location("official_rdss", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    same_order = bool(a.shape == b.shape and np.array_equal(a, b))
    set_a, set_b = set(a.tolist()), set(b.tolist())
    inter = len(set_a & set_b)
    first_diff = None
    if not same_order:
        for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
            if x != y:
                first_diff = {"position": i, "ours": x, "official": y}
                break
    return {
        "same_order": same_order,
        "same_set": set_a == set_b,
        "overlap": inter,
        "overlap_ratio": inter / max(len(set_a), 1),
        "first_diff": first_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--synthetic", type=int, default=0, help=">0 时用随机特征做逻辑自检（不入报告主结果）")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--n-subset", type=int, default=5000)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--alpha", default="auto")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--official", default="../RDSS.py")
    parser.add_argument("--output", default="results/p2/consistency_rdss.json")
    args = parser.parse_args()

    if args.synthetic:
        raw = np.random.RandomState(args.synthetic_seed).randn(args.synthetic, args.dim).astype(np.float32)
        from rdss.kernel import l2_normalize

        X_full = l2_normalize(raw)  # 与真实流水线一致：抽完特征先 L2 归一化
        feat_desc = {"synthetic": True, "n": args.synthetic, "d": args.dim, "seed": args.synthetic_seed}
        print(f"[WARN] 合成特征自检模式（{args.synthetic}×{args.dim}，已 L2 归一化），结论不能写进主报告")
    else:
        if not args.feature_path:
            print("[FAIL] 需要 --feature-path 或 --synthetic")
            return 1
        fp = Path(args.feature_path)
        fp = fp if fp.is_absolute() else PROJECT_ROOT / fp
        if not fp.exists():
            print(f"[FAIL] 特征文件不存在: {fp}")
            return 1
        X_full = np.load(fp)
        feat_desc = {"synthetic": False, "file": fp.name, "sha1": sha1_file(fp), "n": int(X_full.shape[0])}

    # 固定子集：同 seed 下三方法看到同一批候选
    n_full = X_full.shape[0]
    if args.n_subset and args.n_subset < n_full:
        sub = np.sort(np.random.RandomState(args.subset_seed).choice(n_full, size=args.n_subset, replace=False))
        X = np.ascontiguousarray(X_full[sub])
        print(f"子集: n={len(sub)}（原 {n_full}，seed={args.subset_seed}）")
    else:
        sub = np.arange(n_full)
        X = np.ascontiguousarray(X_full)

    if args.budget > X.shape[0]:
        print(f"[FAIL] 预算 {args.budget} > 子集大小 {X.shape[0]}")
        return 1

    kcfg = load_yaml("configs/kernel.yaml")["kernel"]
    gamma = args.gamma if args.gamma is not None else kcfg.get("gamma")
    if gamma is None:
        print("[FAIL] kernel.yaml 无 gamma，请 --gamma 指定（复现官方可用 0.5，但须标注为差异实验）")
        return 1
    alpha = alpha_auto(args.budget) if args.alpha == "auto" else float(args.alpha)

    official = _load_official(PROJECT_ROOT / args.official)
    kernel = lambda a, b=None: official.gaussiankernel(a, b, gamma)  # noqa: E731

    print(f"m={args.budget}  α={alpha:.6f}  γ={gamma:.6f}  μ对角线：官方=exclude / 本实现两版都测")
    t0 = time.perf_counter()
    off_sel, _ = official.khsample(X, kernel, n=args.budget, alpha=alpha)
    off_elapsed = time.perf_counter() - t0
    off = np.asarray(off_sel, dtype=np.int64)

    results = {}
    for diagonal in ("include", "exclude"):
        t0 = time.perf_counter()
        ours = gkhr(X, args.budget, alpha, gamma, diagonal=diagonal)
        elapsed = time.perf_counter() - t0
        cmp = _compare(ours, off)
        results[diagonal] = {"elapsed_sec": round(elapsed, 3), **cmp}
        flag = "完全一致" if cmp["same_order"] else ("集合相同/顺序不同" if cmp["same_set"] else "不完全一致")
        print(f"  本实现 μ.{diagonal:<7} vs 官方: {flag}  重合 {cmp['overlap']}/{args.budget}  用时 {elapsed:.2f}s")

    speedup = max(off_elapsed / max(results[d]["elapsed_sec"], 1e-9) for d in results)
    print(f"官方用时 {off_elapsed:.2f}s；本实现最快 {speedup:.1f}× 加速")

    matched = [d for d in results if results[d]["same_set"]]
    payload = {
        "subset": {"n": int(X.shape[0]), "seed": args.subset_seed if args.n_subset else None,
                   "subset_indices_sha1_note": "子集由 RandomState(subset_seed) 固定"},
        "budget": args.budget,
        "alpha": alpha,
        "gamma": gamma,
        "official": {"file": str(PROJECT_ROOT / args.official), "elapsed_sec": round(off_elapsed, 3),
                     "mu_diagonal": "exclude", "gamma_default": 0.5},
        "ours": results,
        "feature": feat_desc,
        "matched_variants": matched,
        "verdict": ("一致（集合相同）" if matched else "不一致：先查 γ/α，再查索引顺序与并列打破规则"),
    }
    save_json(PROJECT_ROOT / args.output, payload)
    print(f"[{'OK' if matched else 'FAIL'}] 与官方集合{'相同' if matched else '不同'}；详情写入 {args.output}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
