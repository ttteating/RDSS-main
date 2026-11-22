"""P2-1：Random 基线（无放回，固定选择种子）。

用法：
    python scripts/selection_random.py --budget 250 --seed 42 --pool-size 45000 \
        --output selections/random_m250_s42.json
    foreach ($s in 42,1234,12345) { python scripts/selection_random.py --budget 250 --seed $s --output selections/random_m250_s$s.json }

不读取任何图像/标签，只按种子从 [0, pool_size) 抽索引；耗时与索引 sha1 一并落盘。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdss.selection import append_timing, random_select, save_selection  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool-size", type=int, default=None, help="默认取 configs/selection.yaml: pool_size")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-csv", default="results/p2/selection_timing.csv")
    args = parser.parse_args()

    sel_cfg = load_yaml("configs/selection.yaml")["selection"]
    n_pool = args.pool_size or int(sel_cfg["pool_size"])
    seeds = load_yaml("configs/seeds.yaml")
    if args.seed not in seeds["SELECTION_SEEDS"]:
        print(f"[WARN] seed={args.seed} 不在 SELECTION_SEEDS={seeds['SELECTION_SEEDS']} 中，报告须说明")

    t0 = time.perf_counter()
    indices = random_select(n_pool, args.budget, args.seed)
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    meta = save_selection(
        out,
        indices,
        method="random",
        budget=args.budget,
        n_pool=n_pool,
        elapsed_sec=elapsed,
        extra={
            "selector_seed": args.seed,
            "replace": False,
            "implementation": "np.random.RandomState(seed).choice(n, size=m, replace=False)",
            "label_used": False,
        },
    )
    append_timing(
        PROJECT_ROOT / args.timing_csv,
        {
            "created": meta["created"],
            "method": "random",
            "budget": args.budget,
            "n_pool": n_pool,
            "selector_seed": args.seed,
            "elapsed_sec": meta["elapsed_sec"],
            "output": out.name,
        },
    )

    print(f"[OK] random  m={args.budget}  seed={args.seed}  n_pool={n_pool}")
    print(f"     {out.name}  sha1={meta['file_sha1'][:12]}…  用时 {meta['elapsed_sec']:.4f}s")
    print(f"     索引校验通过：数量={meta['n_selected']}，无重复，范围 [0, {n_pool})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
