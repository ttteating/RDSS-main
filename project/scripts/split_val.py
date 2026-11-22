"""P1-2：生成固定的验证集 / 训练池 / 测试集索引，三方法共享（§A5.2）。

用法：
    python scripts/split_val.py --val-seed 0 --output splits/
    python scripts/split_val.py --val-seed 0 --output splits/ --force   # 覆盖已存在文件（慎用）

产物：
    splits/val_indices_seed0.json          {"seed":0,"size":5000,"indices":[...]}
    splits/trainpool_indices_seed0.json    {"seed":0,"size":45000,"indices":[...]}
    splits/test_indices.json               {"size":10000,"indices":[0..9999]}
    splits/splits_meta.json                {seed, sizes, 每个文件的 sha1, 生成时间}

索引语义：均为原始 split 内的位置（0..49999）。特征文件第 r 行 = indices[r] 对应的图片，
因此**选择器输出的是训练池内的位置 r，而不是原始 CIFAR 索引**，映射关系由本文件固定。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rdss.data.split import CIFAR10_TRAIN_SIZE, DEFAULT_VAL_SIZE, make_splits, test_indices  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import save_json, sha1_file  # noqa: E402


def _as_list(a: np.ndarray) -> list:
    return [int(x) for x in a.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-seed", type=int, default=None, help="默认取 configs/seeds.yaml: VAL_SEED")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--n-total", type=int, default=CIFAR10_TRAIN_SIZE)
    parser.add_argument("--output", default="splits")
    parser.add_argument("--force", action="store_true", help="已存在时覆盖（会破坏其它实验的可追溯性）")
    args = parser.parse_args()

    seeds = load_yaml("configs/seeds.yaml")
    val_seed = seeds["VAL_SEED"] if args.val_seed is None else args.val_seed
    if val_seed != seeds["VAL_SEED"]:
        print(f"[WARN] --val-seed={val_seed} 与 configs/seeds.yaml: VAL_SEED={seeds['VAL_SEED']} 不一致，报告中必须说明原因")

    out_dir = Path(args.output)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    val_path = out_dir / f"val_indices_seed{val_seed}.json"
    pool_path = out_dir / f"trainpool_indices_seed{val_seed}.json"
    test_path = out_dir / "test_indices.json"
    meta_path = out_dir / "splits_meta.json"

    existing = [p for p in (val_path, pool_path, test_path) if p.exists()]
    if existing and not args.force:
        print("[SKIP] 以下索引文件已存在，未覆盖（保证可追溯）：")
        for p in existing:
            print(f"       {p.name}")
        print("       如需重建请加 --force（会改变后续所有实验的可追溯性）")
        return 0

    splits = make_splits(n_total=args.n_total, val_size=args.val_size, seed=val_seed)
    test = test_indices()

    # 再次独立复核：数量、无重复、两两不交、并集为全集
    val, pool = splits["val"], splits["train_pool"]
    for name, arr, expect in (("val", val, args.val_size), ("train_pool", pool, args.n_total - args.val_size), ("test", test, 10000)):
        assert np.unique(arr).size == arr.size == expect, f"{name} 数量/去重异常: {np.unique(arr).size}/{arr.size} != {expect}"
    assert np.intersect1d(val, pool).size == 0
    assert np.array_equal(np.sort(np.concatenate([val, pool])), np.arange(args.n_total))

    save_json(val_path, {"seed": val_seed, "size": int(val.size), "indices": _as_list(val)})
    save_json(pool_path, {"seed": val_seed, "size": int(pool.size), "indices": _as_list(pool)})
    save_json(test_path, {"seed": None, "size": int(test.size), "indices": _as_list(test)})

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "val_seed": val_seed,
        "n_total": args.n_total,
        "val_size": int(val.size),
        "train_pool_size": int(pool.size),
        "test_size": int(test.size),
        "files": {
            val_path.name: {"rows": int(val.size), "sha1": sha1_file(val_path)},
            pool_path.name: {"rows": int(pool.size), "sha1": sha1_file(pool_path)},
            test_path.name: {"rows": int(test.size), "sha1": sha1_file(test_path)},
        },
    }
    save_json(meta_path, meta)

    print(f"[OK] 划分完成（val_seed={val_seed}）：val={val.size}  train_pool={pool.size}  test={test.size}")
    print(f"     {val_path}")
    print(f"     {pool_path}")
    print(f"     {test_path}")
    print(f"     meta sha1 已写入 {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
