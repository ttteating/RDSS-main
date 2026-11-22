"""P3：统一 FixMatch 训练入口（三种选样方法共用）。

用法（§H 模板）：
    python scripts/train_fixmatch.py --config configs/fixmatch.yaml \
        --labeled-list selections/random_m250_s42.json \
        --val-indices splits/val_indices_seed0.json --test-indices splits/test_indices.json \
        --seed 42 --output-dir results/p3_sanity_random_m250_seed42

常用变体：
    --supervised            纯监督 sanity check（只用 m 个有标签样本，验证 Acc ≫ 10% 且无泄漏）
    --max-steps 20          烟测（不写正式结果）
    --fake-data             无数据集/无特征时验证训练循环与断点（随机像素，结论不可用）
    --resume <last.pth>     断点续训（模型/优化器/调度器/AMP/随机状态全恢复）
    --epochs 50             课程缩减（单 epoch > 5 min 时使用，须在报告注明）

产物（每个 output-dir）：
    train.log  metrics.json  config.yaml  best_model.pth  last.pth
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from rdss.data.cifar import (  # noqa: E402
    EvalDataset,
    FakeCIFAR,
    LabeledDataset,
    UnlabeledDataset,
    eval_transform,
    split_labeled_unlabeled,
    strong_transform,
    weak_transform,
)
from rdss.train import FixMatchTrainer, set_seed  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import load_json, save_json, sha1_file  # noqa: E402


class Tee:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _indices(path: Path, key: str = "indices") -> np.ndarray:
    return np.asarray(load_json(path)[key], dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fixmatch.yaml")
    p.add_argument("--labeled-list", default=None, help="选择器输出的 JSON（池内行号）")
    p.add_argument("--trainpool-indices", default=None)
    p.add_argument("--val-indices", default=None)
    p.add_argument("--test-indices", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--supervised", action="store_true", help="纯监督 sanity check（不用无标注池）")
    p.add_argument("--epochs", type=int, default=0, help="0 = 取配置")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0, help="覆盖有标签 batch（无标签 = 该值 × mu）")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--fake-data", type=int, default=0, help=">0 时用 N 张随机像素烟测（不读数据集/选择结果）")
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from rdss.models.wrn import wrn_28_2

    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(out_dir / "train.log")

    fm_cfg = load_yaml(args.config)
    data_cfg = load_yaml("configs/data.yaml")
    fm = fm_cfg["fixmatch"]
    bs = args.batch_size or int(fm["batch_size_labeled"])
    mu = int(fm["mu"])
    epochs = args.epochs or int(fm["epochs"])
    num_workers = args.num_workers if args.num_workers is not None else int(fm["num_workers"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    set_seed(args.seed, deterministic=args.deterministic)
    log(f"device={device}  seed={args.seed}  supervised={args.supervised}  amp={fm.get('amp')}  epochs={epochs}")

    # ---------- 数据 ----------
    weak, strong, evalt = weak_transform(), strong_transform(), eval_transform()
    run_meta = {"seed": args.seed, "supervised": bool(args.supervised), "batch_labeled": bs, "mu": mu}

    if args.fake_data:
        n = args.fake_data
        base_train = FakeCIFAR(n=n + 200, seed=0)  # 前 n 张做训练池，后 200 张做验证集
        base_test = FakeCIFAR(n=200, seed=1)
        m = min(250, n // 4)
        pool = np.arange(n, dtype=np.int64)
        labeled_pos = np.arange(m, dtype=np.int64)
        val_idx = np.arange(n, n + 200, dtype=np.int64)
        test_idx = np.arange(200, dtype=np.int64)
        log(f"[WARN] --fake-data={n}：随机像素烟测，metrics 不得写入正式结果")
        run_meta["fake_data"] = n
    else:
        if not args.labeled_list:
            log("[FAIL] 需要 --labeled-list（或用 --fake-data 烟测）")
            return 1
        pool_path = Path(args.trainpool_indices or data_cfg["data"]["split"]["files"]["trainpool_indices"])
        val_path = Path(args.val_indices or data_cfg["data"]["split"]["files"]["val_indices"])
        test_path = Path(args.test_indices or data_cfg["data"]["split"]["files"]["test_indices"])
        for path in (pool_path, val_path, test_path):
            if not (path if path.is_absolute() else PROJECT_ROOT / path).exists():
                log(f"[FAIL] 索引文件不存在: {path}（先跑 scripts/split_val.py）")
                return 1
        pool = _indices(pool_path if pool_path.is_absolute() else PROJECT_ROOT / pool_path)
        val_idx = _indices(val_path if val_path.is_absolute() else PROJECT_ROOT / val_path)
        test_idx = _indices(test_path if test_path.is_absolute() else PROJECT_ROOT / test_path)

        sel_path = Path(args.labeled_list)
        sel_path = sel_path if sel_path.is_absolute() else PROJECT_ROOT / sel_path
        if not sel_path.exists():
            log(f"[FAIL] 选择结果不存在: {sel_path}")
            return 1
        sel = load_json(sel_path)
        labeled_pos = np.asarray(sel["indices"], dtype=np.int64)
        if int(sel.get("n_pool", pool.size)) != pool.size:
            log(f"[FAIL] 选择结果 n_pool={sel.get('n_pool')} 与训练池 {pool.size} 不一致")
            return 1

        data_root = PROJECT_ROOT / data_cfg["data"]["root"]
        if not data_root.exists():
            log(f"[FAIL] 数据目录不存在: {data_root}（先跑 scripts/download_cifar10.py）")
            return 1
        base_train = datasets.CIFAR10(root=str(data_root), train=True, download=False)
        base_test = datasets.CIFAR10(root=str(data_root), train=False, download=False)
        run_meta.update(
            {
                "labeled_list": sel_path.name,
                "labeled_list_sha1": sha1_file(sel_path),
                "selection_method": sel.get("method"),
                "selection_alpha": sel.get("alpha"),
                "selection_gamma": sel.get("gamma"),
                "trainpool_sha1": sha1_file(pool_path if pool_path.is_absolute() else PROJECT_ROOT / pool_path),
            }
        )

    labeled_orig, unlabeled_orig = split_labeled_unlabeled(pool, labeled_pos)
    # 防泄漏：验证集不得出现在训练池或测试集中
    assert not np.intersect1d(val_idx, pool).size, "验证集与训练池重叠"
    assert not np.intersect1d(val_idx, test_idx).size, "验证集与测试集重叠"

    labeled_ds = LabeledDataset(base_train, labeled_orig, weak)
    unlabeled_ds = UnlabeledDataset(base_train, unlabeled_orig, weak, strong)
    val_ds = EvalDataset(base_train, val_idx, evalt)
    test_ds = EvalDataset(base_test, test_idx, evalt)

    labeled_loader = DataLoader(labeled_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=bs * mu, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    log(f"有标签 {len(labeled_ds)}  无标注 {len(unlabeled_ds)}  验证 {len(val_ds)}  测试 {len(test_ds)}")
    log(f"batch: labeled={bs} unlabeled={bs*mu}  steps/epoch={'labeled' if args.supervised else 'unlabeled'}={len(labeled_loader) if args.supervised else len(unlabeled_loader)}")

    # ---------- 模型与训练 ----------
    model = wrn_28_2(num_classes=int(data_cfg["data"]["num_classes"])).to(device)
    n_param = sum(x.numel() for x in model.parameters())
    log(f"WRN-28-2 params={n_param/1e6:.2f}M")

    trainer = FixMatchTrainer(
        model, labeled_loader, unlabeled_loader, val_loader, test_loader,
        fm_cfg, device, out_dir, log,
    )
    t0 = time.time()
    result = trainer.fit(
        epochs=epochs,
        steps_per_epoch=len(labeled_loader) if args.supervised else len(unlabeled_loader),
        supervised=args.supervised,
        max_steps=args.max_steps,
        resume=Path(args.resume) if args.resume else None,
    )

    metrics = {
        "run": {
            **run_meta,
            "config": args.config,
            "config_snapshot": out_dir.joinpath("config.yaml").name,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "epochs": epochs,
            "max_steps": args.max_steps,
            "wall_sec": round(time.time() - t0, 1),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        },
        **result,
    }
    save_json(out_dir / "metrics.json", metrics)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(fm_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    log(f"[OK] 产物: {out_dir}/{{train.log, metrics.json, config.yaml, best_model.pth, last.pth}}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
