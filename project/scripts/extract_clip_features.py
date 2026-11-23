"""P1-3：用冻结的 CLIP ViT-B/32 抽取特征并落盘（三方法共享同一份）。

用法（§H 模板）：
    python scripts/extract_clip_features.py --dataset cifar10 --model openai/clip-vit-base-patch32 \
        --pool trainpool --indices splits/trainpool_indices_seed0.json \
        --output features/cifar10_trainpool_clip.npy
    python scripts/extract_clip_features.py --pool test --output features/cifar10_test_clip.npy
    python scripts/extract_clip_features.py --pool trainpool --limit 128 --output /tmp/smoke.npy   # 烟测

口径（写入报告，三方法一致）：
    - 预处理：Resize(224,224) + ToTensor，不做像素 Normalize（CLIP 内部自带）；
    - 特征：--take cls_last_hidden = final [CLS]（论文口径）；pooler_output = 官方脚本口径（仅差最后一层 LayerNorm）；
    - 工程约定：默认对特征做 L2 归一化（--no-l2-normalize 可关闭），否则 σ 失去意义；
    - 行序：输出第 r 行 = 索引文件第 r 个元素对应的图片（选择器输出的 r 即训练池内位置）。

产物：
    *.npy            float32 [N, 768]
    *.meta.json      {pool, n, dim, model, take, resize, l2_normalize, amp, indices_file, indices_sha1,
                      feature_sha1, elapsed_sec, versions}
本脚本**不读取任何标签**信息（DataLoader 的 label 直接丢弃），避免选样阶段的标签泄漏质疑。
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

from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import load_json, save_json, sha1_file  # noqa: E402

POOL_CHOICES = ["trainpool", "val", "train", "test"]


def _default_indices_file(pool: str, data_cfg: dict) -> Path | None:
    files = data_cfg["data"]["split"]["files"]
    mapping = {
        "trainpool": files["trainpool_indices"],
        "val": files["val_indices"],
        "test": files["test_indices"],
        "train": None,  # 全训练集，无需索引文件
    }
    rel = mapping[pool]
    return None if rel is None else PROJECT_ROOT / rel


def _resolve_indices(pool: str, indices_file: Path | None, data_cfg: dict) -> np.ndarray:
    n_train = data_cfg["data"]["split"]["val_size"] + data_cfg["data"]["split"]["train_pool_size"]
    if pool == "train":
        return np.arange(n_train, dtype=np.int64)
    if indices_file is None or not indices_file.exists():
        raise FileNotFoundError(
            f"索引文件不存在: {indices_file}\n"
            f"请先运行: python scripts/split_val.py --val-seed <VAL_SEED> --output splits/"
        )
    payload = load_json(indices_file)
    arr = np.asarray(payload["indices"], dtype=np.int64)
    if pool == "test" and arr.size != 10000:
        raise ValueError(f"测试集索引数量异常: {arr.size}")
    return arr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--pool", default="trainpool", choices=POOL_CHOICES)
    parser.add_argument("--indices", default=None, help="索引 json；默认按 pool 从 configs/data.yaml 取")
    parser.add_argument("--model", default=None, help="默认取 configs/data.yaml: features.model")
    parser.add_argument("--take", default=None, choices=["cls_last_hidden", "pooler_output"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--root", default=None, help="CIFAR-10 根目录，默认 configs/data.yaml: data.root")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=0, help="只抽前 N 张（烟测用，产物不可用于实验）")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--deterministic", action="store_true", help="开启 cudnn 确定性（复核 sha1 时用）")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets
    except ImportError as e:  # pragma: no cover
        print(f"[FAIL] 缺少依赖: {e}\n       请先在目标机器安装 requirements.txt")
        return 2

    data_cfg = load_yaml("configs/data.yaml")
    feat_cfg = data_cfg["features"]
    model_name = args.model or feat_cfg["model"]
    take = args.take or feat_cfg["take"]
    batch_size = args.batch_size or feat_cfg["batch_size"]
    l2_norm = bool(feat_cfg["l2_normalize"]) and not args.no_l2_normalize
    amp = (bool(feat_cfg.get("amp", True)) if args.amp is None else args.amp)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        amp = False  # CPU autocast 用 bfloat16，会引入额外数值差异

    torch.manual_seed(0)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    indices_file = Path(args.indices) if args.indices else _default_indices_file(args.pool, data_cfg)
    if indices_file is not None and not indices_file.is_absolute():
        indices_file = PROJECT_ROOT / indices_file
    indices = _resolve_indices(args.pool, indices_file, data_cfg)
    if args.limit:
        indices = indices[: args.limit]
        print(f"[WARN] --limit={args.limit} 生效：这是烟测产物，不得用于正式实验")

    output = Path(args.output) if args.output else Path(f"features/cifar10_{args.pool}_clip.npy")
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    root = Path(args.root) if args.root else Path(data_cfg["data"]["root"])
    root = root if root.is_absolute() else PROJECT_ROOT / root
    is_train_split = args.pool != "test"
    if not root.exists():
        print(f"[FAIL] 数据目录不存在: {root}\n       请先运行: python scripts/download_cifar10.py")
        return 1

    from rdss.features.clip import build_transform, extract_features, load_clip_vision

    ds = datasets.CIFAR10(
        root=str(root), train=is_train_split, download=False, transform=build_transform(feat_cfg["resize"])
    )
    subset = Subset(ds, [int(i) for i in indices])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    print(f"pool={args.pool}  n={len(subset)}  model={model_name}  take={take}  l2_norm={l2_norm}  amp={amp}  device={device}")
    model = load_clip_vision(model_name, device=device)
    t0 = time.time()
    feats = extract_features(
        model, loader, device=device, take=take, l2_normalize=l2_norm, amp=amp
    ).cpu().numpy().astype(np.float32, copy=False)
    elapsed = time.time() - t0

    expected = (len(subset), 768)
    if feats.shape != expected:
        print(f"[FAIL] 特征形状 {feats.shape} != 期望 {expected}")
        return 1

    np.save(output, feats)
    meta = {
        "pool": args.pool,
        "n": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "dtype": str(feats.dtype),
        "model": model_name,
        "take": take,
        "resize": feat_cfg["resize"],
        "pixel_normalize": bool(feat_cfg["pixel_normalize"]),
        "l2_normalize": l2_norm,
        "amp": bool(amp),
        "batch_size": batch_size,
        "device": device,
        "indices_file": None if indices_file is None else indices_file.name,
        "indices_sha1": None if indices_file is None else sha1_file(indices_file),
        "feature_file": output.name,
        "feature_sha1": sha1_file(output),
        "elapsed_sec": round(elapsed, 2),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    save_json(output.with_suffix(".meta.json"), meta)

    print(f"[OK] 已保存 {output}  shape={feats.shape}  dtype={feats.dtype}  用时 {elapsed:.1f}s")
    print(f"     sha1={meta['feature_sha1']}")
    print(f"     复核：同参数再跑一次，sha1 必须相同（不一致请加 --no-amp --deterministic）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())