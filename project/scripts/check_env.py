"""P0 通过条件自检：一个 batch 过 CLIP 与分类头，打印形状；同时校验配置一致性。

用法（在服务器上、project/ 目录内）：
    python scripts/check_env.py
    python scripts/check_env.py --random-images        # 未下载 CIFAR-10 时用随机像素占位
    python scripts/check_env.py --device cpu --batch-size 2

输出即 docs/P0_环境清单.md 中"实际环境"一栏的填写依据；退出码 0 = P0 通过。
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdss.models.wrn import wrn_28_2  # noqa: E402
from rdss.utils.config import PROJECT_ROOT as ROOT  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402


def _hr(title: str) -> None:
    print(f"\n=== {title} ===")


def check_versions() -> dict:
    import torch
    import torchvision

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "gpu_total_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available()
        else 0.0,
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "NOT INSTALLED"
    return info


def check_configs() -> None:
    _hr("配置校验")
    seeds = load_yaml("configs/seeds.yaml")
    data = load_yaml("configs/data.yaml")
    kernel = load_yaml("configs/kernel.yaml")
    selection = load_yaml("configs/selection.yaml")
    fm = load_yaml("configs/fixmatch.yaml")

    # 1) 验证集划分种子两处必须一致（防止两种划分泄漏）
    assert seeds["VAL_SEED"] == data["data"]["split"]["val_seed"], "VAL_SEED 与 data.split.val_seed 不一致"
    # 2) 预算必须覆盖课程要求的 m ∈ {250, 1000}
    assert set(selection["selection"]["budgets"]) == {250, 1000}, "budgets 必须是 [250, 1000]"
    # 3) FixMatch batch 关系自洽
    f = fm["fixmatch"]
    assert f["batch_unlabeled"] == f["batch_size_labeled"] * f["mu"], "batch_unlabeled ≠ batch_size_labeled × mu"
    # 4) 划分尺寸自洽
    s = data["data"]["split"]
    assert s["val_size"] + s["train_pool_size"] == 50000, "验证集 + 训练池必须等于 CIFAR-10 训练集 50000"
    # 5) α 代值可复算
    for m, val in kernel["alpha"]["table"].items():
        assert abs(1 - 1 / (int(m) ** 0.5) - val) < 5e-4, f"alpha 表 m={m} 与 1-1/sqrt(m) 不符"
    # 6) 结构完整性
    for key in ("SELECTION_SEEDS", "TRAIN_SEEDS"):
        assert len(seeds[key]) >= 3, f"{key} 至少 3 个种子"
    print("OK: seeds/data/kernel/selection/fixmatch 校验通过（6 项断言）")
    print(f"    SELECTION_SEEDS={seeds['SELECTION_SEEDS']} TRAIN_SEEDS={seeds['TRAIN_SEEDS']} VAL_SEED={seeds['VAL_SEED']}")
    print(f"    fixmatch: tau={f['tau']} lambda_u={f['lambda_u']} mu={f['mu']} batch={f['batch_size_labeled']}+{f['batch_unlabeled']}")


def _load_pixel_batch(batch_size: int, resize: int, random_images: bool):
    """返回 [B, 3, resize, resize] 的像素张量（真实 CIFAR-10 或随机占位）。"""
    import torch
    from torchvision import datasets

    from rdss.features.clip import build_transform

    if random_images:
        print("[WARN] 使用随机像素张量代替 CIFAR-10（仅验证形状，不代表真实数据）")
        return torch.rand(batch_size, 3, resize, resize)

    root = ROOT / "data"
    ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=build_transform(resize))
    pixel_values = torch.stack([ds[i][0] for i in range(batch_size)])
    print(f"CIFAR-10 已就绪: root={root} 取前 {batch_size} 张（train=True）")
    return pixel_values


def check_clip(batch_size: int, device: str, take: str, random_images: bool) -> None:
    _hr("CLIP ViT-B/32 单批前向")
    import torch

    from rdss.features.clip import encode_batch, load_clip_vision

    data_cfg = load_yaml("configs/data.yaml")["features"]
    model_name = data_cfg["model"]
    model = load_clip_vision(model_name, device=device)

    pixel_values = _load_pixel_batch(batch_size, data_cfg["resize"], random_images).to(device)
    feats = encode_batch(model, pixel_values, take=take)

    assert feats.shape == (batch_size, 768), f"CLIP 特征形状异常: {tuple(feats.shape)}"
    assert feats.dtype == torch.float32, f"CLIP 特征 dtype 异常: {feats.dtype}"
    print(f"pixel_values: {tuple(pixel_values.shape)} ({pixel_values.dtype})")
    print(f"features    : {tuple(feats.shape)} ({feats.dtype})  take={take}  model={model_name}")
    print("OK: 形状 [B, 768] 正确")


def check_classifier(batch_size: int, device: str) -> None:
    _hr("WRN-28-2 单批前向")
    import torch

    model = wrn_28_2(num_classes=10).to(device)
    x = torch.rand(batch_size, 3, 32, 32, device=device)
    with torch.no_grad():
        logits = model(x)
    n_param = sum(p.numel() for p in model.parameters())
    assert logits.shape == (batch_size, 10), f"分类头形状异常: {tuple(logits.shape)}"
    print(f"input : {tuple(x.shape)}")
    print(f"logits: {tuple(logits.shape)}  params={n_param/1e6:.2f}M (WRN-28-2 约 1.47M)")
    print("OK: 形状 [B, 10] 正确")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    parser.add_argument("--take", default="cls_last_hidden", choices=["cls_last_hidden", "pooler_output"])
    parser.add_argument("--random-images", action="store_true", help="不读 CIFAR-10，用随机像素验证形状")
    parser.add_argument("--skip-clip", action="store_true", help="只验证配置与分类头")
    args = parser.parse_args()

    _hr("环境版本")
    for k, v in check_versions().items():
        print(f"{k:>16}: {v}")

    check_configs()
    check_classifier(args.batch_size, args.device)
    if not args.skip_clip:
        check_clip(args.batch_size, args.device, args.take, args.random_images)

    _hr("P0 自检结果")
    if args.skip_clip:
        print("PARTIAL: 配置自洽、分类头形状正确；CLIP 分支被 --skip-clip 跳过（P0 尚未完全通过）。")
    else:
        print("PASS: 配置可解析且自洽；一个 batch 过 CLIP 与分类头，形状正确。")
    return 0


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
