"""P1-1：下载并校验 CIFAR-10（不下载时不落盘，只做存在性检查）。

用法：
    python scripts/download_cifar10.py                 # 下载到 project/data（首次约 170MB）
    python scripts/download_cifar10.py --check-only    # 只校验，不联网、不下载

说明：数据只在 P1 下载；本脚本不读标签做任何选择，仅做规模与类别名校验。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdss.data.split import CIFAR10_TEST_SIZE, CIFAR10_TRAIN_SIZE  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import sha1_file  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="数据根目录（默认取 configs/data.yaml: data.root）")
    parser.add_argument("--check-only", action="store_true", help="只校验本地数据完整性，不下载")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path(load_yaml("configs/data.yaml")["data"]["root"])
    root = root if root.is_absolute() else PROJECT_ROOT / root
    batch_dir = root / "cifar-10-batches-py"

    if args.check_only and not batch_dir.exists():
        print(f"[FAIL] 未找到 {batch_dir}，请先不带 --check-only 运行一次以完成下载")
        return 1

    from torchvision import datasets

    train = datasets.CIFAR10(root=str(root), train=True, download=not args.check_only)
    test = datasets.CIFAR10(root=str(root), train=False, download=not args.check_only)

    assert len(train) == CIFAR10_TRAIN_SIZE, f"训练集规模异常: {len(train)}"
    assert len(test) == CIFAR10_TEST_SIZE, f"测试集规模异常: {len(test)}"
    assert train.classes == CLASSES, f"类别名异常: {train.classes}"

    print(f"[OK] CIFAR-10 就绪：root={root}")
    print(f"     train={len(train)}  test={len(test)}  classes={len(train.classes)}")
    for name in ("cifar-10-python.tar.gz", "cifar-10-batches-py"):
        p = root / name
        if p.is_file():
            print(f"     {name}: sha1={sha1_file(p)}")
    print("     下一步：python scripts/split_val.py --val-seed 0 --output splits/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
