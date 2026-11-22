"""P1-4：用 median heuristic 计算核带宽 σ / γ，并回填 configs/kernel.yaml。

用法：
    python scripts/compute_sigma.py --feature-path features/cifar10_trainpool_clip.npy \
        --output configs/kernel.yaml
    # 仅查看不写配置
    python scripts/compute_sigma.py --feature-path features/cifar10_trainpool_clip.npy --dry-run
    # 内存/时间不足时的降级（必须记录 seed）
    python scripts/compute_sigma.py --feature-path ... --sample-size 10000 --sample-seed 0

口径（§A2 + 工程约定）：
    σ = 全池 pairwise 距离的中位数（median heuristic）；γ = 1/σ；k(x,y)=exp(-||x-y||²/σ)。
    默认先 L2 归一化特征（重复归一化幂等），否则特征尺度会让 σ 失去意义。
    统计分块 + 直方图完成，不落地 n×n 核矩阵；精度 ≈ 一个 bin 宽（默认 4/32768 ≈ 1.2e-4）。

产物：
    configs/kernel.yaml                     sigma / gamma / sigma_source 回填（注释保留）
    features/kernel_sigma_stats.json        全部统计 + 特征 sha1 + 复算命令
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rdss.kernel import sigma_from_features  # noqa: E402
from rdss.utils.config import load_yaml  # noqa: E402
from rdss.utils.io import save_json, sha1_file  # noqa: E402


def _update_kernel_yaml(path: Path, sigma: float, gamma: float, source: str) -> None:
    """按行替换，保留注释（用 PyYAML 重写会丢掉全部注释）。"""
    lines = path.read_text(encoding="utf-8").splitlines()

    def _replace(key: str, new_line: str) -> None:
        pat = re.compile(rf"^\s*{re.escape(key)}:[^\n]*$")
        hits = [i for i, line in enumerate(lines) if pat.match(line)]
        if not hits:
            raise KeyError(f"{path.name} 中找不到可回填的键: {key}")
        for i in hits:
            lines[i] = new_line

    _replace("sigma", f"  sigma: {sigma:.6f}")
    _replace("gamma", f"  gamma: {gamma:.6f}")

    src_pat = re.compile(r"^\s*sigma_source:")
    if any(src_pat.match(line) for line in lines):
        lines = [f"  sigma_source: {source}" if src_pat.match(line) else line for line in lines]
    else:
        gi = max(i for i, line in enumerate(lines) if re.match(r"^\s*gamma:", line))
        lines.insert(gi + 1, f"  sigma_source: {source}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--output", default="configs/kernel.yaml")
    parser.add_argument("--stats-out", default="features/kernel_sigma_stats.json")
    parser.add_argument("--block-rows", type=int, default=0, help="0 = 取 configs/kernel.yaml: kernel.block_rows")
    parser.add_argument("--bins", type=int, default=0, help="0 = 取 configs/kernel.yaml: kernel.bins")
    parser.add_argument("--sample-size", type=int, default=0, help=">0 时对训练池随机下采样（降级，须记录 seed）")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不回填配置文件")
    args = parser.parse_args()

    kernel_cfg = load_yaml("configs/kernel.yaml")["kernel"]
    block_rows = args.block_rows or int(kernel_cfg.get("block_rows", 2000))
    bins = args.bins or int(kernel_cfg.get("bins", 32768))

    feat_path = Path(args.feature_path)
    feat_path = feat_path if feat_path.is_absolute() else PROJECT_ROOT / feat_path
    if not feat_path.exists():
        print(f"[FAIL] 特征文件不存在: {feat_path}\n       请先运行: python scripts/extract_clip_features.py ...")
        return 1

    feats = np.load(feat_path)
    if feats.ndim != 2:
        print(f"[FAIL] 特征必须是二维 [N, d]，收到 {feats.shape}")
        return 1
    if feats.shape[0] < 1000 and args.sample_size == 0:
        print(f"[WARN] 只有 {feats.shape[0]} 条特征，σ 估计噪声较大（正式实验应为 45000 条）")

    t0 = datetime.now()
    stats = sigma_from_features(
        feats,
        normalize=not args.no_l2_normalize,
        block_rows=block_rows,
        bins=bins,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    fsha = sha1_file(feat_path)

    print(f"特征: {feat_path.name}  shape={feats.shape}  dtype={feats.dtype}  sha1={fsha[:12]}…")
    print(f"L2 归一化: {bool(stats['l2_normalized'])}  参与样本 n={int(stats['n'])}  距离对数={stats['pairs']:.3e}")
    print(f"median sq-dist = {stats['median_sq_dist']:.6f}   →  σ = {stats['sigma']:.6f}   γ = {stats['gamma']:.6f}")
    print(f"直方图精度 ≈ {stats['bin_width']:.2e}（bin 宽）  用时 {elapsed:.1f}s")

    source = (
        f"median_heuristic n={int(stats['n'])} l2={int(stats['l2_normalized'])} "
        f"bins={int(stats['bins'])} sha1={fsha[:12]} at={datetime.now().isoformat(timespec='seconds')}"
    )

    yaml_path = Path(args.output)
    yaml_path = yaml_path if yaml_path.is_absolute() else PROJECT_ROOT / yaml_path
    if args.dry_run:
        print(f"[DRY-RUN] 未写入 {yaml_path}；将回填 sigma={stats['sigma']:.6f} gamma={stats['gamma']:.6f}")
    else:
        _update_kernel_yaml(yaml_path, stats["sigma"], stats["gamma"], source)
        print(f"[OK] 已回填 {yaml_path}")

    stats_out = Path(args.stats_out)
    stats_out = stats_out if stats_out.is_absolute() else PROJECT_ROOT / stats_out
    save_json(
        stats_out,
        {
            "feature_file": feat_path.name,
            "feature_shape": list(feats.shape),
            "feature_sha1": fsha,
            "kernel": "gaussian",
            "formula": "exp(-||x-y||^2/sigma)",
            **stats,
            "histogram": {"bins": bins, "block_rows": block_rows},
            "elapsed_sec": round(elapsed, 2),
            "command": "python scripts/compute_sigma.py --feature-path "
            f"{args.feature_path} --output {args.output}"
            + (f" --sample-size {args.sample_size} --sample-seed {args.sample_seed}" if args.sample_size else ""),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"     统计写入 {stats_out.name}；复算请用同一特征文件（sha1 必须一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
