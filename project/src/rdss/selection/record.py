"""选择结果落盘、索引校验与耗时表（P2）。

选中索引统一写成 JSON：
    {"method","budget","n_pool","indices":[...],"n_selected", 元数据...}
索引语义 = 训练池内行号（0..n_pool-1），不是原始 CIFAR 索引。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from ..utils.io import save_json, sha1_file

TIMING_HEADER = [
    "created",
    "method",
    "budget",
    "n_pool",
    "selector_seed",
    "alpha",
    "gamma",
    "elapsed_sec",
    "output",
]


def validate_indices(indices: Sequence[int], n_pool: int, m: int) -> np.ndarray:
    """三重校验：数量 = m、无重复、全部落在 [0, n_pool)。"""
    idx = np.asarray(indices)
    if idx.ndim != 1:
        raise ValueError(f"索引必须是一维数组，收到 shape={idx.shape}")
    if idx.size != m:
        raise AssertionError(f"索引数量 {idx.size} != 预算 m={m}")
    if idx.min() < 0 or idx.max() >= n_pool:
        raise AssertionError(f"索引越界: [{idx.min()}, {idx.max()}] 不在 [0, {n_pool})")
    if np.unique(idx).size != idx.size:
        raise AssertionError("索引存在重复")
    return idx.astype(np.int64)


def save_selection(
    path: str | Path,
    indices: np.ndarray,
    *,
    method: str,
    budget: int,
    n_pool: int,
    elapsed_sec: float,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """写 JSON 并返回含 sha1 的完整元数据。"""
    idx = validate_indices(indices, n_pool, budget)
    payload: Dict[str, Any] = {
        "method": method,
        "budget": int(budget),
        "n_pool": int(n_pool),
        "n_selected": int(idx.size),
        "index_semantics": "position_in_trainpool_features (0..n_pool-1)",
        "elapsed_sec": round(float(elapsed_sec), 3),
        "created": datetime.now().isoformat(timespec="seconds"),
        "indices": [int(x) for x in idx.tolist()],
    }
    if extra:
        payload.update(extra)
    save_json(path, payload)
    payload["file"] = str(path)
    payload["file_sha1"] = sha1_file(path)
    return payload


def append_timing(csv_path: str | Path, row: Dict[str, Any]) -> None:
    """把本次选择的耗时追加到汇总表（不存在则建表头）。表由脚本生成，不手抄。"""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TIMING_HEADER})
