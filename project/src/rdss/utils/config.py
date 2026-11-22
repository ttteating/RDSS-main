"""配置读取：所有脚本统一走这里，禁止在脚本里散落硬编码路径与超参。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# src/rdss/utils/config.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(*paths: str | Path) -> Dict[str, Any]:
    """读取并浅合并多个配置文件（后者覆盖同名顶层键）。"""
    merged: Dict[str, Any] = {}
    for path in paths:
        merged.update(load_yaml(path))
    return merged


def resolve_path(path: str | Path) -> Path:
    """把配置里的相对路径按 project/ 根目录解析成绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
