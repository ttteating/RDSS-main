"""P3 单元测试：FixMatch 单步损失可反传、掩码语义、数据拆分无泄漏、种子可复现。

    python tests/test_p3_fixmatch.py
    pytest tests/test_p3_fixmatch.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdss.data.cifar import split_labeled_unlabeled  # noqa: E402
from rdss.train.fixmatch import fixmatch_loss, set_seed  # noqa: E402


def test_loss_backward_finite_grad() -> None:
    set_seed(0)
    b, u, c = 4, 12, 10
    logits_x = torch.randn(b, c, requires_grad=True)
    logits_u_w = torch.randn(u, c)
    logits_u_s = torch.randn(u, c, requires_grad=True)
    y = torch.randint(0, c, (b,))
    loss, loss_x, loss_u, mask = fixmatch_loss(logits_x, y, logits_u_w, logits_u_s, 0.9, 1.0)
    loss.backward()
    assert logits_x.grad is not None and logits_u_s.grad is not None, "两边都必须有梯度"
    assert torch.isfinite(logits_x.grad).all() and torch.isfinite(logits_u_s.grad).all()
    assert 0.0 <= float(mask) <= 1.0
    assert torch.isclose(loss.detach(), loss_x.detach() + 1.0 * loss_u.detach(), atol=1e-6)


def test_mask_zero_when_below_tau() -> None:
    """均匀 logits → 最大概率 0.1 < τ=0.9 → 无监督项必须为 0。"""
    logits_u_w = torch.zeros(6, 10)
    logits_u_s = torch.randn(6, 10, requires_grad=True)
    logits_x = torch.randn(2, 10, requires_grad=True)
    y = torch.tensor([0, 1])
    loss, loss_x, loss_u, mask = fixmatch_loss(logits_x, y, logits_u_w, logits_u_s, 0.9, 1.0)
    assert float(mask) == 0.0 and float(loss_u.detach()) == 0.0
    assert torch.allclose(loss, loss_x)
    loss.backward()
    assert float(logits_u_s.grad.abs().sum()) == 0.0, "mask=0 时强增强分支不应收到梯度"


def test_lambda_u_scales_unsupervised_term() -> None:
    set_seed(1)
    b, u, c = 2, 8, 10
    logits_x = torch.randn(b, c)
    logits_u_w = torch.randn(u, c) * 5.0  # 拉大置信度，保证 mask > 0
    logits_u_s = torch.randn(u, c)
    y = torch.randint(0, c, (b,))
    l1, _, lu, _ = fixmatch_loss(logits_x, y, logits_u_w, logits_u_s, 0.9, 1.0)
    l2, _, _, _ = fixmatch_loss(logits_x, y, logits_u_w, logits_u_s, 0.9, 2.0)
    assert torch.isclose(l2 - l1, lu, atol=1e-5), "λ_u 应线性作用在无监督项上"


def test_split_labeled_unlabeled_no_leak() -> None:
    pool = np.arange(100, dtype=np.int64)
    pos = np.array([0, 3, 5, 9], dtype=np.int64)
    lab, unl = split_labeled_unlabeled(pool, pos)
    assert lab.tolist() == [0, 3, 5, 9]
    assert unl.size == 96
    assert np.intersect1d(lab, unl).size == 0
    assert np.array_equal(np.sort(np.concatenate([lab, unl])), pool)
    for bad in (np.array([0, 0], dtype=np.int64), np.array([100], dtype=np.int64)):
        try:
            split_labeled_unlabeled(pool, bad)
        except (ValueError, AssertionError):
            continue
        raise AssertionError(f"未拦截非法选择结果: {bad}")


def test_seed_reproducible() -> None:
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)
    assert np.isfinite(a.numpy()).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
