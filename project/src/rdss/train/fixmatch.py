"""统一 FixMatch 训练器（P3）：三种选样方法共用同一份实现与超参。

配方（§A5.1）：
    L = CE(p(y|x_w), y) + λ_u · mean[ 1(max p ≥ τ) · CE(p(y|A(u)), argmax p(y|α_w(u))) ]
    SGD(momentum=0.9, nesterov) + lr 0.03 + wd 5e-4 + cosine；WRN-28-2；AMP。
模型选择：只用固定验证集（monitor=val_acc）选最佳 checkpoint；测试集在训练结束后**只评估一次**。

与论文/USB 的已知差异（须写入报告，不影响三方法之间的公平性）：
    - 不含 EMA 影子模型（USB 的 FixMatch 默认带 EMA）；
    - 不使用 USB 的 1024 steps/epoch，改为"1 epoch = 一次未标注池遍历"（约 100 step），100 epochs ≈ 1e4 step。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

Log = Callable[[str], None]


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fixmatch_loss(
    logits_x: torch.Tensor,
    targets: torch.Tensor,
    logits_u_w: torch.Tensor,
    logits_u_s: torch.Tensor,
    tau: float,
    lambda_u: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FixMatch 单步损失：L = CE(L) + λ_u · mean[mask · CE(U_strong, pseudo)]。

    返回 (总损失, 有监督损失, 无监督损失, 伪标签掩码比例)；可直接 backward。
    """
    loss_x = F.cross_entropy(logits_x, targets)
    with torch.no_grad():
        probs = torch.softmax(logits_u_w.detach().float(), dim=1)
        max_probs, pseudo = probs.max(dim=1)
        mask = (max_probs >= tau).float()
    loss_u = (F.cross_entropy(logits_u_s, pseudo, reduction="none") * mask).mean()
    return loss_x + lambda_u * loss_u, loss_x, loss_u, mask.mean()


def _cycle(loader: Iterable) -> Iterable:
    while True:
        for batch in loader:
            yield batch


class FixMatchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        labeled_loader,
        unlabeled_loader,
        val_loader,
        test_loader,
        cfg: Dict,
        device: torch.device,
        output_dir: Path,
        log: Log,
    ):
        self.model = model
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.fm = cfg["fixmatch"]
        self.device = device
        self.output_dir = Path(output_dir)
        self.log = log

        self.amp = bool(self.fm.get("amp", True)) and device.type == "cuda"
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(self.fm["lr"]),
            momentum=float(self.fm["momentum"]),
            weight_decay=float(self.fm["weight_decay"]),
            nesterov=bool(self.fm.get("nesterov", True)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.scheduler = None  # 由 fit() 按 total_steps 建
        self.global_step = 0
        self.best_val = 0.0
        self.best_step = 0
        self.history: list[dict] = []
        self.mask_ratio_acc = 0.0
        self.mask_ratio_n = 0

    # ---------- 基础工具 ----------
    def _autocast(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp)

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        preds, targets = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x)
            preds.append(logits.float().argmax(dim=1).cpu())
            targets.append(y)
        pred = torch.cat(preds).numpy()
        tgt = torch.cat(targets).numpy()
        self.model.train()
        return {
            "acc": float((pred == tgt).mean()) * 100.0,
            "macro_f1": float(f1_score(tgt, pred, average="macro", zero_division=0)) * 100.0,
        }

    def _save(self, name: str, epoch: int, extra: Optional[Dict] = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "step": self.global_step,
            "best_val": self.best_val,
            "best_step": self.best_step,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, self.output_dir / name)

    def _load_resume(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = int(ckpt["step"])
        self.best_val = float(ckpt.get("best_val", 0.0))
        self.best_step = int(ckpt.get("best_step", 0))
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        self.log(f"[RESUME] {path.name}: epoch={ckpt['epoch']} step={self.global_step} best_val={self.best_val:.2f}")
        return int(ckpt["epoch"])

    # ---------- 训练 ----------
    def fit(
        self,
        epochs: int,
        steps_per_epoch: int,
        supervised: bool = False,
        max_steps: int = 0,
        resume: Optional[Path] = None,
    ) -> Dict[str, float]:
        total_steps = epochs * steps_per_epoch
        if max_steps:
            total_steps = min(total_steps, max_steps)
            steps_per_epoch = max(1, min(steps_per_epoch, total_steps))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1))
        start_epoch = 0
        if resume is not None:
            start_epoch = self._load_resume(resume)

        tau = float(self.fm["tau"])
        lambda_u = float(self.fm["lambda_u"])
        eval_step = int(self.fm["eval_step"])
        t_start = time.time()
        self.model.train()

        for epoch in range(start_epoch, epochs):
            if self.global_step >= total_steps:
                break
            lab_iter = _cycle(self.labeled_loader)
            unl_iter = _cycle(self.unlabeled_loader)
            for _ in range(steps_per_epoch):
                if self.global_step >= total_steps:
                    break
                timer = time.time()
                if supervised:
                    x, y = next(lab_iter)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    with self._autocast():
                        loss = F.cross_entropy(self.model(x), y)
                    loss_x, loss_u, mask_ratio = float(loss.detach()), 0.0, 0.0
                else:
                    x, y = next(lab_iter)
                    uw, us = next(unl_iter)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    uw = uw.to(self.device, non_blocking=True)
                    us = us.to(self.device, non_blocking=True)
                    b = x.size(0)
                    with self._autocast():
                        logits = self.model(torch.cat([x, uw], dim=0))
                        logits_x, logits_u_w = logits[:b], logits[b:]
                        logits_u_s = self.model(us)
                        loss, loss_x_t, loss_u_t, mask_mean = fixmatch_loss(
                            logits_x, y, logits_u_w, logits_u_s, tau, lambda_u
                        )
                    mask_ratio = float(mask_mean)
                    self.mask_ratio_acc += mask_ratio
                    self.mask_ratio_n += 1
                    loss_x, loss_u = float(loss_x_t.detach()), float(loss_u_t.detach())

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % 50 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.log(
                        f"step {self.global_step:6d}/{total_steps}  loss={float(loss):.4f} "
                        f"loss_x={loss_x:.4f} loss_u={loss_u:.4f} mask={mask_ratio:.3f} "
                        f"lr={lr:.5f}  {(time.time()-timer)*1000:.0f}ms/step"
                    )
                if self.global_step % eval_step == 0:
                    val = self.evaluate(self.val_loader)
                    self.history.append({"step": self.global_step, "epoch": epoch, **val})
                    if val["acc"] > self.best_val:
                        self.best_val, self.best_step = val["acc"], self.global_step
                        self._save("best_model.pth", epoch)
                    self.log(f"  [eval] step={self.global_step} val_acc={val['acc']:.2f} val_f1={val['macro_f1']:.2f} best={self.best_val:.2f}")
                if self.global_step >= total_steps:
                    break
            # 每 epoch 末存一次断点（含模型/优化器/调度器/随机状态）
            self._save("last.pth", epoch)
            elapsed = time.time() - t_start
            self.log(f"epoch {epoch+1}/{epochs}  done  steps={self.global_step}  elapsed={elapsed/60:.1f}min")

        # 若从未触发 eval（steps < eval_step），补一次
        if not self.history:
            val = self.evaluate(self.val_loader)
            self.history.append({"step": self.global_step, "epoch": epochs - 1, **val})
            self.best_val, self.best_step = val["acc"], self.global_step
            self._save("best_model.pth", epochs - 1)

        # 用最佳 checkpoint 做最终评估；测试集只在此处使用一次
        best = torch.load(self.output_dir / "best_model.pth", map_location=self.device, weights_only=False)
        self.model.load_state_dict(best["model"])
        final_val = self.evaluate(self.val_loader)
        final_test = self.evaluate(self.test_loader)

        elapsed = time.time() - t_start
        peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3 if self.device.type == "cuda" else 0.0
        mask_ratio_mean = self.mask_ratio_acc / self.mask_ratio_n if self.mask_ratio_n else 0.0
        self.log(
            f"[FINAL] best_step={self.best_step}  val_acc={final_val['acc']:.2f}  "
            f"test_acc={final_test['acc']:.2f}  test_f1={final_test['macro_f1']:.2f}  "
            f"elapsed={elapsed/60:.1f}min  peak_mem={peak_gb:.2f}GB"
        )
        return {
            "val": final_val,
            "test": final_test,
            "best_val": self.best_val,
            "best_step": self.best_step,
            "elapsed_sec": round(elapsed, 1),
            "peak_mem_gb": round(peak_gb, 3),
            "steps": self.global_step,
            "mean_mask_ratio": round(mask_ratio_mean, 4),
            "history": self.history,
        }
