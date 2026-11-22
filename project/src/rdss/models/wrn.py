"""WideResNet-28-2，与 FixMatch 官方实现（google-research/fixmatch）结构一致。

论文对 CIFAR 使用 WRN-28-2（depth=28, widen_factor=2, dropRate=0.0）。
P0 只负责通过"一个 batch 过分类头、形状正确"的检查；训练逻辑在 P3。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = stride == 1 and in_planes == out_planes
        self.shortcut = (
            nn.Identity()
            if self.equal_in_out
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = out + self.shortcut(x) if not self.equal_in_out else out + x
        return out


class NetworkBlock(nn.Module):
    def __init__(self, num_layers: int, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                BasicBlock(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    drop_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    """WRN-d-w，CIFAR 版（32×32 输入，10 类输出）。"""

    def __init__(self, depth: int = 28, widen_factor: int = 2, num_classes: int = 10, drop_rate: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth 必须满足 (depth-4) % 6 == 0"
        num_layers = (depth - 4) // 6
        in_planes = 16
        planes = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = nn.Sequential(
            NetworkBlock(num_layers, planes[0], planes[1], stride=1, drop_rate=drop_rate),
            NetworkBlock(num_layers, planes[1], planes[2], stride=2, drop_rate=drop_rate),
            NetworkBlock(num_layers, planes[2], planes[3], stride=2, drop_rate=drop_rate),
        )
        self.bn = nn.BatchNorm2d(planes[3])
        self.fc = nn.Linear(planes[3], num_classes)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.blocks(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def wrn_28_2(num_classes: int = 10) -> WideResNet:
    """论文与课程统一使用的 CIFAR 主干：WRN-28-2。"""
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, drop_rate=0.0)
