# RDSS 课程规模复现（CIFAR-10 · 低标注半监督学习的数据选择）

本目录是课程期末项目 PROJECT 03 的实现工程，复现 NeurIPS 2024 论文 *Enhancing Semi-Supervised Learning via Representative and Diverse Sample Selection (RDSS)* 的核心相对趋势。
范围：**CIFAR-10，m ∈ {250, 1000}**，同一冻结 CLIP 特征 + 同一 FixMatch 配方，比较 **random / k-center / RDSS(GKHR)** 三种选样。
论文原仓库脚本保留在工作区根目录（`../RDSS.py` 等），仅作参考与一致性测试对象，**不修改**。

## 当前进度

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 立项与审计 | ✅ 代码/配置/文档已完成；CLIP 实测与 `requirements.lock` 待服务器执行 | `docs/P0_*.md`、`configs/*.yaml`、`src/`、`scripts/check_env.py` |
| P1 数据与特征 | 🟡 脚本与划分索引已完成；下载/抽特征/σ 待服务器执行 | `scripts/{download_cifar10,split_val,extract_clip_features,compute_sigma}.py`、`splits/*.json` |
| P2 三种选择器 | 🟡 三个选择器 + 官方一致性测试 + toy 图已完成；真实特征上的索引待服务器生成 | `scripts/selection_{random,kcenter,rdss}.py`、`scripts/check_rdss_consistency.py`、`figures/toy_selection.png` |
| P3 统一 FixMatch | 🟡 训练器、数据管线、断点续训已完成并在 CPU 烟测（含纯监督）；真实训练待服务器 | `src/rdss/train/fixmatch.py`、`src/rdss/data/cifar.py`、`scripts/train_fixmatch.py`、`tests/test_p3_fixmatch.py` |
| P4–P6 | ⬜ 待做 | 主表、图表、报告 |

## 目录结构

```text
project/
├── README.md
├── requirements.txt          # 服务器安装用（锁定以 requirements.lock 为准）
├── configs/
│   ├── seeds.yaml            # 全局种子（选择种子 / 训练种子 / 验证划分种子）
│   ├── data.yaml             # 数据、划分、CLIP 特征口径、路径
│   ├── kernel.yaml           # 高斯核 σ/γ、α 取值表、μ 对角线口径
│   ├── selection.yaml        # 三种选择器配置与命名规范
│   └── fixmatch.yaml         # 统一训练配方（三方法完全共享）
├── src/rdss/
│   ├── models/wrn.py         # WRN-28-2（1.47M 参数，已验证前向形状）
│   ├── features/clip.py      # CLIP ViT-B/32 特征抽取（[CLS] / pooler 两种取法）
│   └── utils/config.py       # 统一 YAML 读取与路径解析
├── scripts/
│   └── check_env.py          # P0 自检：环境 + 配置 6 断言 + CLIP/分类头单批形状
├── tests/                    # P2/P3 补充 4 项测试
├── results/                  # 原始日志/指标（脚本生成，不手工编辑）
├── data/ splits/ features/ selections/ figures/   # 运行期数据（不入库）
├── paper/                    # 报告与图表源文件
└── docs/
    ├── P0_立项书.md
    ├── P0_仓库审计表.md
    └── P0_环境清单.md
```

## 安装

```bash
# 1) Python 3.10+（论文环境为 3.8，本项目在 3.12 上验证过基础前向）
conda create -n rdss python=3.10 -y && conda activate rdss

# 2) torch 按服务器 CUDA 选择（示例 cu121）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) 其余依赖
pip install -r requirements.txt

# 4) 锁定环境（P0 通过条件）
pip freeze > requirements.lock
nvidia-smi > hardware_report.txt
```

Windows PowerShell 版本见 `docs/P0_环境清单.md §2.2`。

## P0 自检（一键复现命令）

```bash
cd project
python scripts/check_env.py                       # 完整：CLIP + WRN-28-2 + 配置断言
python scripts/check_env.py --random-images       # 未下载 CIFAR-10 时用随机像素占位
python scripts/check_env.py --skip-clip           # 本地无 transformers 时
```

**预期输出**（关键行）：

```text
=== 配置校验 ===
OK: seeds/data/kernel/selection/fixmatch 校验通过（6 项断言）
=== CLIP ViT-B/32 单批前向 ===
features    : (8, 768) (torch.float32)  take=cls_last_hidden
=== WRN-28-2 单批前向 ===
logits: (8, 10)  params=1.47M
PASS: 配置可解析且自洽；一个 batch 过 CLIP 与分类头，形状正确。
```

退出码 0 表示 P0 通过；断言失败会指出具体配置项。

## P1 数据与特征（脚本已就绪，需在服务器执行）

```bash
cd project

# 1) 下载 CIFAR-10（约 170MB，只下一次；--check-only 可只校验）
python scripts/download_cifar10.py

# 2) 固定划分（已有产物时可省略；重建需 --force）
python scripts/split_val.py --val-seed 0 --output splits/

# 3) 抽 CLIP 特征（trainpool 45000 条 + test 10000 条）
python scripts/extract_clip_features.py --dataset cifar10 --model openai/clip-vit-base-patch32 \
    --pool trainpool --indices splits/trainpool_indices_seed0.json \
    --output features/cifar10_trainpool_clip.npy
python scripts/extract_clip_features.py --pool test --output features/cifar10_test_clip.npy

# 4) 计算 σ / γ 并回填 configs/kernel.yaml
python scripts/compute_sigma.py --feature-path features/cifar10_trainpool_clip.npy \
    --output configs/kernel.yaml

# 5) 单元测试（无需数据集）
python tests/test_p1_kernel_split.py
```

**预期输出**（关键行，数值以实测为准）：

```text
[OK] CIFAR-10 就绪：root=...\project\data
     train=50000  test=10000  classes=10
[OK] 划分完成（val_seed=0）：val=5000  train_pool=45000  test=10000
pool=trainpool  n=45000 ... take=cls_last_hidden l2_norm=True amp=True device=cuda
[OK] 已保存 ...\features\cifar10_trainpool_clip.npy  shape=(45000, 768)  dtype=float32
median sq-dist = 0.xxxxxx   →  σ = 0.xxxxxx   γ = x.xxxxxx
[OK] 已回填 configs/kernel.yaml
```

P1 通过条件对应关系：
1. **特征可复现** → 同参数再跑一次，`features/*.meta.json` 里的 `feature_sha1` 必须相同（不一致加 `--no-amp --deterministic`）。
2. **σ 可复算** → `features/kernel_sigma_stats.json` 记录特征 sha1、bin 数、分块大小与复算命令。
3. **索引可追溯** → `splits/splits_meta.json` 记录三个索引文件的 sha1 与 val_seed。

## P2 三种选择器（脚本已就绪）

```bash
# Random（3 个选择种子）
foreach ($s in 42,1234,12345) { python scripts/selection_random.py --budget 250 --seed $s --pool-size 45000 --output selections/random_m250_s$s.json }
# k-center（3 个选择种子）
foreach ($s in 42,1234,12345) { python scripts/selection_kcenter.py --budget 250 --seed $s --feature-path features/cifar10_trainpool_clip.npy --output selections/kcenter_m250_s$s.json }
# RDSS / GKHR（确定性，α 自动 = 1-1/√m）
python scripts/selection_rdss.py --budget 250 --feature-path features/cifar10_trainpool_clip.npy \
    --gamma-path configs/kernel.yaml --alpha auto --output selections/rdss_m250.json

# 与官方 RDSS.py 的一致性测试（真实特征；子集 5000 是因为官方要开 n×n float64 矩阵）
python scripts/check_rdss_consistency.py --feature-path features/cifar10_trainpool_clip.npy --budget 250 --n-subset 5000

# 二维 toy 直觉图（不需要数据集，可本地跑）
python scripts/viz_toy_selection.py --budget 100 --output figures/toy_selection.png

# 单元测试（无需数据集）
python tests/test_p2_selection.py
```

每次选择都会把耗时追加到 `results/p2/selection_timing.csv`（表由脚本生成，不手抄），并在 JSON 里记录
`alpha / gamma / feature_sha1 / label_used=false / index_semantics`。索引语义是**训练池内行号**。

已验证（本地合成特征）：三个选择器均满足"数量=m、无重复、范围 [0,n)"；GKHR 增量实现与"每轮重算"的朴素 Algorithm 1 索引逐位相同；
与官方 `RDSS.py` 在 n=2000、m=250、α=0.9368、γ=0.7071 下**索引完全一致**（μ 含/不含对角线两版都一致），耗时 21.45s → 1.8s（约 12×）。
真实特征上的复测结果以 `results/p2/consistency_rdss.json` 为准。

## P3 统一 FixMatch（脚本已就绪）

```bash
# 0) 先做纯监督 sanity check（只用 m 个有标签样本，期望 Acc ≫ 10%，排除标签泄漏）
python scripts/train_fixmatch.py --config configs/fixmatch.yaml \
    --labeled-list selections/random_m250_s42.json --supervised \
    --seed 42 --output-dir results/p3_sanity_sup_random_m250_seed42

# 1) FixMatch 单次跑通（random × 1 种子）
python scripts/train_fixmatch.py --config configs/fixmatch.yaml \
    --labeled-list selections/random_m250_s42.json \
    --val-indices splits/val_indices_seed0.json --test-indices splits/test_indices.json \
    --seed 42 --output-dir results/p3_sanity_random_m250_seed42

# 2) 无数据/无特征时验证训练循环与断点（随机像素，结论不可用）
python scripts/train_fixmatch.py --fake-data 800 --epochs 2 --max-steps 2 --batch-size 16 \
    --num-workers 0 --device cpu --output-dir results/_smoke

# 3) 断点续训
python scripts/train_fixmatch.py ... --resume results/_smoke/last.pth

# 4) 单元测试（无需数据集）
python tests/test_p3_fixmatch.py
```

**预期输出**（关键行）：

```text
有标签 250  无标注 44750  验证 5000  测试 10000
batch: labeled=64 unlabeled=448  steps/epoch=unlabeled=99
step    500/9900  loss=1.8321 loss_x=0.9712 loss_u=0.8609 mask=0.612 lr=0.02853
  [eval] step=500 val_acc=72.31 val_f1=71.88 best=72.31
[FINAL] best_step=4500  val_acc=86.42  test_acc=85.90  test_f1=85.61  elapsed=38.2min  peak_mem=5.31GB
[OK] 产物: .../{train.log, metrics.json, config.yaml, best_model.pth, last.pth}
```

口径与实现说明（写报告时必须原样交代）：
- **1 epoch = 一次未标注池遍历**（m=250 时 ≈100 step），否则 `eval_step=500` 无法落在有意义的位置；
- 验证集只用于选最佳 checkpoint，测试集在训练结束后**只评估一次**；`metrics.json` 的 `test` 字段即该次结果；
- 与 USB 版 FixMatch 的已知差异：**不含 EMA 影子模型**；因此绝对数值预期低于论文，但三方法同配方，相对比较仍成立；
- 无标注数据集接口只返回 `(弱增强, 强增强)`，不读取标签；有标签集合由选择结果的**池内行号**经 `trainpool_indices_seed0.json` 映射；
- 断点 `last.pth` 含模型/优化器/调度器/AMP/`torch/numpy/python` 随机状态。

## 后续阶段命令（P4–P6，尚未实现，按 §H 模板）

```bash
# P4 主实验矩阵（m × 方法 × 训练种子）
python scripts/summarize_main_table.py --results-dir results/ --output results/main_table.csv
python scripts/plot_budget_curve.py --results-dir results/ --output figures/budget_curve.png
```

## 关键口径（写报告时照抄，改口径必须先改配置并记录）

- **α**：默认 `1 − 1/√m`（m=250→0.9368，m=1000→0.9684）；α=1 只留代表性，α=0 只留多样性。
- **σ**：训练池 L2 归一化特征上 pairwise 距离的**中位数**（median heuristic），γ=1/σ。
- **索引语义**：选择器输出的是**训练池内行号** r（0..44999），原始 CIFAR 索引 = `trainpool_indices_seed0.json[r]`；两者不可混用。
- **μ 对角线**：主口径按论文含自身（`mu.diagonal=include`），与官方实现一致性测试后再定。
- **验证集**：5000 张、seed=0，只用于选最佳 checkpoint；**测试集只在最终评估用一次**。
- **选择阶段不读标签**：标签仅用于训练与事后诊断。

## 已知问题

1. 本地机器（RTX 4060 Laptop 8GB / D 盘剩余 4.1GB）**不做训练**，训练迁移远程服务器；本地未安装 `transformers`。
2. `requirements.lock` 必须在目标机器上 `pip freeze` 生成，仓库内暂缺。
3. 官方 `RDSS.py` 的 SVHN 分支有复制粘贴 bug（L94），本项目不使用 SVHN，已在审计表标注废弃。
4. 官方 `visualization.py` 的 stratified 基线按索引块切分，仅对 toy 数据成立，不能用于 CIFAR-10 公平对比。
5. `get_cifar_feature.py` 取 `pooler_output`，与论文 final [CLS] 仅差最后一层 LayerNorm；本项目默认 `cls_last_hidden`，三方法统一。
