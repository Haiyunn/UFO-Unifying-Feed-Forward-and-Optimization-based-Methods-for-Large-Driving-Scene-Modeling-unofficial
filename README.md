# UFO（非官方复现）

> **声明：本仓库是论文 UFO 的非官方复现代码**，与原作者无关。官方仓库（[wm-research/UFO](https://github.com/wm-research/UFO)）截至 2026-10-21 只有 README、没有代码和权重，因此这里的全部代码根据论文描述从零实现，可能与官方实现存在差异。
>
> **Disclaimer: this is an UNOFFICIAL reimplementation**, written from the paper only and not affiliated with the authors.

论文：**UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling**
Tan et al. · CVPR 2026 · [arXiv:2602.20943](https://arxiv.org/abs/2602.20943)

## 实现内容

`ufo/model.py`（共 92.0M 参数，论文约 92M）：

- 场景 token（xyz + 768 维）；图像 token = 8×8 patch 的 RGB + Plücker 射线；box token（中心 + 8 角点的 Fourier 编码）；天空 token；逐相机仿射 token
- 12 层 768 维 ViT 更新 transformer，可见性过滤（视锥 + 最近 K=3600），相机局部坐标系
- 替换并追加的循环更新（Eq. 4）、软物体分配（Eq. 5）、box 加权运动（Eq. 6–7）、分配交叉熵（Eq. 8）、寿命不透明度与正则（Eq. 9–10）
- 2 层 MLP 高斯解码器，gsplat 渲染，AdamW 1e-4 + 线性 warmup

论文未说明的设计选择（新 token 深度、每 token 高斯数 M=16、运动混合方式等）全部记录在 [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)。

## 目录结构

```
ufo/           模型、几何、损失、渲染、数据（Waymo / Argoverse 预处理与深度先验）
scripts/       train.py  eval.py  render_shift.py  bench_scalability.py  sample_scene.py
configs/       stage1_2s.yaml（2 s 预训练）  stage2_8s.yaml（8 s 微调）等
splits/        数据划分
tests/         冒烟测试、几何与数学检查
*.sh           训练 / 评测流水线示例
```

## 环境

Python 3.10+、PyTorch 2.x（CUDA）、[gsplat](https://github.com/nerfstudio-project/gsplat)、numpy、imageio、matplotlib、Pillow、PyYAML。

## 使用

数据为 Waymo NOTR（[DriveStudio](https://github.com/ziyc/drivestudio) 预处理格式），160×240，3 个前视相机。所有路径都通过命令行参数传入，默认值相对仓库根目录（`data/waymo/processed/training`、`data/argoverse/processed/training`、`third_party/InfiniDepth`）。

```bash
# 1. 预处理成 160x240 缓存（cache/waymo_160x240、cache/argoverse_160x240）
python -m ufo.data.preprocess --root /path/to/waymo/processed/training
bash preprocess_argo.sh /path/to/argoverse/processed/training
# 2. 深度先验（可选，v2 配置使用）
python -m ufo.data.depth_prior --cache cache/waymo_160x240 --dataset waymo \
    --root /path/to/waymo/processed/training --scenes 016 017 \
    --depth-repo /path/to/InfiniDepth --depth-ckpt /path/to/infinidepth_depthsensor.ckpt
# 训练脚本只读取缓存（配置中的 cache_root），不再需要原始数据路径
# 阶段 1：2 s 序列训练（或 bash launch.sh configs/stage1_2s.yaml 0,1 29500）
torchrun --nproc_per_node 2 -m scripts.train --config configs/stage1_2s.yaml
# 阶段 2：8 s 序列微调
torchrun --nproc_per_node 2 -m scripts.train --config configs/stage2_8s.yaml --init runs/stage1_2s/ckpt_latest.pt
# 评测（2 / 8 / 16 s 窗口，PSNR / SSIM / D-RMSE / LPIPS）
python -m scripts.eval --ckpt runs/stage2_8s/ckpt_latest.pt --seconds 2 8 16 --out results/stage2_8s.json
```

## 与论文设置的差异

| 项 | 本仓库 | 论文 |
|---|---|---|
| 训练数据 | PointForward 划分：60 训练 / 11 验证场景 | WOD 官方划分（798 训练） |
| 3D 框 | Waymo GT 轨迹 | 现成检测 + 跟踪器 |

训练场景远少于论文，这是与论文指标存在差距的主要原因。

## 引用

请引用原论文：

```bibtex
@inproceedings{tan2026ufo,
  title     = {UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling},
  author    = {Tan and others},
  booktitle = {CVPR},
  year      = {2026}
}
```
