# 采样模块 / Sampling Module

> **复现项目说明 / Reproduction Notice**
> 
> 本项目是对 CL-Splats 的复现工作。原始项目请参见：https://github.com/jan-ackermann/cl-splats
> 
> This is a reproduction of CL-Splats. For the original project, see: https://github.com/jan-ackermann/cl-splats

---

[中文](#中文) | [English](#english)

---

# 中文

基于高斯混合模型的点采样，用于在变化区域生成新的 3D 高斯点。

## 概述

当场景中出现新物体时，需要采样新的 3D 点。本模块实现两种策略：

1. **全场景采样（算法2）**：在场景边界框内均匀采样
2. **区域采样（算法3）**：使用 GMM 在现有变化点周围采样

## 架构

```
sampling/
├── base_sampler.py       # 抽象基类
├── gaussian_sampler.py   # GMM 采样器实现
└── __init__.py
```

## 使用方法

```python
from ircgs.sampling import GaussianSampler
from ircgs.utils.majority_vote_lifter import MajorityVoteLifter
import omegaconf

lifter_cfg = omegaconf.OmegaConf.create({'vote_threshold': 2})
lifter = MajorityVoteLifter(lifter_cfg)

cfg = omegaconf.OmegaConf.create({
    'num_clusters': 10,                      # K-Means 聚类数
    'samples_per_round': 100,                # 每轮采样数
    'max_rounds': 10,                        # 最大轮数
    'min_points_for_region_sampling': 50,    # 策略切换阈值
})

sampler = GaussianSampler(cfg, lifter)

# 采样新点
sampled_points, valid_mask = sampler.sample_new_points(
    existing_gaussians,  # [N, 3] 现有高斯点
    change_masks_2d,     # List[Tensor] 2D 变化掩码
    cameras,             # List[Camera] 相机列表
    num_samples=500      # 目标采样数
)

# 提取有效样本
valid_samples = sampler.get_valid_samples(sampled_points, valid_mask)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_clusters` | 10 | K-Means 聚类数量 |
| `samples_per_round` | 100 | 每轮采样数量 |
| `max_rounds` | 10 | 最大采样轮数 |
| `min_points_for_region_sampling` | 50 | 切换策略的阈值 |

## 策略选择

- **变化点 < 50**：使用全场景采样（新物体出现）
- **变化点 ≥ 50**：使用区域采样（物体移动/修改）

## 性能

| 策略 | 采样效率 | 速度 |
|------|----------|------|
| 全场景采样 | 10-20% | ~50ms/轮 |
| 区域采样 | 30-50% | ~30ms/轮 |

---

# English

GMM-based point sampling for generating new 3D Gaussian points in changed regions.

## Overview

When new objects appear in the scene, new 3D points need to be sampled. This module implements two strategies:

1. **Full-scene Sampling (Algorithm 2)**: Uniform sampling within scene bounding box
2. **Region Sampling (Algorithm 3)**: GMM sampling around existing changed points

## Architecture

```
sampling/
├── base_sampler.py       # Abstract base class
├── gaussian_sampler.py   # GMM sampler implementation
└── __init__.py
```

## Usage

```python
from ircgs.sampling import GaussianSampler
from ircgs.utils.majority_vote_lifter import MajorityVoteLifter
import omegaconf

lifter_cfg = omegaconf.OmegaConf.create({'vote_threshold': 2})
lifter = MajorityVoteLifter(lifter_cfg)

cfg = omegaconf.OmegaConf.create({
    'num_clusters': 10,                      # K-Means clusters
    'samples_per_round': 100,                # Samples per round
    'max_rounds': 10,                        # Maximum rounds
    'min_points_for_region_sampling': 50,    # Strategy switch threshold
})

sampler = GaussianSampler(cfg, lifter)

# Sample new points
sampled_points, valid_mask = sampler.sample_new_points(
    existing_gaussians,  # [N, 3] existing Gaussians
    change_masks_2d,     # List[Tensor] 2D change masks
    cameras,             # List[Camera] camera list
    num_samples=500      # Target sample count
)

# Extract valid samples
valid_samples = sampler.get_valid_samples(sampled_points, valid_mask)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | 10 | Number of K-Means clusters |
| `samples_per_round` | 100 | Samples per iteration |
| `max_rounds` | 10 | Maximum sampling rounds |
| `min_points_for_region_sampling` | 50 | Threshold for strategy switch |

## Strategy Selection

- **Changed points < 50**: Use full-scene sampling (new object appeared)
- **Changed points ≥ 50**: Use region sampling (object moved/modified)

## Performance

| Strategy | Sampling Efficiency | Speed |
|----------|---------------------|-------|
| Full-scene | 10-20% | ~50ms/round |
| Region | 30-50% | ~30ms/round |
