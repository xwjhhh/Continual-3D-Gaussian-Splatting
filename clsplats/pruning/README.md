# 剪枝模块 / Pruning Module

> **复现项目说明 / Reproduction Notice**
> 
> 本项目是对 CL-Splats 的复现工作。原始项目请参见：https://github.com/jan-ackermann/cl-splats
> 
> This is a reproduction of CL-Splats. For the original project, see: https://github.com/jan-ackermann/cl-splats

---

[中文](#中文) | [English](#english)

---

# 中文

基于球体的空间剪枝，确保高斯优化保持在检测到的变化区域内。

## 概述

剪枝模块防止高斯点在局部优化过程中漂移到未改变的区域：

1. **HDBSCAN 聚类**：自动识别不同的变化区域
2. **包围球拟合**：为每个聚类拟合包围球
3. **周期性剪枝**：每 15 次迭代移除超出边界的高斯点

## 架构

```
pruning/
├── base_pruner.py       # 抽象基类
├── sphere_pruner.py     # 球体剪枝器实现
└── __init__.py
```

## 使用方法

```python
from clsplats.pruning import SpherePruner
import omegaconf

cfg = omegaconf.OmegaConf.create({
    'min_cluster_size': 1000,    # 最小聚类大小
    'quantile': 0.98,            # 半径百分位数
    'radius_multiplier': 1.1     # 安全边界因子
})

pruner = SpherePruner(cfg)

# 对变化的高斯点拟合球体
pruner.fit(changed_gaussians)  # [N, 3]

# 在优化循环中
for iteration in range(num_iterations):
    # ... 优化步骤 ...
    
    # 每 15 次迭代剪枝
    if iteration % 15 == 0:
        should_prune = pruner.should_prune(gaussians)
        gaussians = gaussians[~should_prune]
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_cluster_size` | 1000 | 形成聚类的最小点数 |
| `quantile` | 0.98 | 半径计算的百分位数 |
| `radius_multiplier` | 1.1 | 球体大小的安全因子 |

## 算法原理

1. **聚类**：使用 HDBSCAN 识别变化区域中的聚类
2. **球体拟合**：
   - 中心：`m_i = mean(cluster_points)`
   - 半径：`r_i = quantile_98(distances) × 1.1`
3. **剪枝**：移除不在任何球体内的高斯点

## API

```python
# 获取球体数量
num_spheres = pruner.get_num_spheres()

# 检查点是否在边界内
inside_mask = pruner.is_inside(points)

# 获取球体信息
sphere_info = pruner.get_sphere_info()
# [{'center': [x, y, z], 'radius': r}, ...]
```

## 性能

- 聚类：~50ms/1000 点
- 剪枝检查：~1ms/1000 点

---

# English

Sphere-based spatial pruning to ensure Gaussian optimization stays within detected change regions.

## Overview

The pruning module prevents Gaussians from drifting to unchanged regions during local optimization:

1. **HDBSCAN Clustering**: Automatically identify distinct change regions
2. **Bounding Sphere Fitting**: Fit bounding spheres for each cluster
3. **Periodic Pruning**: Remove Gaussians outside bounds every 15 iterations

## Architecture

```
pruning/
├── base_pruner.py       # Abstract base class
├── sphere_pruner.py     # Sphere pruner implementation
└── __init__.py
```

## Usage

```python
from clsplats.pruning import SpherePruner
import omegaconf

cfg = omegaconf.OmegaConf.create({
    'min_cluster_size': 1000,    # Minimum cluster size
    'quantile': 0.98,            # Radius percentile
    'radius_multiplier': 1.1     # Safety margin factor
})

pruner = SpherePruner(cfg)

# Fit spheres to changed Gaussians
pruner.fit(changed_gaussians)  # [N, 3]

# In optimization loop
for iteration in range(num_iterations):
    # ... optimization step ...
    
    # Prune every 15 iterations
    if iteration % 15 == 0:
        should_prune = pruner.should_prune(gaussians)
        gaussians = gaussians[~should_prune]
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 1000 | Minimum points to form a cluster |
| `quantile` | 0.98 | Percentile for radius calculation |
| `radius_multiplier` | 1.1 | Safety factor for sphere size |

## Algorithm

1. **Clustering**: Use HDBSCAN to identify clusters in change regions
2. **Sphere Fitting**:
   - Center: `m_i = mean(cluster_points)`
   - Radius: `r_i = quantile_98(distances) × 1.1`
3. **Pruning**: Remove Gaussians not inside any sphere

## API

```python
# Get number of spheres
num_spheres = pruner.get_num_spheres()

# Check if points are inside bounds
inside_mask = pruner.is_inside(points)

# Get sphere information
sphere_info = pruner.get_sphere_info()
# [{'center': [x, y, z], 'radius': r}, ...]
```

## Performance

- Clustering: ~50ms/1000 points
- Pruning check: ~1ms/1000 points
