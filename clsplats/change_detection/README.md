# 变化检测模块 / Change Detection Module

> **复现项目说明 / Reproduction Notice**
> 
> 本项目是对 CL-Splats 的复现工作。原始项目请参见：https://github.com/jan-ackermann/cl-splats
> 
> This is a reproduction of CL-Splats. For the original project, see: https://github.com/jan-ackermann/cl-splats

---

[中文](#中文) | [English](#english)

---

# 中文

基于 DINOv2 的语义变化检测，用于识别场景中发生变化的区域。

## 概述

变化检测是 CL-Splats 流程的第一步：
1. 比较渲染图像与新观测图像
2. 生成 2D 变化掩码
3. 用于后续的 3D 高斯点识别和采样

## 架构

```
change_detection/
├── base_detector.py      # 抽象基类
├── dinov2_detector.py    # DINOv2 检测器实现
└── __init__.py
```

## 使用方法

```python
from clsplats.change_detection import DinoV2Detector
import omegaconf

cfg = omegaconf.OmegaConf.create({
    'threshold': 0.2,           # 余弦相似度阈值（越小越敏感）
    'dilate_mask': True,        # 是否膨胀掩码
    'dilate_kernel_size': 13,   # 膨胀核大小
    'upsample': True            # 是否上采样到原始分辨率
})

detector = DinoV2Detector(cfg)

# 输入: [H, W, 3] torch.Tensor，范围 [0, 1]
mask = detector.predict_change_mask(rendered_image, observed_image)
# 输出: [H, W] bool tensor，True 表示变化区域
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 0.2 | 余弦相似度阈值，低于此值认为变化 |
| `dilate_mask` | True | 是否膨胀掩码以覆盖边缘 |
| `dilate_kernel_size` | 13 | 膨胀核大小（像素） |
| `upsample` | True | 是否上采样到原始分辨率 |

## 工作原理

1. **特征提取**：使用 DINOv2-ViT-B/14 提取语义特征
2. **相似度计算**：计算两张图像特征的余弦相似度
3. **阈值化**：相似度低于阈值的像素标记为变化
4. **后处理**：膨胀掩码以确保完整覆盖

## 性能

- 速度：~30ms/图像对（RTX 3090）
- 召回率：96.1%
- 模型大小：~350MB

---

# English

DINOv2-based semantic change detection for identifying changed regions in scenes.

## Overview

Change detection is the first step in the CL-Splats pipeline:
1. Compare rendered images with new observations
2. Generate 2D change masks
3. Used for subsequent 3D Gaussian identification and sampling

## Architecture

```
change_detection/
├── base_detector.py      # Abstract base class
├── dinov2_detector.py    # DINOv2 detector implementation
└── __init__.py
```

## Usage

```python
from clsplats.change_detection import DinoV2Detector
import omegaconf

cfg = omegaconf.OmegaConf.create({
    'threshold': 0.2,           # Cosine similarity threshold (lower = more sensitive)
    'dilate_mask': True,        # Whether to dilate mask
    'dilate_kernel_size': 13,   # Dilation kernel size
    'upsample': True            # Whether to upsample to original resolution
})

detector = DinoV2Detector(cfg)

# Input: [H, W, 3] torch.Tensor, range [0, 1]
mask = detector.predict_change_mask(rendered_image, observed_image)
# Output: [H, W] bool tensor, True indicates changed region
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.2 | Cosine similarity threshold, below which is considered changed |
| `dilate_mask` | True | Whether to dilate mask to cover edges |
| `dilate_kernel_size` | 13 | Dilation kernel size (pixels) |
| `upsample` | True | Whether to upsample to original resolution |

## How It Works

1. **Feature Extraction**: Extract semantic features using DINOv2-ViT-B/14
2. **Similarity Computation**: Compute cosine similarity between image features
3. **Thresholding**: Mark pixels with similarity below threshold as changed
4. **Post-processing**: Dilate mask to ensure complete coverage

## Performance

- Speed: ~30ms/image pair (RTX 3090)
- Recall: 96.1%
- Model size: ~350MB
