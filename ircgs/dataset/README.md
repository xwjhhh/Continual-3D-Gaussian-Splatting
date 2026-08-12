# 数据集模块 / Dataset Module

> **复现项目说明 / Reproduction Notice**
> 
> 本项目是对 CL-Splats 的复现工作。原始项目请参见：https://github.com/jan-ackermann/cl-splats
> 
> This is a reproduction of CL-Splats. For the original project, see: https://github.com/jan-ackermann/cl-splats

---

[中文](#中文) | [English](#english)

---

# 中文

负责加载和管理 CL-Splats 所需的数据，包括图像、相机参数和 COLMAP 重建结果。

## 架构
python -m ircgs.train   dataset.path=/root/autodl-tmp/cl-splats-reproduction-main/data/cl-splats/Blender-Levels/Level-1   dataset.type=blender   dataset.change_type=delete   dataset.eval=true  train.num_times=2   train.iterations=30000   train.incremental_iterations=1000   resume_from=/root/autodl-tmp/cl-splats-reproduction-main/outputs/checkpoint_t0.pt   lifter.geometry_backend=mast3r   lifter.mast3r.model_name_or_path=/root/autodl-tmp/hf_models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric   lifter.mast3r.cache_dir=/root/autodl-tmp/mast3r_cache
```python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ackermannj/cl-splats-dataset",
    repo_type="dataset",
    local_dir="autodl-tmp/cl-splats-reproduction-main/data",
    resume_download=True
)
PY
dataset/
├── cameras.py           # 相机类定义
├── colmap_reader.py     # COLMAP 数据读取器
├── dataset_reader.py    # 数据集读取器
└── __init__.py
```

## 数据格式

### 目录结构

```
your_dataset/
├── t0/                          # 时间步 0
│   ├── images/
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── t1/                          # 时间步 1
│   ├── images/
│   └── sparse/0/
└── ...
```

### 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)

## 使用方法

```python
from ircgs.dataset import CLSplatsDataset

# 加载数据集
dataset = CLSplatsDataset(
    path="your_dataset/output",
    resolution_scale=1.0,
    white_background=False
)

# 获取时间步数量
num_timesteps = dataset.get_num_timesteps()

# 获取特定时间步的相机和图像
cameras = dataset.get_cameras(timestep=0)
images = dataset.get_images(timestep=0)

# 获取场景信息（点云等）
scene_info = dataset.get_scene_info(timestep=0)
```

## Camera 类

```python
class Camera:
    uid: int                    # 唯一标识符
    R: np.ndarray              # 旋转矩阵 [3, 3]
    T: np.ndarray              # 平移向量 [3]
    FoVx: float                # 水平视场角
    FoVy: float                # 垂直视场角
    original_image: Tensor     # 原始图像 [3, H, W]
    image_width: int
    image_height: int
    world_view_transform: Tensor      # 世界到视图变换
    projection_matrix: Tensor         # 投影矩阵
    full_proj_transform: Tensor       # 完整投影变换
    camera_center: Tensor             # 相机中心
```

## 数据预处理

使用预处理脚本自动处理原始数据：

```bash
python -m ircgs.utils.preprocessing --input_dir your_dataset
```

脚本会：
1. 运行 COLMAP 特征提取和匹配
2. 进行稀疏重建
3. 增量注册新时间步图像
4. 去畸变所有图像
5. 生成训练所需的目录结构

---

# English

Responsible for loading and managing data required by CL-Splats, including images, camera parameters, and COLMAP reconstruction results.

## Architecture

```
dataset/
├── cameras.py           # Camera class definition
├── colmap_reader.py     # COLMAP data reader
├── dataset_reader.py    # Dataset reader
└── __init__.py
```

## Data Format

### Directory Structure

```
your_dataset/
├── t0/                          # Timestep 0
│   ├── images/
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── t1/                          # Timestep 1
│   ├── images/
│   └── sparse/0/
└── ...
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Usage

```python
from ircgs.dataset import CLSplatsDataset

# Load dataset
dataset = CLSplatsDataset(
    path="your_dataset/output",
    resolution_scale=1.0,
    white_background=False
)

# Get number of timesteps
num_timesteps = dataset.get_num_timesteps()

# Get cameras and images for specific timestep
cameras = dataset.get_cameras(timestep=0)
images = dataset.get_images(timestep=0)

# Get scene info (point cloud, etc.)
scene_info = dataset.get_scene_info(timestep=0)
```

## Camera Class

```python
class Camera:
    uid: int                    # Unique identifier
    R: np.ndarray              # Rotation matrix [3, 3]
    T: np.ndarray              # Translation vector [3]
    FoVx: float                # Horizontal field of view
    FoVy: float                # Vertical field of view
    original_image: Tensor     # Original image [3, H, W]
    image_width: int
    image_height: int
    world_view_transform: Tensor      # World to view transform
    projection_matrix: Tensor         # Projection matrix
    full_proj_transform: Tensor       # Full projection transform
    camera_center: Tensor             # Camera center
```

## Data Preprocessing

Use the preprocessing script to automatically process raw data:

```bash
python -m ircgs.utils.preprocessing --input_dir your_dataset
```

The script will:
1. Run COLMAP feature extraction and matching
2. Perform sparse reconstruction
3. Incrementally register new timestep images
4. Undistort all images
5. Generate directory structure for training
