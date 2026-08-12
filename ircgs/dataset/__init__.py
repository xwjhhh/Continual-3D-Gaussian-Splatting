"""
Dataset module for CL-Splats.
"""

from ircgs.dataset.dataset_reader import (
    CLSplatsDataset,
    CameraInfo,
    SceneInfo,
    load_dataset
)

__all__ = ["CLSplatsDataset", "CameraInfo", "SceneInfo", "load_dataset"]
