"""
Graphics utility functions for CL-Splats.
"""

from typing import NamedTuple
import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    """Basic point cloud representation."""
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """Get world to view transformation matrix."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Get projection matrix from camera parameters."""
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    """Convert focal length to field of view."""
    return 2 * np.arctan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Convert field of view to focal length."""
    return pixels / (2 * np.tan(fov / 2))
