"""
Camera utilities for CL-Splats.
"""

import torch
import numpy as np
from typing import NamedTuple

from ircgs.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:
    """
    Camera class for rendering.
    
    Stores camera intrinsics and extrinsics, and provides
    transformation matrices for rendering.
    """
    
    def __init__(
        self,
        uid: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image_width: int,
        image_height: int,
        image_name: str = "",
        image: torch.Tensor = None,
        object_mask: torch.Tensor = None,
        znear: float = 0.01,
        zfar: float = 100.0,
        device: str = "cuda"
    ):
        """
        Initialize camera.
        
        Args:
            uid: Unique camera ID
            R: Rotation matrix [3, 3]
            T: Translation vector [3]
            FoVx: Horizontal field of view (radians)
            FoVy: Vertical field of view (radians)
            image_width: Image width in pixels
            image_height: Image height in pixels
            image_name: Name of the image
            image: Ground truth image tensor
            object_mask: Optional per-pixel object mask tensor
            znear: Near clipping plane
            zfar: Far clipping plane
            device: Device to store tensors
        """
        self.uid = uid
        self.image_name = image_name
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.znear = znear
        self.zfar = zfar
        self.device = device
        
        # Store original image
        self.original_image = image
        self.object_mask = object_mask.to(device) if object_mask is not None else None
        
        # Compute transformation matrices
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T), dtype=torch.float32, device=device
        ).transpose(0, 1)
        
        self.projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy
        ).transpose(0, 1).to(device)
        
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        
        # Camera center in world coordinates
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """Minimal camera for rendering without full camera info."""
    
    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def loadCam(cam_info, resolution_scale: float = 1.0, device: str = "cuda"):
    """
    Load camera from CameraInfo.
    
    Args:
        cam_info: CameraInfo namedtuple
        resolution_scale: Scale factor for resolution
        device: Device to store tensors
        
    Returns:
        Camera object
    """
    from PIL import Image
    
    # Load and resize image if needed
    orig_w, orig_h = cam_info.width, cam_info.height
    
    if resolution_scale != 1.0:
        new_w = int(orig_w / resolution_scale)
        new_h = int(orig_h / resolution_scale)
    else:
        new_w, new_h = orig_w, orig_h
    
    # Load image
    image = None
    if cam_info.image_path:
        try:
            pil_image = Image.open(cam_info.image_path)
            if resolution_scale != 1.0:
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            image = torch.from_numpy(np.array(pil_image)).float() / 255.0
            if image.dim() == 2:
                image = image.unsqueeze(-1).repeat(1, 1, 3)
            image = image.permute(2, 0, 1).to(device)  # [C, H, W]
        except Exception as e:
            print(f"Warning: Could not load image {cam_info.image_path}: {e}")
    
    return Camera(
        uid=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image_width=new_w,
        image_height=new_h,
        image_name=cam_info.image_name,
        image=image,
        device=device
    )
