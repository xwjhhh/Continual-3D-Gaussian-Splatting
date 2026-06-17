"""
Utility functions for CL-Splats.
"""

from clsplats.utils.loss_utils import l1_loss, l2_loss, ssim, combined_loss
from clsplats.utils.general_utils import (
    inverse_sigmoid,
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
)
from clsplats.utils.graphics_utils import (
    BasicPointCloud,
    getWorld2View2,
    getProjectionMatrix,
    focal2fov,
    fov2focal,
)
from clsplats.utils.sh_utils import RGB2SH, SH2RGB, eval_sh
from clsplats.utils.camera_utils import Camera, MiniCam, loadCam
from clsplats.utils.base_lifter import BaseLifter
from clsplats.utils.majority_vote_lifter import MajorityVoteLifter
from clsplats.utils.hexplane_utils import (
    get_normalized_directions,
    normalize_aabb,
    grid_sample_wrapper,
    init_grid_param,
    interpolate_ms_features,
    HexPlaneField,
)
from clsplats.utils.deformation_utils import (
    SpatialTriPlaneField,
    Deformation,
)

__all__ = [
    # Loss functions
    "l1_loss", "l2_loss", "ssim", "combined_loss",
    # General utils
    "inverse_sigmoid", "build_rotation", "build_scaling_rotation", "get_expon_lr_func",
    # Graphics utils
    "BasicPointCloud", "getWorld2View2", "getProjectionMatrix", "focal2fov", "fov2focal",
    # SH utils
    "RGB2SH", "SH2RGB", "eval_sh",
    # Camera utils
    "Camera", "MiniCam", "loadCam",
    # Lifter utils
    "BaseLifter", "MajorityVoteLifter",
    # 4DGS hexplane utils
    "get_normalized_directions", "normalize_aabb", "grid_sample_wrapper",
    "init_grid_param", "interpolate_ms_features", "HexPlaneField",
    # 4DGS-style deformation utils
    "SpatialTriPlaneField", "Deformation",
]
