"""
Utility functions for CL-Splats.
"""

from ircgs.utils.loss_utils import l1_loss, l2_loss, ssim, combined_loss
from ircgs.utils.general_utils import (
    inverse_sigmoid,
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
)
from ircgs.utils.graphics_utils import (
    BasicPointCloud,
    getWorld2View2,
    getProjectionMatrix,
    focal2fov,
    fov2focal,
)
from ircgs.utils.sh_utils import RGB2SH, SH2RGB, eval_sh
from ircgs.utils.camera_utils import Camera, MiniCam, loadCam
from ircgs.utils.base_lifter import BaseLifter
from ircgs.utils.majority_vote_lifter import MajorityVoteLifter
from ircgs.utils.hexplane_utils import (
    get_normalized_directions,
    normalize_aabb,
    grid_sample_wrapper,
    init_grid_param,
    interpolate_ms_features,
    HexPlaneField,
)
from ircgs.utils.deformation_utils import (
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
