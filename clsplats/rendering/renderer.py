"""
Gaussian Splatting Renderer.

Wrapper around diff-gaussian-rasterization for rendering Gaussian scenes.
"""

import torch
import math
from typing import Optional

# Try to import the CUDA rasterizer
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer
    )
    RASTERIZER_AVAILABLE = True
except ImportError:
    RASTERIZER_AVAILABLE = False
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    print("Warning: diff_gaussian_rasterization not available. Rendering will not work.")


def _make_raster_settings(**kwargs):
    """Create settings across rasterizer builds with slightly different APIs."""
    fields = getattr(GaussianRasterizationSettings, "_fields", None)
    if fields is not None:
        kwargs = {key: value for key, value in kwargs.items() if key in fields}
    return GaussianRasterizationSettings(**kwargs)


def _empty_depth_like(rendered_image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return torch.zeros(
        (1, height, width),
        device=rendered_image.device,
        dtype=rendered_image.dtype,
    )


def render(
    viewpoint_camera,
    gaussians,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    override_shs: Optional[torch.Tensor] = None,
    antialiasing: bool = False,
    active_gaussian_mask: Optional[torch.Tensor] = None,
    tile_mask: Optional[torch.Tensor] = None,
):
    """
    Render the scene from a viewpoint.
    
    Args:
        viewpoint_camera: Camera object with view/projection matrices
        gaussians: GaussianModel containing the scene
        bg_color: Background color [3]
        scaling_modifier: Scale modifier for Gaussians
        override_color: Optional override for Gaussian colors
        override_shs: Optional override for Gaussian SH features [N, coeff, 3]
        antialiasing: Whether to use antialiasing
        active_gaussian_mask: Optional mask for local optimization (CL-Splats)
        tile_mask: Optional tile mask for local optimization (CL-Splats)
        
    Returns:
        Dictionary containing:
            - render: Rendered image [3, H, W]
            - viewspace_points: 2D positions of Gaussians
            - visibility_filter: Mask of visible Gaussians
            - radii: 2D radii of Gaussians
            - depths: Mean depth map [1, H, W] when supported by rasterizer
    """
    if not RASTERIZER_AVAILABLE:
        raise RuntimeError(
            "diff_gaussian_rasterization is not available. "
            "Please install it from submodules/diff-gaussian-rasterization"
        )
    
    # Create rasterization settings
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
    raster_settings = _make_raster_settings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=antialiasing,
        active_gaussian_mask=active_gaussian_mask,
        tile_mask=tile_mask,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Get Gaussian properties
    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
    
    try:
        means2D.retain_grad()
    except:
        pass
    
    opacity = gaussians.get_opacity
    
    # Get scales and rotations
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    
    # Get colors (SH or override). Exactly one override mode can be active.
    if override_color is not None and override_shs is not None:
        raise ValueError("render() received both override_color and override_shs; only one is allowed.")
    if override_color is None:
        shs = gaussians.get_features if override_shs is None else override_shs
    else:
        shs = None
    
    # Rasterize. Some rasterizer builds additionally return mean depth.
    raster_out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=override_color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    if len(raster_out) == 2:
        rendered_image, radii = raster_out
        invdepths = _empty_depth_like(rendered_image, image_height, image_width)
        mean_depths = _empty_depth_like(rendered_image, image_height, image_width)
    elif len(raster_out) == 3:
        rendered_image, radii, invdepths = raster_out
        mean_depths = _empty_depth_like(rendered_image, image_height, image_width)
    elif len(raster_out) == 4:
        rendered_image, radii, invdepths, mean_depths = raster_out
    else:
        raise RuntimeError(
            f"Unexpected GaussianRasterizer output size={len(raster_out)}; expected 2, 3, or 4."
        )
    
    # Visibility filter
    visibility_filter = radii > 0
    
    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "invdepths": invdepths,
        "depths": mean_depths,
    }


def render_simple(
    viewpoint_camera,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    bg_color: torch.Tensor,
    sh_degree: int = 0,
    antialiasing: bool = False,
    active_gaussian_mask: Optional[torch.Tensor] = None,
    tile_mask: Optional[torch.Tensor] = None,
):
    """
    Simplified render function with explicit parameters.
    
    Useful for rendering without a full GaussianModel.
    """
    if not RASTERIZER_AVAILABLE:
        raise RuntimeError("diff_gaussian_rasterization is not available.")
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = _make_raster_settings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=antialiasing,
        active_gaussian_mask=active_gaussian_mask,
        tile_mask=tile_mask,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
    
    raster_out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    if len(raster_out) == 2:
        rendered_image, radii = raster_out
    elif len(raster_out) == 3:
        rendered_image, radii, _ = raster_out
    elif len(raster_out) == 4:
        rendered_image, radii, _, _ = raster_out
    else:
        raise RuntimeError(
            f"Unexpected GaussianRasterizer output size={len(raster_out)}; expected 2, 3, or 4."
        )

    return rendered_image, radii
