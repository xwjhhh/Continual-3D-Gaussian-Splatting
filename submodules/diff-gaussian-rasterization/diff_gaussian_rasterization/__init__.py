#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple, Optional
import torch.nn as nn
import torch
import math
from . import _C


class SparseGaussianAdam(torch.optim.Optimizer):
    """
    Sparse Adam optimizer for Gaussian Splatting with Local Optimization.
    
    This optimizer only updates parameters for active Gaussians, which significantly
    improves training efficiency when only a subset of Gaussians need to be optimized
    (e.g., in continual learning scenarios like CL-Splats).
    
    The sparse update is controlled by setting `visible_mask` before each step,
    which indicates which Gaussians should receive updates.
    
    Based on the Adam algorithm but with sparse updates:
    - Only active Gaussians' parameters are updated
    - Only active Gaussians' momentum (exp_avg) and variance (exp_avg_sq) are updated
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15):
        """
        Initialize SparseGaussianAdam optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseGaussianAdam, self).__init__(params, defaults)
        
        # Mask for visible/active Gaussians (set before each step)
        self._visible_mask = None
    
    def set_visible_mask(self, mask: Optional[torch.Tensor]):
        """
        Set the visibility mask for sparse updates.
        
        Args:
            mask: Boolean tensor [N] indicating which Gaussians are active.
                  If None, all Gaussians will be updated (dense update).
        """
        self._visible_mask = mask
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with sparse updates.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('SparseGaussianAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                
                # Determine which elements to update
                if self._visible_mask is not None and p.dim() >= 1:
                    # For Gaussian parameters, first dimension is the Gaussian index
                    # Only update active Gaussians
                    mask = self._visible_mask
                    
                    # Handle dimension mismatch
                    if mask.shape[0] == p.shape[0]:
                        # Sparse update for active Gaussians only
                        if p.dim() == 1:
                            # 1D tensor (e.g., opacity): [N]
                            active_grad = grad[mask]
                            active_exp_avg = exp_avg[mask]
                            active_exp_avg_sq = exp_avg_sq[mask]
                            active_param = p.data[mask].clone()
                            
                            # Update biased first moment estimate
                            active_exp_avg.mul_(beta1).add_(active_grad, alpha=1 - beta1)
                            # Update biased second raw moment estimate
                            active_exp_avg_sq.mul_(beta2).addcmul_(active_grad, active_grad, value=1 - beta2)
                            
                            # Bias correction
                            bias_correction1 = 1 - beta1 ** state['step']
                            bias_correction2 = 1 - beta2 ** state['step']
                            
                            # Compute step size
                            step_size = lr / bias_correction1
                            
                            # Compute denominator
                            denom = (active_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                            
                            # Update parameters
                            active_param.addcdiv_(active_exp_avg, denom, value=-step_size)
                            
                            # Write back using .data to avoid autograd issues
                            exp_avg.data[mask] = active_exp_avg
                            exp_avg_sq.data[mask] = active_exp_avg_sq
                            p.data[mask] = active_param
                            
                        else:
                            # Multi-dimensional tensor (e.g., xyz [N,3], features [N,C,D])
                            active_grad = grad[mask]
                            active_exp_avg = exp_avg[mask]
                            active_exp_avg_sq = exp_avg_sq[mask]
                            active_param = p.data[mask].clone()
                            
                            # Update biased first moment estimate
                            active_exp_avg.mul_(beta1).add_(active_grad, alpha=1 - beta1)
                            # Update biased second raw moment estimate
                            active_exp_avg_sq.mul_(beta2).addcmul_(active_grad, active_grad, value=1 - beta2)
                            
                            # Bias correction
                            bias_correction1 = 1 - beta1 ** state['step']
                            bias_correction2 = 1 - beta2 ** state['step']
                            
                            # Compute step size
                            step_size = lr / bias_correction1
                            
                            # Compute denominator
                            denom = (active_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                            
                            # Update parameters
                            active_param.addcdiv_(active_exp_avg, denom, value=-step_size)
                            
                            # Write back using .data to avoid autograd issues
                            exp_avg.data[mask] = active_exp_avg
                            exp_avg_sq.data[mask] = active_exp_avg_sq
                            p.data[mask] = active_param
                    else:
                        # Mask size doesn't match, fall back to dense update
                        self._dense_adam_step(p, grad, exp_avg, exp_avg_sq, state['step'], 
                                             beta1, beta2, lr, eps)
                else:
                    # No mask or non-Gaussian parameter, use dense update
                    self._dense_adam_step(p, grad, exp_avg, exp_avg_sq, state['step'],
                                         beta1, beta2, lr, eps)
        
        return loss
    
    def _dense_adam_step(self, p, grad, exp_avg, exp_avg_sq, step, beta1, beta2, lr, eps):
        """Standard dense Adam update step."""
        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Compute step size
        step_size = lr / bias_correction1
        
        # Compute denominator
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        # Update parameters using .data for consistency
        p.data.addcdiv_(exp_avg, denom, value=-step_size)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def compute_tile_mask_from_active_gaussians(
    active_gaussian_mask: torch.Tensor,
    means2D: torch.Tensor,
    radii: torch.Tensor,
    image_height: int,
    image_width: int,
    block_x: int = 16,
    block_y: int = 16
) -> torch.Tensor:
    """
    Compute tile mask from active Gaussian mask.
    
    根据活跃 Gaussian 的 2D 投影计算哪些 tile 需要渲染。
    
    Args:
        active_gaussian_mask: [P] bool tensor, True for active Gaussians
        means2D: [P, 2] float tensor, 2D positions of Gaussians
        radii: [P] int tensor, radii of Gaussians in pixels
        image_height: image height in pixels
        image_width: image width in pixels
        block_x: tile width (default 16, matching CUDA kernel)
        block_y: tile height (default 16, matching CUDA kernel)
    
    Returns:
        tile_mask: [num_tiles] bool tensor (flattened, row-major order)
                   where num_tiles = tiles_y * tiles_x
    """
    device = active_gaussian_mask.device
    
    # Calculate number of tiles
    tiles_x = (image_width + block_x - 1) // block_x
    tiles_y = (image_height + block_y - 1) // block_y
    num_tiles = tiles_x * tiles_y
    
    # Initialize tile mask as all False
    tile_mask = torch.zeros(num_tiles, dtype=torch.bool, device=device)
    
    # Get indices of active Gaussians
    active_indices = torch.where(active_gaussian_mask)[0]
    
    if len(active_indices) == 0:
        return tile_mask
    
    # Get 2D positions and radii of active Gaussians
    active_means2D = means2D[active_indices]  # [N_active, 2]
    active_radii = radii[active_indices]      # [N_active]
    
    # Filter out Gaussians with zero radius (not visible)
    visible_mask = active_radii > 0
    if not visible_mask.any():
        return tile_mask
    
    active_means2D = active_means2D[visible_mask]
    active_radii = active_radii[visible_mask]
    
    # Calculate bounding box for each active Gaussian
    # rect_min and rect_max in tile coordinates
    x_coords = active_means2D[:, 0]
    y_coords = active_means2D[:, 1]
    radii_float = active_radii.float()
    
    # Pixel coordinates of bounding box
    rect_min_x = (x_coords - radii_float).clamp(min=0)
    rect_min_y = (y_coords - radii_float).clamp(min=0)
    rect_max_x = (x_coords + radii_float + 1).clamp(max=image_width)
    rect_max_y = (y_coords + radii_float + 1).clamp(max=image_height)
    
    # Convert to tile coordinates
    tile_min_x = (rect_min_x / block_x).long()
    tile_min_y = (rect_min_y / block_y).long()
    tile_max_x = ((rect_max_x + block_x - 1) / block_x).long().clamp(max=tiles_x)
    tile_max_y = ((rect_max_y + block_y - 1) / block_y).long().clamp(max=tiles_y)
    
    # Mark tiles as active
    for i in range(len(active_means2D)):
        for ty in range(tile_min_y[i].item(), tile_max_y[i].item()):
            for tx in range(tile_min_x[i].item(), tile_max_x[i].item()):
                tile_id = ty * tiles_x + tx
                if 0 <= tile_id < num_tiles:
                    tile_mask[tile_id] = True
    
    return tile_mask

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        # CL-Splats: Prepare tile_mask for forward pass
        tile_mask = raster_settings.tile_mask
        if tile_mask is None:
            tile_mask = torch.empty(0, dtype=torch.bool, device=means3D.device)
        
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            # CL-Splats: tile_mask for local optimization
            tile_mask
        )

        # Invoke C++/CUDA rasterizer.
        # Strict protocol:
        #   7-tuple: (num_rendered, color, radii, geom, binning, img, invdepth)
        #   8-tuple: (num_rendered, color, radii, geom, binning, img, invdepth, mean_depth)
        # No heuristic remapping is applied.
        raster_out = _C.rasterize_gaussians(*args)

        def _meta(x):
            if torch.is_tensor(x):
                return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
            return f"{type(x).__name__}({x})"

        out_meta = ", ".join([f"[{i}]={_meta(v)}" for i, v in enumerate(raster_out)])
        if len(raster_out) == 7:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = raster_out
            mean_depth = None
        elif len(raster_out) == 8:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                invdepths,
                mean_depth,
            ) = raster_out
        else:
            raise RuntimeError(
                f"Unexpected rasterize_gaussians return size={len(raster_out)}; expected 7 or 8. "
                f"Raw outputs: {out_meta}"
            )

        if torch.is_tensor(num_rendered):
            if num_rendered.ndim == 0:
                num_rendered = int(num_rendered.item())
            else:
                raise RuntimeError(
                    f"Unexpected num_rendered tensor shape={tuple(num_rendered.shape)}"
                )
        elif isinstance(num_rendered, int):
            num_rendered = int(num_rendered)
        else:
            raise RuntimeError(
                f"Unexpected num_rendered type={type(num_rendered)}"
            )

        if not (
            torch.is_tensor(color)
            and color.ndim == 3
            and int(color.shape[0]) == 3
            and color.is_floating_point()
        ):
            raise RuntimeError(
                "Unexpected rasterizer output for color: "
                f"shape={getattr(color, 'shape', None)}, dtype={getattr(color, 'dtype', None)}. "
                "Expected return[1] to be float color [3, H, W]. "
                f"Raw outputs: {out_meta}"
            )
        if not (
            torch.is_tensor(radii)
            and radii.ndim == 1
            and radii.dtype == torch.int32
        ):
            raise RuntimeError(
                "Unexpected rasterizer output for radii: "
                f"shape={getattr(radii, 'shape', None)}, dtype={getattr(radii, 'dtype', None)}. "
                "Expected return[2] to be int32 radii [P]. "
                f"Raw outputs: {out_meta}"
            )
        if not (torch.is_tensor(geomBuffer) and geomBuffer.ndim == 1 and geomBuffer.dtype == torch.uint8):
            raise RuntimeError(
                "Unexpected rasterizer geomBuffer: "
                f"shape={getattr(geomBuffer, 'shape', None)}, dtype={getattr(geomBuffer, 'dtype', None)}"
            )
        if not (torch.is_tensor(binningBuffer) and binningBuffer.ndim == 1 and binningBuffer.dtype == torch.uint8):
            raise RuntimeError(
                "Unexpected rasterizer binningBuffer: "
                f"shape={getattr(binningBuffer, 'shape', None)}, dtype={getattr(binningBuffer, 'dtype', None)}"
            )
        if not (torch.is_tensor(imgBuffer) and imgBuffer.ndim == 1 and imgBuffer.dtype == torch.uint8):
            raise RuntimeError(
                "Unexpected rasterizer imgBuffer: "
                f"shape={getattr(imgBuffer, 'shape', None)}, dtype={getattr(imgBuffer, 'dtype', None)}"
            )

        if not (
            torch.is_tensor(invdepths)
            and invdepths.ndim == 3
            and invdepths.is_floating_point()
            and int(invdepths.shape[1]) == int(color.shape[1])
            and int(invdepths.shape[2]) == int(color.shape[2])
            and int(invdepths.shape[0]) in (0, 1)
        ):
            raise RuntimeError(
                "Unexpected rasterizer invdepths: "
                f"shape={getattr(invdepths, 'shape', None)}, dtype={getattr(invdepths, 'dtype', None)}"
            )

        if mean_depth is None:
            mean_depth = torch.empty(
                (0, int(color.shape[1]), int(color.shape[2])),
                device=color.device,
                dtype=color.dtype,
            )
        elif not (
            torch.is_tensor(mean_depth)
            and mean_depth.ndim == 3
            and mean_depth.is_floating_point()
            and int(mean_depth.shape[1]) == int(color.shape[1])
            and int(mean_depth.shape[2]) == int(color.shape[2])
            and int(mean_depth.shape[0]) in (0, 1)
        ):
            raise RuntimeError(
                "Unexpected rasterizer mean_depth: "
                f"shape={getattr(mean_depth, 'shape', None)}, dtype={getattr(mean_depth, 'dtype', None)}"
            )

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths, mean_depth

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_invdepth, grad_out_mean_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        if grad_out_mean_depth is not None and grad_out_mean_depth.numel() > 0:
            if torch.count_nonzero(grad_out_mean_depth).item() > 0:
                raise RuntimeError(
                    "Gradient through mean_depth is not implemented in this rasterizer build."
                )

        if grad_out_invdepth is None:
            grad_out_invdepth = torch.empty(
                (0,),
                device=grad_out_color.device,
                dtype=grad_out_color.dtype,
            )

        # CL-Splats: Prepare local optimization masks
        # If masks are not provided, use empty tensors (will be treated as no masking)
        active_gaussian_mask = raster_settings.active_gaussian_mask
        tile_mask = raster_settings.tile_mask
        if active_gaussian_mask is None:
            active_gaussian_mask = torch.empty(0, dtype=torch.bool, device=means3D.device)
        if tile_mask is None:
            tile_mask = torch.empty(0, dtype=torch.bool, device=means3D.device)

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_invdepth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug,
                # CL-Splats: Local optimization parameters
                active_gaussian_mask,
                tile_mask)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    # CL-Splats: Local optimization parameters
    active_gaussian_mask : torch.Tensor = None  # [P] bool tensor, True for active Gaussians
    tile_mask : torch.Tensor = None  # [num_tiles] bool tensor (flattened, row-major), True for active tiles

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
