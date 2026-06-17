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
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import geom_transform_points

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    temporal_render_mask = None
    if hasattr(pc, "get_temporal_render_mask"):
        temporal_render_mask = pc.get_temporal_render_mask()
        if temporal_render_mask is not None and int(temporal_render_mask.shape[0]) == int(visible_mask.shape[0]):
            visible_mask = torch.logical_and(visible_mask, temporal_render_mask.to(device=visible_mask.device, dtype=torch.bool))
    visible_anchor_ids = torch.where(visible_mask)[0]
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    if (
        is_training
        and getattr(pc, "temporal_phase3_active", False)
        and hasattr(pc, "get_temporal_local_mask")
    ):
        local_mask = pc.get_temporal_local_mask()
        if local_mask is not None and int(local_mask.shape[0]) == int(visible_mask.shape[0]):
            visible_local_mask = local_mask.to(device=visible_mask.device, dtype=torch.bool)[visible_mask]
            if visible_local_mask.numel() > 0:
                anchor_keep = visible_local_mask.unsqueeze(-1)
                offset_keep = visible_local_mask.reshape(-1, 1, 1)
                feat = torch.where(anchor_keep, feat, feat.detach())
                anchor = torch.where(anchor_keep, anchor, anchor.detach())
                grid_offsets = torch.where(offset_keep, grid_offsets, grid_offsets.detach())
                grid_scaling = torch.where(anchor_keep, grid_scaling, grid_scaling.detach())

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    visible_feature_input = feat
    if getattr(pc, "temporal_enabled", False) and hasattr(pc, "get_temporal_visible_latent"):
        temporal_feature_input = pc.get_temporal_visible_latent(visible_mask, visible_feat=feat)
        if temporal_feature_input is not None:
            visible_feature_input = temporal_feature_input

    cat_local_view = torch.cat([visible_feature_input, ob_view, ob_dist], dim=1) # [N, feature+3+1]
    cat_local_view_wodist = torch.cat([visible_feature_input, ob_view], dim=1) # [N, feature+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    anchor_ids = visible_anchor_ids.unsqueeze(1).expand(-1, pc.n_offsets).reshape(-1)
    offset_slot_ids = torch.arange(pc.n_offsets, device=anchor.device, dtype=torch.long).unsqueeze(0).expand(anchor.shape[0], -1).reshape(-1)
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    masked_anchor_ids = anchor_ids[mask]
    masked_offset_slot_ids = offset_slot_ids[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, visible_anchor_ids, masked_anchor_ids, masked_offset_slot_ids
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, return_depth=False, exact_importance_gt=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, visible_anchor_ids, anchor_ids, offset_slot_ids = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
        view_xyz_h = torch.cat([xyz, ones], dim=1)
        view_xyz = view_xyz_h @ viewpoint_camera.world_view_transform
        proj_xyz = geom_transform_points(xyz, viewpoint_camera.full_proj_transform)
        rendered_xy = torch.stack(
            [
                (proj_xyz[:, 0] + 1.0) * 0.5 * float(max(int(viewpoint_camera.image_width) - 1, 1)),
                (1.0 - (proj_xyz[:, 1] + 1.0) * 0.5) * float(max(int(viewpoint_camera.image_height) - 1, 1)),
            ],
            dim=-1,
        )
        rendered_depth = view_xyz[:, 2]

        out = {"render": rendered_image,
               "viewspace_points": screenspace_points,
               "visibility_filter" : radii > 0,
               "radii": radii,
               "selection_mask": mask,
               "neural_opacity": neural_opacity,
               "visible_anchor_ids": visible_anchor_ids,
               "scaling": scaling,
               "rendered_xyz": xyz,
               "rendered_xy": rendered_xy,
               "rendered_depth": rendered_depth,
               "rendered_color": color,
               "rendered_opacity": opacity,
               "rendered_anchor_ids": anchor_ids,
               "rendered_offset_slot_ids": offset_slot_ids,
               }
        if exact_importance_gt is not None:
            gt_color = exact_importance_gt.to(device=xyz.device, dtype=torch.float32)
            if gt_color.shape[0] > 3:
                gt_color = gt_color[:3]
            expected_shape = (
                3,
                int(viewpoint_camera.image_height),
                int(viewpoint_camera.image_width),
            )
            if tuple(gt_color.shape) != expected_shape:
                raise RuntimeError(
                    "exact_importance_gt shape mismatch: "
                    f"expected={expected_shape}, actual={tuple(gt_color.shape)}"
                )
            if not hasattr(rasterizer, "exact_importance"):
                raise RuntimeError(
                    "diff_gaussian_rasterization was built without exact_importance. "
                    "Rebuild Scaffold-GS-main copy/submodules/diff-gaussian-rasterization."
                )
            exact_score, exact_count, exact_radii = rasterizer.exact_importance(
                means3D=xyz,
                opacities=opacity,
                gt_color=gt_color.contiguous(),
                shs=None,
                colors_precomp=color,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None,
            )
            out["exact_importance_score"] = exact_score
            out["exact_importance_count"] = exact_count
            out["exact_importance_radii"] = exact_radii
        if return_depth:
            depth_map = torch.zeros(
                (1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)),
                dtype=xyz.dtype,
                device=xyz.device,
            )
            if xyz.numel() > 0:
                positive_depth = rendered_depth[torch.isfinite(rendered_depth) & (rendered_depth > 1e-6)]
                if positive_depth.numel() > 0:
                    depth_min = positive_depth.min()
                    depth_max = positive_depth.max()
                    depth_span = torch.clamp(depth_max - depth_min, min=1e-6)
                    depth_norm = ((rendered_depth - depth_min) / depth_span).unsqueeze(-1).repeat(1, 3)
                    depth_bg = torch.zeros_like(bg_color)
                    depth_raster_settings = GaussianRasterizationSettings(
                        image_height=int(viewpoint_camera.image_height),
                        image_width=int(viewpoint_camera.image_width),
                        tanfovx=tanfovx,
                        tanfovy=tanfovy,
                        bg=depth_bg,
                        scale_modifier=scaling_modifier,
                        viewmatrix=viewpoint_camera.world_view_transform,
                        projmatrix=viewpoint_camera.full_proj_transform,
                        sh_degree=1,
                        campos=viewpoint_camera.camera_center,
                        prefiltered=False,
                        debug=pipe.debug
                    )
                    depth_rasterizer = GaussianRasterizer(raster_settings=depth_raster_settings)
                    depth_render_rgb, _ = depth_rasterizer(
                        means3D=xyz,
                        means2D=screenspace_points.detach(),
                        shs=None,
                        colors_precomp=depth_norm,
                        opacities=opacity,
                        scales=scaling,
                        rotations=rot,
                        cov3D_precomp=None,
                    )
                    depth_map = depth_render_rgb[:1] * depth_span + depth_min
            out["depth"] = depth_map
        return out
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    visible_mask = radii_pure > 0
    if hasattr(pc, "get_temporal_render_mask"):
        temporal_render_mask = pc.get_temporal_render_mask()
        if temporal_render_mask is not None and int(temporal_render_mask.shape[0]) == int(visible_mask.shape[0]):
            visible_mask = torch.logical_and(visible_mask, temporal_render_mask.to(device=visible_mask.device, dtype=torch.bool))
    return visible_mask
