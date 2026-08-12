# Copied from https://github.com/graphdeco-inria/gaussian-splat-pytorch/blob/main/gaussian_model.py
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
import json
import os

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn

from ircgs.utils.general_utils import strip_symmetric, build_scaling_rotation
from ircgs.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from ircgs.utils.graphics_utils import BasicPointCloud
from ircgs.utils.sh_utils import RGB2SH
from ircgs.utils.system_utils import mkdir_p
from simple_knn._C import distCUDA2

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Keep exposure-related attributes always available across all restore paths.
        self._exposure = None
        self.exposure_mapping = {}
        self.pretrained_exposures = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def append_from_points(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        spatial_lr_scale: float = None,
        init_opacity: float = 0.1,
    ) -> int:
        """
        Append new Gaussians from xyz/rgb seed points.

        Args:
            xyz: [N, 3] point positions.
            rgb: [N, 3] colors in [0, 1].
            spatial_lr_scale: Optional scene extent override.
            init_opacity: Initial opacity value in [0, 1].

        Returns:
            Number of appended Gaussians.
        """
        if xyz is None or rgb is None or xyz.numel() == 0:
            return 0

        if spatial_lr_scale is not None:
            self.spatial_lr_scale = float(spatial_lr_scale)

        # Resolve target device from current tensors.
        if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0:
            device = self._xyz.device
        else:
            device = xyz.device

        xyz = xyz.to(device=device, dtype=torch.float32)
        rgb = rgb.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

        if xyz.dim() != 2 or xyz.shape[1] != 3:
            raise ValueError(f"append_from_points expects xyz [N,3], got {tuple(xyz.shape)}")
        if rgb.dim() != 2 or rgb.shape[1] < 3 or rgb.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"append_from_points expects rgb [N,3] aligned with xyz, got {tuple(rgb.shape)}"
            )
        if rgb.shape[1] > 3:
            rgb = rgb[:, :3]

        n_new = int(xyz.shape[0])
        if n_new == 0:
            return 0

        fused_color = RGB2SH(rgb)
        n_coeff = (self.max_sh_degree + 1) ** 2
        features = torch.zeros((n_new, 3, n_coeff), dtype=torch.float32, device=device)
        features[:, :3, 0] = fused_color

        new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()

        base_scale = max(float(self.spatial_lr_scale) * 0.005, 1e-4)
        new_scaling = torch.log(
            torch.full((n_new, 3), base_scale, dtype=torch.float32, device=device)
        )
        new_rotation = torch.zeros((n_new, 4), dtype=torch.float32, device=device)
        new_rotation[:, 0] = 1.0

        init_opacity = float(np.clip(init_opacity, 1e-4, 1.0 - 1e-4))
        new_opacity = self.inverse_opacity_activation(
            torch.full((n_new, 1), init_opacity, dtype=torch.float32, device=device)
        )

        if self.optimizer is not None:
            # Follow densification append path to keep optimizer state consistent.
            if not hasattr(self, "tmp_radii") or self.tmp_radii is None:
                self.tmp_radii = torch.zeros((self.get_xyz.shape[0]), device=device)
            new_tmp_radii = torch.zeros((n_new), device=device)
            self.densification_postfix(
                xyz,
                new_features_dc,
                new_features_rest,
                new_opacity,
                new_scaling,
                new_rotation,
                new_tmp_radii,
            )
            self.tmp_radii = None
            return n_new

        # Fallback path when optimizer is not yet initialized.
        def _cat_or_init(curr: torch.Tensor, ext: torch.Tensor) -> nn.Parameter:
            if curr.numel() == 0:
                return nn.Parameter(ext.requires_grad_(True))
            return nn.Parameter(torch.cat((curr.detach(), ext), dim=0).requires_grad_(True))

        self._xyz = _cat_or_init(self._xyz, xyz)
        self._features_dc = _cat_or_init(self._features_dc, new_features_dc)
        self._features_rest = _cat_or_init(self._features_rest, new_features_rest)
        self._opacity = _cat_or_init(self._opacity, new_opacity)
        self._scaling = _cat_or_init(self._scaling, new_scaling)
        self._rotation = _cat_or_init(self._rotation, new_rotation)

        n_total = self.get_xyz.shape[0]
        self.max_radii2D = torch.zeros((n_total), device=device)
        self.xyz_gradient_accum = torch.zeros((n_total, 1), device=device)
        self.denom = torch.zeros((n_total, 1), device=device)
        return n_new

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Ensure exposure bookkeeping exists on every call path.
        if not hasattr(self, "exposure_mapping") or self.exposure_mapping is None:
            self.exposure_mapping = {}
        if not hasattr(self, "pretrained_exposures"):
            self.pretrained_exposures = None

        # Initialize _exposure if not already set (e.g., when loading from PLY)
        if not hasattr(self, '_exposure') or self._exposure is None:
            # Create a default exposure parameter (identity transform)
            # This will be properly sized when cameras are available
            exposure = torch.eye(3, 4, device="cuda")[None]
            self._exposure = nn.Parameter(exposure.requires_grad_(True))
            self.exposure_mapping = {}

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, active_mask: torch.Tensor = None):
        """
        Reset opacity of Gaussians to 0.01.
        
        Args:
            active_mask: Optional boolean mask [N]. If provided, only reset opacity
                        for Gaussians where mask is True (Local Optimization mode).
                        If None, reset all Gaussians (standard 3DGS behavior).
        """
        if active_mask is not None:
            # Local Optimization: only reset opacity for active (changed) Gaussians
            if active_mask.shape[0] != self._opacity.shape[0]:
                # CRITICAL FIX for CL-Splats: 
                # If mask size doesn't match opacity size, DO NOT fallback to global reset!
                # This would destroy the inactive (frozen) part of the scene.
                # Just skip reset for this iteration.
                return

            opacities_new = self._opacity.clone()
            # Compute reset value (inverse sigmoid of 0.01)
            reset_value = self.inverse_opacity_activation(
                torch.tensor([[0.01]], device=self._opacity.device)
            )
            opacities_new[active_mask] = reset_value
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        else:
            # Standard behavior: reset all Gaussians
            opacities_new = self.inverse_opacity_activation(
                torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
            )
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # Initialize max_radii2D on CUDA
        self.max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")
        
        # Initialize exposure mapping and exposure parameter (will be updated when cameras are set)
        self.exposure_mapping = {}
        self.pretrained_exposures = None
        # _exposure will be initialized in set_exposure_for_cameras or training_setup

        self.active_sh_degree = self.max_sh_degree

    def set_exposure_for_cameras(self, cam_infos):
        """Set up exposure parameters for a list of cameras.
        
        This should be called after load_ply when cameras are available.
        
        Args:
            cam_infos: List of camera info objects with image_name attribute
        """
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, active_mask=None):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # tmp_radii may not exist when called from sphere pruning
        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]
            
        # CL-Splats: Synchronize active_mask if provided
        if active_mask is not None:
            active_mask = active_mask[valid_points_mask]
            return active_mask
        return None

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        active_mask=None,
        N=2,
        selected_pts_mask=None,
        scaling_override=None,
    ):
        n_init_points = self.get_xyz.shape[0]
        if selected_pts_mask is None:
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
            )
        else:
            selected_pts_mask = selected_pts_mask.to(device="cuda", dtype=torch.bool).reshape(-1)
            if selected_pts_mask.shape[0] != n_init_points:
                raise ValueError(
                    "selected_pts_mask has invalid length in densify_and_split: "
                    f"expected {n_init_points}, got {selected_pts_mask.shape[0]}"
                )
        
        # CL-Splats: Only split active Gaussians (protect inactive regions)
        if active_mask is not None and active_mask.shape[0] == selected_pts_mask.shape[0]:
            selected_pts_mask = torch.logical_and(selected_pts_mask, active_mask)

        if scaling_override is None:
            selected_scaling = self.get_scaling[selected_pts_mask]
        else:
            scaling_override = scaling_override.to(device="cuda", dtype=self.get_scaling.dtype)
            if scaling_override.ndim != 2 or scaling_override.shape[1] != 3:
                raise ValueError(
                    "scaling_override in densify_and_split must have shape [N, 3] "
                    "or [num_selected, 3]."
                )
            if scaling_override.shape[0] == n_init_points:
                selected_scaling = scaling_override[selected_pts_mask]
            elif scaling_override.shape[0] == int(selected_pts_mask.sum().item()):
                selected_scaling = scaling_override
            else:
                raise ValueError(
                    "scaling_override has invalid length in densify_and_split: "
                    f"expected {n_init_points} or {int(selected_pts_mask.sum().item())}, "
                    f"got {scaling_override.shape[0]}"
                )

        stds = selected_scaling.repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(selected_scaling.repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        
        if active_mask is not None:
            # Append new points to active mask (all new points are active)
            # New points count = N * selected_pts_mask.sum()
            new_count = N * selected_pts_mask.sum()
            new_actives = torch.ones(new_count, device="cuda", dtype=bool)
            active_mask = torch.cat([active_mask, new_actives])
            
            # Apply pruning to synced mask
            active_mask = self.prune_points(prune_filter, active_mask)
            return active_mask
        else:
            self.prune_points(prune_filter)
            return None

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        scene_extent,
        active_mask=None,
        selected_pts_mask=None,
    ):
        if selected_pts_mask is None:
            # Extract points that satisfy the gradient condition
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
            )
        else:
            selected_pts_mask = selected_pts_mask.to(device="cuda", dtype=torch.bool).reshape(-1)
            if selected_pts_mask.shape[0] != self.get_xyz.shape[0]:
                raise ValueError(
                    "selected_pts_mask has invalid length in densify_and_clone: "
                    f"expected {self.get_xyz.shape[0]}, got {selected_pts_mask.shape[0]}"
                )
        
        # CL-Splats: Only clone active Gaussians (protect inactive regions)
        if active_mask is not None and active_mask.shape[0] == selected_pts_mask.shape[0]:
            selected_pts_mask = torch.logical_and(selected_pts_mask, active_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        
        if active_mask is not None:
            # key: densification_postfix appends points at end
            # we need to append True to active_mask for these new points
            new_count = new_xyz.shape[0]
            new_actives = torch.ones(new_count, device="cuda", dtype=bool)
            active_mask = torch.cat([active_mask, new_actives])
            return active_mask
        return None

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, 
                          active_mask: torch.Tensor = None):
        """
        Densify and prune Gaussians based on gradient and opacity thresholds.
        
        Args:
            max_grad: Gradient threshold for densification
            min_opacity: Minimum opacity threshold for pruning
            extent: Scene extent for size-based pruning
            max_screen_size: Maximum screen-space size threshold
            radii: Screen-space radii of Gaussians
            active_mask: Optional boolean mask [N]. If provided (t>0), only prune
                        active Gaussians by opacity. Inactive Gaussians are protected.
                        
        Returns:
            Updated active_mask if provided, None otherwise
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        
        # Track original count for mask synchronization
        n_before_clone = self.get_xyz.shape[0]
        
        # CL-Splats: Pass active_mask to clone to protect inactive regions
        # CL-Splats: Pass active_mask to clone and capture updated mask
        # Note: densify_and_clone appends new points (actives), so it just extends the mask
        updated_mask = self.densify_and_clone(grads, max_grad, extent, active_mask=active_mask)
        if updated_mask is not None:
            active_mask = updated_mask
        
        # CL-Splats: Pass active_mask to split and capture updated mask
        # Note: densify_and_split removes points and adds new ones, so it MUST manage the mask internally
        updated_mask = self.densify_and_split(grads, max_grad, extent, active_mask=active_mask)
        if updated_mask is not None:
            active_mask = updated_mask

        # Opacity-based pruning
        opacity_prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # CL-Splats Local Optimization: only prune active Gaussians by opacity
        # Inactive (frozen) Gaussians should never be pruned by opacity
        if active_mask is not None and active_mask.shape[0] == opacity_prune_mask.shape[0]:
            # Only apply opacity pruning to active Gaussians
            # inactive Gaussians: opacity_prune_mask is forced to False
            opacity_prune_mask = torch.logical_and(opacity_prune_mask, active_mask)
        
        # Size-based pruning (applies to all Gaussians for scene quality)
        prune_mask = opacity_prune_mask
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            size_prune_mask = torch.logical_or(big_points_vs, big_points_ws)
            
            # Size pruning also only applies to active Gaussians in CL mode
            if active_mask is not None and active_mask.shape[0] == size_prune_mask.shape[0]:
                size_prune_mask = torch.logical_and(size_prune_mask, active_mask)
            
            prune_mask = torch.logical_or(prune_mask, size_prune_mask)
        
        # CL-Splats: Synchronize active_mask during final pruning
        if active_mask is not None:
            active_mask = self.prune_points(prune_mask, active_mask)
        else:
            self.prune_points(prune_mask)
            
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
        
        return active_mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter, active_mask=None):
        # CL-Splats: Only accumulate gradients for active Gaussians (protect inactive regions)
        if active_mask is not None and active_mask.shape[0] == update_filter.shape[0]:
            update_filter = torch.logical_and(update_filter, active_mask)
        
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def freeze_inactive_gaussians(self, active_mask: torch.Tensor) -> None:
        """
        Freeze optimizer state (momentum) for inactive Gaussians.
        
        This ensures that inactive Gaussians are not updated by the optimizer,
        even if gradients accidentally flow to them. The momentum (exp_avg and
        exp_avg_sq) for inactive Gaussians is zeroed out.
        
        Args:
            active_mask: Boolean tensor [N] where True = active (can be updated),
                        False = inactive (frozen, should not be updated)
        """
        if active_mask is None:
            return
        
        inactive_mask = ~active_mask
        if not inactive_mask.any():
            return
        
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # Zero out momentum for inactive Gaussians
                # This prevents any accumulated momentum from affecting frozen Gaussians
                if 'exp_avg' in stored_state:
                    stored_state['exp_avg'][inactive_mask] = 0.0
                if 'exp_avg_sq' in stored_state:
                    stored_state['exp_avg_sq'][inactive_mask] = 0.0

    def zero_gradients_for_inactive(self, active_mask: torch.Tensor) -> None:
        """
        Zero out gradients for inactive Gaussians before optimizer step.
        
        This is a safety measure to ensure inactive Gaussians receive no updates.
        Should be called after loss.backward() and before optimizer.step().
        
        Args:
            active_mask: Boolean tensor [N] where True = active, False = inactive
        """
        if active_mask is None:
            return
        
        inactive_mask = ~active_mask
        if not inactive_mask.any():
            return
        
        # Zero gradients for all parameters of inactive Gaussians
        if self._xyz.grad is not None:
            self._xyz.grad[inactive_mask] = 0.0
        if self._features_dc.grad is not None:
            self._features_dc.grad[inactive_mask] = 0.0
        if self._features_rest.grad is not None:
            self._features_rest.grad[inactive_mask] = 0.0
        if self._opacity.grad is not None:
            self._opacity.grad[inactive_mask] = 0.0
        if self._scaling.grad is not None:
            self._scaling.grad[inactive_mask] = 0.0
        if self._rotation.grad is not None:
            self._rotation.grad[inactive_mask] = 0.0

    def scale_gradients(self, mask: torch.Tensor, scale: float) -> None:
        """
        Scale gradients for selected Gaussians by a multiplicative factor.

        Args:
            mask: Boolean tensor [N], True where gradients should be scaled.
            scale: Multiplicative factor in [0, +inf). Typical weak-training values: 0.05~0.2.
        """
        if mask is None:
            return
        s = float(scale)
        if s == 1.0:
            return
        if s < 0.0:
            s = 0.0
        if not mask.any():
            return

        if self._xyz.grad is not None:
            self._xyz.grad[mask] *= s
        if self._features_dc.grad is not None:
            self._features_dc.grad[mask] *= s
        if self._features_rest.grad is not None:
            self._features_rest.grad[mask] *= s
        if self._opacity.grad is not None:
            self._opacity.grad[mask] *= s
        if self._scaling.grad is not None:
            self._scaling.grad[mask] *= s
        if self._rotation.grad is not None:
            self._rotation.grad[mask] *= s
