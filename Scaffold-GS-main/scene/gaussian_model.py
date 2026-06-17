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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from scene.temporal_state import TemporalLatentState

    
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

    def _time_key(self, time_step: int) -> str:
        return str(int(time_step))

    def _active_temporal_dim(self, time_step: int | None = None) -> int:
        if not self.temporal_enabled:
            return 0
        target_step = self.current_time_step if time_step is None else int(time_step)
        return min(self.temporal_latent_dim, max(target_step + 1, 0) * self.temporal_block_dim)

    def _history_temporal_dim(self, time_step: int | None = None) -> int:
        if not self.temporal_enabled:
            return 0
        target_step = self.current_time_step if time_step is None else int(time_step)
        return min(self.temporal_latent_dim, max(target_step, 0) * self.temporal_block_dim)

    def _property_tail_dim(self, kind: str) -> int:
        if kind == "opacity":
            return 3 + self.opacity_dist_dim
        if kind == "cov":
            return 3 + self.cov_dist_dim
        if kind == "color":
            return 3 + self.color_dist_dim + self.appearance_dim
        raise KeyError(f"Unknown property MLP kind: {kind}")

    def _property_output_dim(self, kind: str) -> int:
        if kind == "opacity":
            return self.n_offsets
        if kind == "cov":
            return 7 * self.n_offsets
        if kind == "color":
            return 3 * self.n_offsets
        raise KeyError(f"Unknown property MLP kind: {kind}")

    def _build_property_mlp(
        self,
        *,
        temporal_input_dim: int,
        kind: str,
    ):
        feature_input_dim = int(temporal_input_dim) if int(temporal_input_dim) > 0 else self.feat_dim
        input_dim = feature_input_dim + self._property_tail_dim(kind)
        output_dim = self._property_output_dim(kind)
        layers = [
            nn.Linear(input_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, output_dim),
        ]
        if kind == "opacity":
            layers.append(nn.Tanh())
        elif kind == "color":
            layers.append(nn.Sigmoid())
        mlp = nn.Sequential(*layers).cuda()
        if temporal_input_dim > 0:
            with torch.no_grad():
                # The temporal feature block already includes the base anchor feature,
                # so only the appended temporal channels should start from zero.
                temporal_only_dim = max(int(temporal_input_dim) - self.feat_dim, 0)
                if temporal_only_dim > 0:
                    start = self.feat_dim
                    end = self.feat_dim + temporal_only_dim
                    mlp[0].weight[:, start:end].zero_()
        return mlp

    def _clone_property_mlp_from_prev(
        self,
        prev_mlp,
        *,
        prev_temporal_dim: int,
        new_temporal_dim: int,
        kind: str,
    ):
        new_mlp = self._build_property_mlp(
            temporal_input_dim=new_temporal_dim,
            kind=kind,
        )
        with torch.no_grad():
            new_mlp[0].bias.copy_(prev_mlp[0].bias)
            new_mlp[2].weight.copy_(prev_mlp[2].weight)
            new_mlp[2].bias.copy_(prev_mlp[2].bias)

            feat_dim = self.feat_dim
            old_tail_dim = self._property_tail_dim(kind)
            new_tail_dim = old_tail_dim
            prev_feature_input_dim = int(prev_temporal_dim) if int(prev_temporal_dim) > 0 else feat_dim
            new_feature_input_dim = int(new_temporal_dim) if int(new_temporal_dim) > 0 else feat_dim
            prev_temporal_only_dim = max(prev_feature_input_dim - feat_dim, 0)
            new_temporal_only_dim = max(new_feature_input_dim - feat_dim, 0)

            new_mlp[0].weight[:, :feat_dim].copy_(prev_mlp[0].weight[:, :feat_dim])
            old_tail_start = prev_feature_input_dim
            new_tail_start = new_feature_input_dim
            copy_temporal_dim = min(prev_temporal_only_dim, new_temporal_only_dim)
            if copy_temporal_dim > 0:
                new_mlp[0].weight[:, feat_dim:feat_dim + copy_temporal_dim].copy_(
                    prev_mlp[0].weight[:, feat_dim:feat_dim + copy_temporal_dim]
                )
            if new_temporal_only_dim > copy_temporal_dim:
                new_mlp[0].weight[:, feat_dim + copy_temporal_dim:feat_dim + new_temporal_only_dim].zero_()
            new_mlp[0].weight[:, new_tail_start:new_tail_start + new_tail_dim].copy_(
                prev_mlp[0].weight[:, old_tail_start:old_tail_start + old_tail_dim]
            )
        return new_mlp

    def _build_history_mlp_for_time(self, time_step: int):
        input_dim = self.temporal_block_dim
        if input_dim <= 0:
            return None
        output_dim = self.feat_dim
        hidden_dim = max(self.feat_dim, output_dim)
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        ).cuda()
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _get_base_property_mlp(self, kind: str):
        if kind == "opacity":
            return self.mlp_opacity
        if kind == "cov":
            return self.mlp_cov
        if kind == "color":
            return self.mlp_color
        raise KeyError(f"Unknown property MLP kind: {kind}")

    def _get_temporal_property_dict(self, kind: str):
        if kind == "opacity":
            return self.temporal_opacity_mlps
        if kind == "cov":
            return self.temporal_cov_mlps
        if kind == "color":
            return self.temporal_color_mlps
        raise KeyError(f"Unknown property MLP kind: {kind}")

    def _get_current_history_mlp(self):
        key = self._time_key(self.current_time_step)
        if key in self.temporal_history_mlps:
            return self.temporal_history_mlps[key]
        return None

    def _get_current_property_mlp(self, kind: str):
        if int(self.current_time_step) <= 0:
            return self._get_base_property_mlp(kind)
        module_dict = self._get_temporal_property_dict(kind)
        key = self._time_key(self.current_time_step)
        if key in module_dict:
            return module_dict[key]
        return self._get_base_property_mlp(kind)

    def _set_module_requires_grad(self, module, enabled: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(bool(enabled))

    def _ensure_temporal_modules_for_time(self, time_step: int):
        time_step = int(time_step)
        if not self.temporal_enabled or time_step <= 0:
            return

        key = self._time_key(time_step)
        prev_time_step = max(time_step - 1, 0)
        current_temporal_dim = self.temporal_input_dim

        if key not in self.temporal_history_mlps:
            history_mlp = self._build_history_mlp_for_time(time_step)
            if history_mlp is not None:
                self.temporal_history_mlps[key] = history_mlp

        for kind in ("opacity", "cov", "color"):
            module_dict = self._get_temporal_property_dict(kind)
            if key in module_dict:
                continue
            if prev_time_step <= 0:
                prev_mlp = self._get_base_property_mlp(kind)
                prev_temporal_dim = 0
            else:
                prev_key = self._time_key(prev_time_step)
                if prev_key in module_dict:
                    prev_mlp = module_dict[prev_key]
                else:
                    prev_mlp = self._get_base_property_mlp(kind)
                prev_temporal_dim = self.temporal_input_dim
            module_dict[key] = self._clone_property_mlp_from_prev(
                prev_mlp,
                prev_temporal_dim=prev_temporal_dim,
                new_temporal_dim=current_temporal_dim,
                kind=kind,
            )

    def _zero_temporal_input_slice(self, linear_layer: nn.Linear):
        if self.temporal_input_dim <= 0:
            return
        with torch.no_grad():
            start = self.feat_dim
            end = self.temporal_input_dim
            linear_layer.weight[:, start:end].zero_()

    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 temporal_latent_dim: int = 6,
                 temporal_num_times: int = 1,
                  ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.temporal_block_dim = max(int(temporal_latent_dim), 1)
        self.temporal_num_times = max(int(temporal_num_times), 1)
        self.temporal_latent_dim = int(self.temporal_block_dim * self.temporal_num_times)
        self.temporal_enabled = self.temporal_num_times > 1
        self.enable_temporal_lora = self.temporal_enabled
        self.enable_color_lora = False
        self.enable_opacity_lora = False
        self.enable_cov_lora = False
        self.enable_bank_lora = False

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.temporal_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.base_temporal_state = None
        self.temporal_state = None
        self.mlp_temporal_bank = None
        self.temporal_bank_mlp = None
        self.temporal_opacity_mlp = None
        self.temporal_cov_mlp = None
        self.temporal_color_mlp = None
        self.current_time_step = 0
        self.temporal_anchor_birth_timestep = None
        self.temporal_anchor_death_timestep = None
        self.temporal_opacity_accum = None
        self.temporal_opacity_count = None
        self.temporal_last_opacity_max = None
        self.temporal_local_mask = None
        self.temporal_local_parent_ids = None
        self.temporal_anomaly_score = None
        self.temporal_anomaly_count = None
        self.temporal_anomaly_candidate_mask = None
        self.temporal_split_suppression_mask = None
        self.temporal_attribute_grad_mask = None
        self.temporal_split_event_count = 0
        self.temporal_death_event_count = 0
        self.temporal_suppression_event_count = 0
        self.temporal_phase3_active = False
        self.temporal_local_growth_active = False
        self.temporal_phase3_start_iter = -1
        self.temporal_base_anchor_count = 0
        self.N_base_anchors = 0
        self.temporal_compact_time_step = None
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.temporal_input_dim = (self.feat_dim + self.temporal_block_dim) if self.temporal_enabled else 0
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        # Base heads always operate on pure anchor features.
        self.mlp_opacity = self._build_property_mlp(
            temporal_input_dim=0,
            kind="opacity",
        )
        self.mlp_cov = self._build_property_mlp(
            temporal_input_dim=0,
            kind="cov",
        )
        self.mlp_color = self._build_property_mlp(
            temporal_input_dim=0,
            kind="color",
        )
        self.temporal_history_mlps = nn.ModuleDict()
        self.temporal_opacity_mlps = nn.ModuleDict()
        self.temporal_cov_mlps = nn.ModuleDict()
        self.temporal_color_mlps = nn.ModuleDict()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.temporal_history_mlps.eval()
        self.temporal_opacity_mlps.eval()
        self.temporal_cov_mlps.eval()
        self.temporal_color_mlps.eval()
        if self.temporal_state is not None:
            self.temporal_state.eval()
        if self.base_temporal_state is not None:
            self.base_temporal_state.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.temporal_history_mlps.train()
        self.temporal_opacity_mlps.train()
        self.temporal_cov_mlps.train()
        self.temporal_color_mlps.train()
        if self.temporal_state is not None:
            self.temporal_state.train()
        if self.base_temporal_state is not None:
            self.base_temporal_state.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_temporal_bank_mlp(self):
        return self._get_current_history_mlp()
    
    @property
    def get_opacity_mlp(self):
        return self._get_current_property_mlp("opacity")
    
    @property
    def get_cov_mlp(self):
        return self._get_current_property_mlp("cov")

    @property
    def get_color_mlp(self):
        return self._get_current_property_mlp("color")

    @property
    def get_base_color_mlp(self):
        return self.mlp_color

    @property
    def get_base_temporal_bank_mlp(self):
        return self.temporal_history_mlps

    @property
    def has_temporal_adaptation(self):
        return self.temporal_state is not None or self.base_temporal_state is not None

    def _build_temporal_state_from_tensor(self, latent_tensor: torch.Tensor):
        state = TemporalLatentState(
            latent_tensor.shape[0],
            latent_tensor.shape[1],
            num_times=self.temporal_num_times,
            chunk_dim=self.temporal_block_dim,
        ).cuda()
        state.latent = nn.Parameter(latent_tensor.requires_grad_(True))
        state.set_active_time_step(int(self.current_time_step))
        return state

    def _state_active_dim(self, state, time_step: int | None = None) -> int:
        target_step = self.current_time_step if time_step is None else int(time_step)
        if hasattr(state, "get_active_dim"):
            return int(state.get_active_dim(target_step))
        chunk_dim = int(getattr(state, "chunk_dim", self.temporal_block_dim))
        latent_dim = int(getattr(state, "latent_dim", self.temporal_latent_dim))
        return min(latent_dim, max(target_step + 1, 0) * chunk_dim)

    def _state_history_dim(self, state, time_step: int | None = None) -> int:
        target_step = self.current_time_step if time_step is None else int(time_step)
        if hasattr(state, "get_history_dim"):
            return int(state.get_history_dim(target_step))
        chunk_dim = int(getattr(state, "chunk_dim", self.temporal_block_dim))
        latent_dim = int(getattr(state, "latent_dim", self.temporal_latent_dim))
        return min(latent_dim, max(target_step, 0) * chunk_dim)

    def _state_current_block_slice(self, state, time_step: int | None = None) -> tuple[int, int]:
        target_step = self.current_time_step if time_step is None else int(time_step)
        latent = getattr(state, "latent", None)
        if isinstance(latent, torch.Tensor):
            latent_width = int(latent.shape[1]) if latent.ndim == 2 else 0
            block_dim = int(getattr(state, "chunk_dim", self.temporal_block_dim))
            if latent_width <= block_dim:
                return 0, int(latent_width)
        if hasattr(state, "current_block_slice"):
            start, end = state.current_block_slice(target_step)
            return int(start), int(end)
        chunk_dim = int(getattr(state, "chunk_dim", self.temporal_block_dim))
        latent_dim = int(getattr(state, "latent_dim", self.temporal_latent_dim))
        start = max(0, target_step) * chunk_dim
        end = min(latent_dim, start + chunk_dim)
        return int(start), int(end)

    def _pack_sparse_latent_rows(self, latent_tensor: torch.Tensor, active_mask=None):
        if not isinstance(latent_tensor, torch.Tensor):
            return latent_tensor
        latent_cpu = latent_tensor.detach().cpu()
        if latent_cpu.ndim != 2:
            return latent_cpu
        if active_mask is None:
            return latent_cpu
        active_mask = active_mask.detach().reshape(-1).to(
            device=latent_tensor.device,
            dtype=torch.bool,
        )
        if int(active_mask.shape[0]) != int(latent_cpu.shape[0]):
            return latent_cpu
        if bool(active_mask.all().item()):
            return latent_cpu
        active_anchor_ids = torch.where(active_mask)[0].to(dtype=torch.long)
        return {
            "storage": "sparse_v1",
            "num_anchors": int(latent_cpu.shape[0]),
            "latent_dim": int(latent_cpu.shape[1]),
            "active_anchor_ids": active_anchor_ids.cpu(),
            "values": latent_cpu[active_anchor_ids.cpu()],
        }

    def _unpack_sparse_latent_rows(
        self,
        latent_payload,
        *,
        expected_num_anchors: int,
        expected_latent_dim: int,
        device,
        dtype,
    ):
        if latent_payload is None:
            return None
        if isinstance(latent_payload, dict):
            if latent_payload.get("storage", None) == "sparse_v1":
                num_anchors = int(latent_payload.get("num_anchors", expected_num_anchors))
                latent_dim = int(latent_payload.get("latent_dim", expected_latent_dim))
                if num_anchors != int(expected_num_anchors) or latent_dim != int(expected_latent_dim):
                    return None
                active_anchor_ids = latent_payload.get("active_anchor_ids", None)
                values = latent_payload.get("values", None)
                if not isinstance(active_anchor_ids, torch.Tensor) or not isinstance(values, torch.Tensor):
                    return None
                active_anchor_ids = active_anchor_ids.to(device=device, dtype=torch.long).reshape(-1)
                values = values.to(device=device, dtype=dtype)
                if (
                    values.ndim != 2
                    or int(values.shape[0]) != int(active_anchor_ids.shape[0])
                    or int(values.shape[1]) != int(expected_latent_dim)
                ):
                    return None
                valid_ids = (active_anchor_ids >= 0) & (active_anchor_ids < int(expected_num_anchors))
                if not bool(valid_ids.all().item()):
                    active_anchor_ids = active_anchor_ids[valid_ids]
                    values = values[valid_ids]
                dense = torch.zeros(
                    (int(expected_num_anchors), int(expected_latent_dim)),
                    device=device,
                    dtype=dtype,
                )
                if active_anchor_ids.numel() > 0:
                    dense[active_anchor_ids] = values
                return dense
            if "latent" in latent_payload:
                latent_payload = latent_payload.get("latent", None)
        if not isinstance(latent_payload, torch.Tensor):
            return None
        latent_tensor = latent_payload.to(device=device, dtype=dtype)
        if (
            latent_tensor.ndim != 2
            or int(latent_tensor.shape[0]) != int(expected_num_anchors)
            or int(latent_tensor.shape[1]) != int(expected_latent_dim)
        ):
            return None
        return latent_tensor

    def serialize_temporal_checkpoint_payload(self, payload):
        if not isinstance(payload, dict):
            return payload
        serialized = dict(payload)
        serialized.pop("latent", None)
        serialized.pop("latent_block", None)
        serialized.pop("latent_block_dim", None)
        return serialized

    def deserialize_temporal_checkpoint_payload(self, payload, num_anchors: int = None):
        if not isinstance(payload, dict):
            return payload
        deserialized = dict(payload)
        expected_num_anchors = int(self.get_anchor.shape[0] if num_anchors is None else num_anchors)
        latent_block = deserialized.get("latent_block", None)
        if latent_block is not None:
            expected_block_dim = int(deserialized.get("latent_block_dim", self.temporal_block_dim))
            latent_tensor = self._unpack_sparse_latent_rows(
                latent_block,
                expected_num_anchors=expected_num_anchors,
                expected_latent_dim=expected_block_dim,
                device="cpu",
                dtype=torch.float32,
            )
            if latent_tensor is not None:
                deserialized["latent_block"] = latent_tensor.cpu()
        return deserialized

    def _replace_or_add_optimizer_group(self, name: str, parameter: nn.Parameter, lr: float):
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)
            if stored_state is not None:
                del self.optimizer.state[old_param]
            group["params"][0] = parameter
            if stored_state is not None:
                exp_avg = stored_state.get("exp_avg", None)
                exp_avg_sq = stored_state.get("exp_avg_sq", None)
                if exp_avg is not None and tuple(exp_avg.shape) != tuple(parameter.shape):
                    exp_avg = torch.zeros_like(parameter)
                if exp_avg_sq is not None and tuple(exp_avg_sq.shape) != tuple(parameter.shape):
                    exp_avg_sq = torch.zeros_like(parameter)
                stored_state["exp_avg"] = exp_avg if exp_avg is not None else torch.zeros_like(parameter)
                stored_state["exp_avg_sq"] = exp_avg_sq if exp_avg_sq is not None else torch.zeros_like(parameter)
                self.optimizer.state[parameter] = stored_state
            group["lr"] = lr
            return
        self.optimizer.add_param_group({"params": [parameter], "lr": lr, "name": name})

    def _append_optimizer_group_rows(self, name: str, extension_tensor: torch.Tensor):
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)
            new_param = nn.Parameter(torch.cat((old_param, extension_tensor), dim=0).requires_grad_(True))
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            return

    def _prune_optimizer_group_rows(self, name: str, mask: torch.Tensor):
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)
            new_param = nn.Parameter(old_param[mask].requires_grad_(True))
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            return

    def setup_base_temporal_state(self, training_args):
        if not self.temporal_enabled:
            return
        num_anchors = int(self.get_anchor.shape[0])
        block_dim = int(self.temporal_block_dim)
        needs_init = (
            self.base_temporal_state is None
            or self.base_temporal_state.num_anchors != num_anchors
            or getattr(self.base_temporal_state, "latent", None) is None
            or int(getattr(self.base_temporal_state, "latent").shape[1]) != block_dim
            or int(getattr(self, "temporal_compact_time_step", -1)) != int(self.current_time_step)
        )
        if needs_init:
            latent = None
            old_state = self.base_temporal_state
            old_latent = getattr(old_state, "latent", None)
            if (
                isinstance(old_latent, torch.Tensor)
                and old_latent.ndim == 2
                and int(old_latent.shape[0]) == num_anchors
            ):
                old_width = int(old_latent.shape[1])
                if old_width == block_dim:
                    old_time = getattr(self, "temporal_compact_time_step", None)
                    if old_time is None or int(old_time) == int(self.current_time_step):
                        latent = old_latent.detach().clone()
                elif old_width > block_dim:
                    old_block_dim = int(getattr(old_state, "chunk_dim", block_dim))
                    start = max(0, int(self.current_time_step)) * old_block_dim
                    end = min(old_width, start + block_dim)
                    if end > start and int(end - start) == block_dim:
                        latent = old_latent.detach()[:, start:end].clone()
            if latent is None:
                latent = 0.01 * torch.randn(
                    (num_anchors, block_dim),
                    device=self.get_anchor.device,
                    dtype=self.get_anchor.dtype,
                )
            self.base_temporal_state = self._build_temporal_state_from_tensor(latent)
            self.temporal_compact_time_step = int(self.current_time_step)
        self.base_temporal_state.set_layout(
            num_times=self.temporal_num_times,
            chunk_dim=self.temporal_block_dim,
        )
        self.base_temporal_state.set_active_time_step(int(self.current_time_step))
        self._replace_or_add_optimizer_group(
            "base_temporal_latent",
            self.base_temporal_state.latent,
            float(training_args.latent_lr),
        )

    def _append_base_temporal_latent(self, num_new: int):
        if self.base_temporal_state is None or num_new <= 0:
            return
        has_group = any(
            group.get("name", "") == "base_temporal_latent"
            for group in (self.optimizer.param_groups if self.optimizer is not None else [])
        )
        if not has_group:
            raise RuntimeError(
                "base_temporal_latent optimizer group is missing; call setup_base_temporal_state() before temporal clone/prune."
            )
        extension = 0.01 * torch.randn(
            (int(num_new), int(self.temporal_block_dim)),
            device=self.base_temporal_state.latent.device,
            dtype=self.base_temporal_state.latent.dtype,
        )
        self._append_optimizer_group_rows("base_temporal_latent", extension)
        for group in self.optimizer.param_groups:
            if group["name"] == "base_temporal_latent":
                self.base_temporal_state.latent = group["params"][0]
                self.base_temporal_state.num_anchors = int(self.base_temporal_state.latent.shape[0])
                break

    def _prune_base_temporal_latent(self, valid_mask: torch.Tensor):
        if self.base_temporal_state is None:
            return
        has_group = any(
            group.get("name", "") == "base_temporal_latent"
            for group in (self.optimizer.param_groups if self.optimizer is not None else [])
        )
        if not has_group:
            raise RuntimeError(
                "base_temporal_latent optimizer group is missing; call setup_base_temporal_state() before temporal clone/prune."
            )
        self._prune_optimizer_group_rows("base_temporal_latent", valid_mask)
        for group in self.optimizer.param_groups:
            if group["name"] == "base_temporal_latent":
                self.base_temporal_state.latent = group["params"][0]
                self.base_temporal_state.num_anchors = int(self.base_temporal_state.latent.shape[0])
                break
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def setup_temporal_adaptation(self, training_args, time_step: int = 1):
        self.current_time_step = int(time_step)
        self.temporal_phase3_active = False
        self.temporal_local_growth_active = False
        self.temporal_phase3_start_iter = -1
        self.temporal_base_anchor_count = int(self.get_anchor.shape[0])
        self.N_base_anchors = int(self.temporal_base_anchor_count)

        self.setup_base_temporal_state(training_args)
        if self.base_temporal_state is not None:
            self.base_temporal_state.set_layout(
                num_times=self.temporal_num_times,
                chunk_dim=self.temporal_block_dim,
            )
            self.base_temporal_state.set_active_time_step(int(self.current_time_step))

        self._ensure_temporal_modules_for_time(int(self.current_time_step))

        for module in [self.mlp_opacity, self.mlp_cov, self.mlp_color]:
            self._set_module_requires_grad(module, False)
        if self.use_feat_bank:
            self._set_module_requires_grad(self.mlp_feature_bank, False)
        for module in self.temporal_history_mlps.values():
            self._set_module_requires_grad(module, False)
        for module in self.temporal_opacity_mlps.values():
            self._set_module_requires_grad(module, False)
        for module in self.temporal_cov_mlps.values():
            self._set_module_requires_grad(module, False)
        for module in self.temporal_color_mlps.values():
            self._set_module_requires_grad(module, False)

        for tensor in [self._anchor, self._offset, self._anchor_feat, self._opacity, self._scaling, self._rotation]:
            if isinstance(tensor, nn.Parameter):
                tensor.requires_grad_(False)

        if self.embedding_appearance is not None:
            for param in self.embedding_appearance.parameters():
                param.requires_grad_(False)
        if self.base_temporal_state is not None:
            for param in self.base_temporal_state.parameters():
                param.requires_grad_(True)

        self.temporal_state = None
        self.temporal_bank_mlp = self._get_current_history_mlp()
        self.temporal_opacity_mlp = self._get_current_property_mlp("opacity")
        self.temporal_cov_mlp = self._get_current_property_mlp("cov")
        self.temporal_color_mlp = self._get_current_property_mlp("color")
        if self.temporal_bank_mlp is not None:
            self._set_module_requires_grad(self.temporal_bank_mlp, True)
        for kind in ("opacity", "cov", "color"):
            self._set_module_requires_grad(self._get_current_property_mlp(kind), True)

        params = []
        if self.base_temporal_state is not None:
            params.append(
                {"params": self.base_temporal_state.parameters(), "lr": training_args.latent_lr, "name": "temporal_latent"}
            )
        if self.temporal_bank_mlp is not None:
            params.append(
                {"params": self.temporal_bank_mlp.parameters(), "lr": training_args.mlp_temporal_bank_lr_init, "name": "temporal_history_mlp"}
            )
        params.append(
            {"params": self.get_opacity_mlp.parameters(), "lr": training_args.mlp_opacity_lr_init, "name": "temporal_opacity_mlp"}
        )
        params.append(
            {"params": self.get_cov_mlp.parameters(), "lr": training_args.mlp_cov_lr_init, "name": "temporal_cov_mlp"}
        )
        params.append(
            {"params": self.get_color_mlp.parameters(), "lr": training_args.mlp_color_lr_init, "name": "temporal_color_mlp"}
        )
        self.temporal_optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15) if params else None
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get("name", "") == "base_temporal_latent":
                    group["lr"] = 0.0

        num_anchors = int(self.get_anchor.shape[0])
        prev_birth_timestep = self.temporal_anchor_birth_timestep
        prev_death_timestep = self.temporal_anchor_death_timestep
        if not isinstance(prev_birth_timestep, torch.Tensor):
            prev_birth_timestep = None
        if not isinstance(prev_death_timestep, torch.Tensor):
            prev_death_timestep = None
        self.temporal_anchor_birth_timestep = torch.zeros((num_anchors,), dtype=torch.long, device="cuda")
        self.temporal_anchor_death_timestep = torch.full(
            (num_anchors,),
            2**31 - 1,
            dtype=torch.long,
            device="cuda",
        )
        if prev_birth_timestep is not None:
            copy_rows = min(int(prev_birth_timestep.numel()), num_anchors)
            if copy_rows > 0:
                self.temporal_anchor_birth_timestep[:copy_rows] = prev_birth_timestep[:copy_rows].to(
                    device="cuda",
                    dtype=torch.long,
                )
        if prev_death_timestep is not None:
            copy_rows = min(int(prev_death_timestep.numel()), num_anchors)
            if copy_rows > 0:
                self.temporal_anchor_death_timestep[:copy_rows] = prev_death_timestep[:copy_rows].to(
                    device="cuda",
                    dtype=torch.long,
                )
        self.temporal_opacity_accum = torch.zeros((num_anchors, 1), dtype=torch.float32, device="cuda")
        self.temporal_opacity_count = torch.zeros((num_anchors, 1), dtype=torch.int32, device="cuda")
        self.temporal_last_opacity_max = torch.zeros((num_anchors, 1), dtype=torch.float32, device="cuda")
        self.temporal_local_mask = torch.zeros((num_anchors,), dtype=torch.bool, device="cuda")
        self.temporal_local_parent_ids = torch.full((num_anchors,), -1, dtype=torch.long, device="cuda")
        self.temporal_anomaly_score = torch.zeros((num_anchors,), dtype=torch.float32, device="cuda")
        self.temporal_anomaly_count = torch.zeros((num_anchors,), dtype=torch.int32, device="cuda")
        self.temporal_anomaly_candidate_mask = torch.zeros((num_anchors,), dtype=torch.bool, device="cuda")
        self.temporal_split_suppression_mask = torch.zeros((num_anchors,), dtype=torch.bool, device="cuda")
        self.temporal_attribute_grad_mask = None

    def update_temporal_learning_rate(self, iteration, training_args):
        if self.temporal_optimizer is None:
            return
        for param_group in self.temporal_optimizer.param_groups:
            group_name = param_group.get("name", "")
            if group_name == "temporal_latent":
                param_group["lr"] = training_args.latent_lr
            elif group_name == "temporal_history_mlp":
                param_group["lr"] = training_args.mlp_temporal_bank_lr_init
            elif group_name == "temporal_opacity_mlp":
                param_group["lr"] = training_args.mlp_opacity_lr_init
            elif group_name == "temporal_cov_mlp":
                param_group["lr"] = training_args.mlp_cov_lr_init
            elif group_name == "temporal_color_mlp":
                param_group["lr"] = training_args.mlp_color_lr_init

    def get_temporal_visible_latent(self, visible_mask, visible_feat=None):
        active_state = self.temporal_state if self.temporal_state is not None else self.base_temporal_state
        if active_state is None:
            return None
        active_state.set_layout(
            num_times=self.temporal_num_times,
            chunk_dim=self.temporal_block_dim,
        )
        active_state.set_active_time_step(int(self.current_time_step))
        start, end = self._state_current_block_slice(active_state, int(self.current_time_step))
        if end <= start:
            return None
        latent = getattr(active_state, "latent", None)
        if latent is None:
            return None
        visible_mask = visible_mask.to(device=latent.device, dtype=torch.bool).reshape(-1)
        if int(visible_mask.shape[0]) != int(latent.shape[0]):
            return None
        z_cur = latent[visible_mask, start:end]
        if visible_feat is None:
            visible_feat = self._anchor_feat[visible_mask]
        if visible_feat is None:
            return None
        visible_feat = visible_feat.to(device=z_cur.device, dtype=z_cur.dtype)

        birth_time = None
        if self.temporal_anchor_birth_timestep is not None:
            birth_time = self.temporal_anchor_birth_timestep[visible_mask].to(device=z_cur.device, dtype=torch.long)
        else:
            birth_time = torch.zeros((visible_feat.shape[0],), device=z_cur.device, dtype=torch.long)
        time_id = int(self.current_time_step)
        old_mask = birth_time < time_id
        new_mask = birth_time == time_id

        h_t = visible_feat.clone()
        z_for_head = torch.zeros(
            (visible_feat.shape[0], self.temporal_block_dim),
            device=visible_feat.device,
            dtype=visible_feat.dtype,
        )
        history_mlp = self._get_current_history_mlp()
        if history_mlp is not None and bool(old_mask.any().item()):
            delta_old = history_mlp(z_cur[old_mask])
            h_t[old_mask] = visible_feat[old_mask] + delta_old
            z_for_head[old_mask] = z_cur[old_mask]
        if bool(new_mask.any().item()):
            h_t[new_mask] = visible_feat[new_mask]
            z_for_head[new_mask].zero_()
        temporal_input = torch.cat([h_t, z_for_head], dim=1)
        visible_grad_mask = self.get_temporal_attribute_gradient_mask(visible_mask)
        if visible_grad_mask is not None and int(visible_grad_mask.shape[0]) == int(temporal_input.shape[0]):
            temporal_input = torch.where(
                visible_grad_mask.unsqueeze(-1),
                temporal_input,
                temporal_input.detach(),
            )
        return temporal_input

    def get_temporal_regularization_loss(self):
        device = self.get_anchor.device
        latent_reg = torch.tensor(0.0, device=device)
        mlp_reg = torch.tensor(0.0, device=device)

        active_state = self.temporal_state if self.temporal_state is not None else self.base_temporal_state
        if active_state is not None:
            active_state.set_layout(
                num_times=self.temporal_num_times,
                chunk_dim=self.temporal_block_dim,
            )
            active_state.set_active_time_step(int(self.current_time_step))
            start, end = self._state_current_block_slice(active_state, int(self.current_time_step))
            if end > start:
                latent_reg = (active_state.latent[:, start:end] ** 2).mean()
        return latent_reg, mlp_reg

    def _replace_temporal_optimizer_param(self, group_name: str, new_param: nn.Parameter):
        if self.temporal_optimizer is None:
            return
        for group in self.temporal_optimizer.param_groups:
            if group.get("name", "") != group_name:
                continue
            old_param = group["params"][0]
            stored_state = self.temporal_optimizer.state.get(old_param, None)
            if stored_state is not None:
                del self.temporal_optimizer.state[old_param]
                exp_avg = stored_state.get("exp_avg", None)
                exp_avg_sq = stored_state.get("exp_avg_sq", None)
                if exp_avg is not None and tuple(exp_avg.shape) != tuple(new_param.shape):
                    exp_avg = torch.zeros_like(new_param)
                if exp_avg_sq is not None and tuple(exp_avg_sq.shape) != tuple(new_param.shape):
                    exp_avg_sq = torch.zeros_like(new_param)
                stored_state["exp_avg"] = exp_avg if exp_avg is not None else torch.zeros_like(new_param)
                stored_state["exp_avg_sq"] = exp_avg_sq if exp_avg_sq is not None else torch.zeros_like(new_param)
                self.temporal_optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            return

    def _append_temporal_state_rows(self, latent_extension: torch.Tensor):
        if self.temporal_state is None or latent_extension is None or latent_extension.numel() == 0:
            return
        latent_extension = latent_extension.to(
            device=self.temporal_state.latent.device,
            dtype=self.temporal_state.latent.dtype,
        )
        new_param = nn.Parameter(
            torch.cat((self.temporal_state.latent.detach(), latent_extension), dim=0).requires_grad_(
                self.temporal_state.latent.requires_grad
            )
        )
        self._replace_temporal_optimizer_param("temporal_latent", new_param)
        self.temporal_state.latent = new_param
        self.temporal_state.num_anchors = int(new_param.shape[0])

    def _prune_temporal_state_rows(self, valid_mask: torch.Tensor):
        if self.temporal_state is None:
            return
        valid_mask = valid_mask.to(device=self.temporal_state.latent.device, dtype=torch.bool).reshape(-1)
        new_param = nn.Parameter(
            self.temporal_state.latent.detach()[valid_mask].requires_grad_(
                self.temporal_state.latent.requires_grad
            )
        )
        self._replace_temporal_optimizer_param("temporal_latent", new_param)
        self.temporal_state.latent = new_param
        self.temporal_state.num_anchors = int(new_param.shape[0])

    def _append_temporal_metadata(
        self,
        num_new: int,
        *,
        local_mask_value: bool = False,
        parent_ids=None,
        latent_extension=None,
        suppress_parent_rows=None,
        birth_timestep=None,
        death_timestep=None,
    ):
        if num_new <= 0:
            return

        device = self.get_anchor.device
        if latent_extension is None:
            latent_extension = torch.zeros(
                (int(num_new), int(self.temporal_block_dim)),
                device=device,
                dtype=self.get_anchor.dtype,
            )
        elif latent_extension.ndim == 2 and int(latent_extension.shape[1]) != int(self.temporal_block_dim):
            if int(latent_extension.shape[1]) > int(self.temporal_block_dim):
                latent_extension = latent_extension[:, : int(self.temporal_block_dim)]
            else:
                padded = torch.zeros(
                    (int(latent_extension.shape[0]), int(self.temporal_block_dim)),
                    device=latent_extension.device,
                    dtype=latent_extension.dtype,
                )
                padded[:, : int(latent_extension.shape[1])] = latent_extension
                latent_extension = padded
        self._append_temporal_state_rows(latent_extension)

        if self.base_temporal_state is not None:
            if parent_ids is not None and parent_ids.numel() == num_new:
                parent_ids = parent_ids.to(
                    device=self.base_temporal_state.latent.device,
                    dtype=torch.long,
                ).reshape(-1)
                copied = self.base_temporal_state.latent.detach()[parent_ids].clone()
                self._append_optimizer_group_rows("base_temporal_latent", copied)
            else:
                self._append_base_temporal_latent(int(num_new))
            for group in self.optimizer.param_groups:
                if group["name"] == "base_temporal_latent":
                    self.base_temporal_state.latent = group["params"][0]
                    self.base_temporal_state.num_anchors = int(self.base_temporal_state.latent.shape[0])
                    break

        if self.temporal_anchor_birth_timestep is not None:
            extra_birth = torch.full(
                (int(num_new),),
                int(self.current_time_step if birth_timestep is None else birth_timestep),
                dtype=torch.long,
                device=device,
            )
            self.temporal_anchor_birth_timestep = torch.cat(
                (self.temporal_anchor_birth_timestep, extra_birth),
                dim=0,
            )
        if self.temporal_anchor_death_timestep is not None:
            default_death = 2**31 - 1
            extra_death = torch.full(
                (int(num_new),),
                int(default_death if death_timestep is None else death_timestep),
                dtype=torch.long,
                device=device,
            )
            self.temporal_anchor_death_timestep = torch.cat(
                (self.temporal_anchor_death_timestep, extra_death),
                dim=0,
            )
            if suppress_parent_rows is not None and suppress_parent_rows.numel() > 0:
                parent_rows = suppress_parent_rows.to(device=device, dtype=torch.long).reshape(-1)
                parent_rows = parent_rows[
                    (parent_rows >= 0) & (parent_rows < self.temporal_anchor_death_timestep.shape[0])
                ]
                if parent_rows.numel() > 0:
                    self.temporal_anchor_death_timestep[parent_rows] = torch.minimum(
                        self.temporal_anchor_death_timestep[parent_rows],
                        torch.full_like(
                            self.temporal_anchor_death_timestep[parent_rows],
                            int(self.current_time_step),
                        ),
                    )

        if self.temporal_opacity_accum is not None:
            self.temporal_opacity_accum = torch.cat(
                (
                    self.temporal_opacity_accum,
                    torch.zeros((int(num_new), 1), dtype=torch.float32, device=device),
                ),
                dim=0,
            )
        if self.temporal_opacity_count is not None:
            self.temporal_opacity_count = torch.cat(
                (
                    self.temporal_opacity_count,
                    torch.zeros((int(num_new), 1), dtype=torch.int32, device=device),
                ),
                dim=0,
            )
        if self.temporal_last_opacity_max is not None:
            self.temporal_last_opacity_max = torch.cat(
                (
                    self.temporal_last_opacity_max,
                    torch.zeros((int(num_new), 1), dtype=torch.float32, device=device),
                ),
                dim=0,
            )
        if self.temporal_local_mask is not None:
            extra_local = torch.full(
                (int(num_new),),
                bool(local_mask_value),
                dtype=torch.bool,
                device=device,
            )
            self.temporal_local_mask = torch.cat((self.temporal_local_mask, extra_local), dim=0)
        if self.temporal_local_parent_ids is not None:
            if parent_ids is None:
                extra_parent_ids = torch.full((int(num_new),), -1, dtype=torch.long, device=device)
            else:
                extra_parent_ids = parent_ids.to(device=device, dtype=torch.long).reshape(-1)
            self.temporal_local_parent_ids = torch.cat((self.temporal_local_parent_ids, extra_parent_ids), dim=0)
        if self.temporal_anomaly_score is not None:
            self.temporal_anomaly_score = torch.cat(
                (self.temporal_anomaly_score, torch.zeros((int(num_new),), dtype=torch.float32, device=device)),
                dim=0,
            )
        if self.temporal_anomaly_count is not None:
            self.temporal_anomaly_count = torch.cat(
                (self.temporal_anomaly_count, torch.zeros((int(num_new),), dtype=torch.int32, device=device)),
                dim=0,
            )
        if self.temporal_anomaly_candidate_mask is not None:
            self.temporal_anomaly_candidate_mask = torch.cat(
                (self.temporal_anomaly_candidate_mask, torch.zeros((int(num_new),), dtype=torch.bool, device=device)),
                dim=0,
            )
        if self.temporal_split_suppression_mask is not None:
            self.temporal_split_suppression_mask = torch.cat(
                (self.temporal_split_suppression_mask, torch.zeros((int(num_new),), dtype=torch.bool, device=device)),
                dim=0,
            )

    def _prune_temporal_metadata(self, valid_mask: torch.Tensor):
        valid_mask = valid_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
        if (
            self.temporal_anchor_birth_timestep is not None
            and int(self.temporal_anchor_birth_timestep.shape[0]) == int(valid_mask.shape[0])
        ):
            self.temporal_anchor_birth_timestep = self.temporal_anchor_birth_timestep[valid_mask]
        if (
            self.temporal_anchor_death_timestep is not None
            and int(self.temporal_anchor_death_timestep.shape[0]) == int(valid_mask.shape[0])
        ):
            self.temporal_anchor_death_timestep = self.temporal_anchor_death_timestep[valid_mask]
        if self.temporal_opacity_accum is not None and int(self.temporal_opacity_accum.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_opacity_accum = self.temporal_opacity_accum[valid_mask]
        if self.temporal_opacity_count is not None and int(self.temporal_opacity_count.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_opacity_count = self.temporal_opacity_count[valid_mask]
        if self.temporal_last_opacity_max is not None and int(self.temporal_last_opacity_max.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_last_opacity_max = self.temporal_last_opacity_max[valid_mask]
        if self.temporal_local_mask is not None and int(self.temporal_local_mask.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_local_mask = self.temporal_local_mask[valid_mask]
        if self.temporal_local_parent_ids is not None and int(self.temporal_local_parent_ids.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_local_parent_ids = self.temporal_local_parent_ids[valid_mask]
        if self.temporal_anomaly_score is not None and int(self.temporal_anomaly_score.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_anomaly_score = self.temporal_anomaly_score[valid_mask]
        if self.temporal_anomaly_count is not None and int(self.temporal_anomaly_count.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_anomaly_count = self.temporal_anomaly_count[valid_mask]
        if self.temporal_anomaly_candidate_mask is not None and int(self.temporal_anomaly_candidate_mask.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_anomaly_candidate_mask = self.temporal_anomaly_candidate_mask[valid_mask]
        if self.temporal_split_suppression_mask is not None and int(self.temporal_split_suppression_mask.shape[0]) == int(valid_mask.shape[0]):
            self.temporal_split_suppression_mask = self.temporal_split_suppression_mask[valid_mask]
        self._prune_temporal_state_rows(valid_mask)

    def get_temporal_local_mask(self):
        n = int(self.get_anchor.shape[0])
        if self.temporal_local_mask is None or int(self.temporal_local_mask.shape[0]) != n:
            return torch.zeros((n,), dtype=torch.bool, device=self.get_anchor.device)
        return self.temporal_local_mask

    def get_temporal_render_mask(self, time_step: int | None = None):
        n = int(self.get_anchor.shape[0])
        if n <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.get_anchor.device)
        target_time = int(self.current_time_step if time_step is None else time_step)
        device = self.get_anchor.device
        visible_mask = torch.ones((n,), dtype=torch.bool, device=device)
        if (
            self.temporal_anchor_birth_timestep is not None
            and int(self.temporal_anchor_birth_timestep.shape[0]) == n
        ):
            visible_mask &= self.temporal_anchor_birth_timestep.to(device=device, dtype=torch.long) <= target_time
        if (
            self.temporal_anchor_death_timestep is not None
            and int(self.temporal_anchor_death_timestep.shape[0]) == n
        ):
            visible_mask &= self.temporal_anchor_death_timestep.to(device=device, dtype=torch.long) > target_time
        return visible_mask

    def get_temporal_anomaly_candidate_mask(self):
        n = int(self.get_anchor.shape[0])
        if (
            self.temporal_anomaly_candidate_mask is None
            or int(self.temporal_anomaly_candidate_mask.shape[0]) != n
        ):
            return torch.zeros((n,), dtype=torch.bool, device=self.get_anchor.device)
        return self.temporal_anomaly_candidate_mask

    def get_temporal_split_suppression_mask(self):
        n = int(self.get_anchor.shape[0])
        if (
            self.temporal_split_suppression_mask is None
            or int(self.temporal_split_suppression_mask.shape[0]) != n
        ):
            return torch.zeros((n,), dtype=torch.bool, device=self.get_anchor.device)
        return self.temporal_split_suppression_mask

    def get_temporal_death_candidate_mask(self):
        return self.get_temporal_split_suppression_mask()

    def set_temporal_attribute_gradient_mask(self, anchor_mask=None):
        if anchor_mask is None:
            self.temporal_attribute_grad_mask = None
            return
        n = int(self.get_anchor.shape[0])
        anchor_mask = anchor_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
        if int(anchor_mask.shape[0]) != n:
            self.temporal_attribute_grad_mask = None
            return
        self.temporal_attribute_grad_mask = anchor_mask

    def get_temporal_attribute_gradient_mask(self, visible_mask=None):
        n = int(self.get_anchor.shape[0])
        if (
            self.temporal_attribute_grad_mask is None
            or int(self.temporal_attribute_grad_mask.shape[0]) != n
        ):
            return None
        if visible_mask is None:
            return self.temporal_attribute_grad_mask
        visible_mask = visible_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
        if int(visible_mask.shape[0]) != n:
            return None
        return self.temporal_attribute_grad_mask[visible_mask]

    @torch.no_grad()
    def reset_temporal_anomaly_state(self):
        n = int(self.get_anchor.shape[0])
        device = self.get_anchor.device
        self.temporal_anomaly_score = torch.zeros((n,), dtype=torch.float32, device=device)
        self.temporal_anomaly_count = torch.zeros((n,), dtype=torch.int32, device=device)
        self.temporal_anomaly_candidate_mask = torch.zeros((n,), dtype=torch.bool, device=device)
        self.temporal_split_suppression_mask = torch.zeros((n,), dtype=torch.bool, device=device)

    def get_temporal_event_counters(self):
        return {
            "split_events": int(getattr(self, "temporal_split_event_count", 0)),
            "death_events": int(getattr(self, "temporal_death_event_count", 0)),
            "suppression_events": int(getattr(self, "temporal_suppression_event_count", 0)),
        }

    @torch.no_grad()
    def update_temporal_anomaly_votes(self, anchor_votes: torch.Tensor, vote_threshold: float):
        if anchor_votes is None:
            return
        n = int(self.get_anchor.shape[0])
        if n <= 0:
            return
        anchor_votes = anchor_votes.to(device=self.get_anchor.device, dtype=torch.float32).reshape(-1)
        if int(anchor_votes.shape[0]) != n:
            return
        if (
            self.temporal_anomaly_score is None
            or int(self.temporal_anomaly_score.shape[0]) != n
            or self.temporal_anomaly_count is None
            or int(self.temporal_anomaly_count.shape[0]) != n
            or self.temporal_anomaly_candidate_mask is None
            or int(self.temporal_anomaly_candidate_mask.shape[0]) != n
        ):
            self.reset_temporal_anomaly_state()
        positive_mask = anchor_votes > 0
        self.temporal_anomaly_score += anchor_votes
        self.temporal_anomaly_count[positive_mask] += 1
        self.temporal_anomaly_candidate_mask |= anchor_votes > float(vote_threshold)

    @torch.no_grad()
    def mark_temporal_anchor_death(self, death_mask: torch.Tensor, death_timestep: int):
        if self.temporal_anchor_death_timestep is None:
            return
        death_mask = death_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
        n = int(self.get_anchor.shape[0])
        if int(death_mask.shape[0]) != n:
            return
        if bool(death_mask.any()):
            self.temporal_anchor_death_timestep[death_mask] = torch.minimum(
                self.temporal_anchor_death_timestep[death_mask],
                torch.full_like(self.temporal_anchor_death_timestep[death_mask], int(death_timestep)),
            )

    @torch.no_grad()
    def update_temporal_split_suppression_from_deaths(self, death_mask: torch.Tensor, neighbor_radius: float):
        n = int(self.get_anchor.shape[0])
        if n <= 1:
            return 0
        if (
            self.temporal_split_suppression_mask is None
            or int(self.temporal_split_suppression_mask.shape[0]) != n
        ):
            self.temporal_split_suppression_mask = torch.zeros((n,), dtype=torch.bool, device=self.get_anchor.device)

        death_mask = death_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
        if int(death_mask.shape[0]) != n or not bool(death_mask.any()):
            return 0

        neighbor_count = max(int(round(float(neighbor_radius))), 0)
        if neighbor_count <= 0:
            return 0

        dead_ids = torch.where(death_mask)[0]
        alive_ids = torch.where(~death_mask)[0]
        if dead_ids.numel() == 0 or alive_ids.numel() == 0:
            return 0

        k = min(neighbor_count, int(alive_ids.numel()))
        if k <= 0:
            return 0

        dead_xyz = self.get_anchor[dead_ids]
        alive_xyz = self.get_anchor[alive_ids]
        dist = torch.cdist(dead_xyz, alive_xyz)
        if dist.numel() == 0:
            return 0
        nearest_cols = torch.topk(dist, k=k, dim=1, largest=False).indices.reshape(-1)
        neighbor_ids = alive_ids[nearest_cols]
        if neighbor_ids.numel() == 0:
            return 0
        neighbor_ids = torch.unique(neighbor_ids)
        new_mask = ~self.temporal_split_suppression_mask[neighbor_ids]
        if not bool(new_mask.any()):
            return 0
        selected_neighbor_ids = neighbor_ids[new_mask]
        self.temporal_split_suppression_mask[selected_neighbor_ids] = True
        return int(selected_neighbor_ids.numel())

    def enter_temporal_clone_phase(self, start_iteration: int):
        self.temporal_phase3_active = True
        self.temporal_local_growth_active = False
        self.temporal_phase3_start_iter = int(start_iteration)

        self.set_scaffold_geometry_requires_grad(True)

        if self.base_temporal_state is not None:
            for param in self.base_temporal_state.parameters():
                param.requires_grad_(True)
        if self.temporal_state is not None:
            for param in self.temporal_state.parameters():
                param.requires_grad_(True)
        self._set_module_requires_grad(self._get_current_history_mlp(), True)
        for kind in ("opacity", "cov", "color"):
            self._set_module_requires_grad(self._get_current_property_mlp(kind), True)

    def set_scaffold_geometry_requires_grad(self, enabled: bool):
        for tensor in [self._anchor, self._offset, self._anchor_feat, self._opacity, self._scaling, self._rotation]:
            if isinstance(tensor, nn.Parameter):
                tensor.requires_grad_(bool(enabled))

    def exit_temporal_clone_phase(self):
        self.temporal_phase3_active = False
        self.temporal_local_growth_active = False

        for tensor in [self._anchor, self._offset, self._anchor_feat, self._opacity, self._scaling, self._rotation]:
            if isinstance(tensor, nn.Parameter):
                tensor.requires_grad_(False)

        for module in [self.mlp_opacity, self.mlp_cov, self.mlp_color]:
            self._set_module_requires_grad(module, False)

        if self.embedding_appearance is not None:
            for param in self.embedding_appearance.parameters():
                param.requires_grad_(self.appearance_dim > 0)

        if self.base_temporal_state is not None:
            for param in self.base_temporal_state.parameters():
                param.requires_grad_(True)

        if self.temporal_state is not None:
            for param in self.temporal_state.parameters():
                param.requires_grad_(True)

        self._set_module_requires_grad(self._get_current_history_mlp(), True)
        for kind in ("opacity", "cov", "color"):
            self._set_module_requires_grad(self._get_current_property_mlp(kind), True)

    def set_temporal_local_growth_active(self, enabled: bool):
        self.temporal_local_growth_active = bool(enabled)

    @torch.no_grad()
    def append_temporal_local_clones(self, parent_row_ids: torch.Tensor, suppress_parents: bool = True) -> int:
        if parent_row_ids is None or parent_row_ids.numel() == 0:
            return 0

        n_total = int(self.get_anchor.shape[0])
        parent_row_ids = parent_row_ids.to(device=self.get_anchor.device, dtype=torch.long).reshape(-1)
        parent_row_ids = torch.unique(parent_row_ids)
        parent_row_ids = parent_row_ids[(parent_row_ids >= 0) & (parent_row_ids < n_total)]
        if parent_row_ids.numel() == 0:
            return 0

        local_mask = self.get_temporal_local_mask()
        parent_row_ids = parent_row_ids[~local_mask[parent_row_ids]]
        if parent_row_ids.numel() == 0:
            return 0

        clone_payload = {
            "anchor": self._anchor.detach()[parent_row_ids].clone(),
            "offset": self._offset.detach()[parent_row_ids].clone(),
            "anchor_feat": self._anchor_feat.detach()[parent_row_ids].clone(),
            "opacity": self._opacity.detach()[parent_row_ids].clone(),
            "scaling": self._scaling.detach()[parent_row_ids].clone(),
            "rotation": self._rotation.detach()[parent_row_ids].clone(),
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(clone_payload)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        n_new = int(parent_row_ids.shape[0])
        self.anchor_demon = torch.cat(
            (self.anchor_demon, torch.zeros((n_new, 1), dtype=self.anchor_demon.dtype, device=self.anchor_demon.device)),
            dim=0,
        )
        self.opacity_accum = torch.cat(
            (self.opacity_accum, torch.zeros((n_new, 1), dtype=self.opacity_accum.dtype, device=self.opacity_accum.device)),
            dim=0,
        )
        self.offset_denom = torch.cat(
            (
                self.offset_denom,
                torch.zeros((n_new * self.n_offsets, 1), dtype=self.offset_denom.dtype, device=self.offset_denom.device),
            ),
            dim=0,
        )
        self.offset_gradient_accum = torch.cat(
            (
                self.offset_gradient_accum,
                torch.zeros(
                    (n_new * self.n_offsets, 1),
                    dtype=self.offset_gradient_accum.dtype,
                    device=self.offset_gradient_accum.device,
                ),
            ),
            dim=0,
        )
        self.max_radii2D = torch.cat(
            (self.max_radii2D, torch.zeros((n_new,), dtype=self.max_radii2D.dtype, device=self.max_radii2D.device)),
            dim=0,
        )
        latent_extension = None
        if self.temporal_state is not None:
            latent_extension = self.temporal_state.latent.detach()[parent_row_ids].clone()
        self._append_temporal_metadata(
            n_new,
            local_mask_value=True,
            parent_ids=parent_row_ids,
            latent_extension=latent_extension,
            suppress_parent_rows=parent_row_ids if suppress_parents else None,
            birth_timestep=int(self.current_time_step),
        )
        return n_new

    def apply_temporal_local_gradient_mask(self, active_anchor_mask=None):
        if active_anchor_mask is None:
            active_anchor_mask = self.get_temporal_local_mask()
        active_anchor_mask = active_anchor_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
        n = int(active_anchor_mask.shape[0])
        if n <= 0:
            return
        freeze_mask = ~active_anchor_mask
        for tensor_name in ["_anchor", "_offset", "_anchor_feat", "_opacity", "_scaling", "_rotation"]:
            tensor = getattr(self, tensor_name, None)
            grad = None if tensor is None else tensor.grad
            if grad is None or int(grad.shape[0]) != n:
                continue
            grad[freeze_mask] = 0
            if self.optimizer is None:
                continue
            stored_state = self.optimizer.state.get(tensor, None)
            if not isinstance(stored_state, dict):
                continue
            exp_avg = stored_state.get("exp_avg", None)
            exp_avg_sq = stored_state.get("exp_avg_sq", None)
            if isinstance(exp_avg, torch.Tensor) and int(exp_avg.shape[0]) == n:
                exp_avg[freeze_mask] = 0
            if isinstance(exp_avg_sq, torch.Tensor) and int(exp_avg_sq.shape[0]) == n:
                exp_avg_sq[freeze_mask] = 0

    def apply_temporal_latent_block_gradient_mask(self, active_anchor_mask=None):
        active_state = self.temporal_state if self.temporal_state is not None else self.base_temporal_state
        if active_state is None:
            return

        latent = active_state.latent
        grad = getattr(latent, "grad", None)
        if grad is None:
            return

        active_state.set_layout(
            num_times=self.temporal_num_times,
            chunk_dim=self.temporal_block_dim,
        )
        active_state.set_active_time_step(int(self.current_time_step))
        start, end = self._state_current_block_slice(active_state, int(self.current_time_step))
        if end <= start:
            grad.zero_()
            return

        def _zero_non_current_block(tensor: torch.Tensor) -> None:
            if tensor.ndim != 2 or tuple(tensor.shape) != tuple(grad.shape):
                return
            if start > 0:
                tensor[:, :start].zero_()
            if end < int(tensor.shape[1]):
                tensor[:, end:].zero_()

        _zero_non_current_block(grad)
        if active_anchor_mask is not None:
            active_anchor_mask = active_anchor_mask.to(device=grad.device, dtype=torch.bool).reshape(-1)
            if int(active_anchor_mask.shape[0]) == int(grad.shape[0]):
                grad[~active_anchor_mask, start:end] = 0

        for optimizer in (self.temporal_optimizer, self.optimizer):
            if optimizer is None:
                continue
            stored_state = optimizer.state.get(latent, None)
            if not isinstance(stored_state, dict):
                continue
            exp_avg = stored_state.get("exp_avg", None)
            exp_avg_sq = stored_state.get("exp_avg_sq", None)
            if isinstance(exp_avg, torch.Tensor) and tuple(exp_avg.shape) == tuple(grad.shape):
                _zero_non_current_block(exp_avg)
                if active_anchor_mask is not None and int(active_anchor_mask.shape[0]) == int(exp_avg.shape[0]):
                    exp_avg[~active_anchor_mask, start:end] = 0
            if isinstance(exp_avg_sq, torch.Tensor) and tuple(exp_avg_sq.shape) == tuple(grad.shape):
                _zero_non_current_block(exp_avg_sq)
                if active_anchor_mask is not None and int(active_anchor_mask.shape[0]) == int(exp_avg_sq.shape[0]):
                    exp_avg_sq[~active_anchor_mask, start:end] = 0

    @torch.no_grad()
    def clamp_trainable_scaling(self, active_anchor_mask=None, max_log_scale: float = 0.05):
        if self._scaling is None or self._scaling.numel() == 0 or int(self._scaling.shape[1]) <= 3:
            return
        if active_anchor_mask is None:
            self._scaling[:, 3:].clamp_(max=float(max_log_scale))
            return
        active_anchor_mask = active_anchor_mask.to(
            device=self._scaling.device,
            dtype=torch.bool,
        ).reshape(-1)
        if int(active_anchor_mask.shape[0]) != int(self._scaling.shape[0]):
            return
        if not bool(active_anchor_mask.any().item()):
            return
        self._scaling[active_anchor_mask, 3:].clamp_(max=float(max_log_scale))

    def get_temporal_checkpoint_payload(self, time_step: int):
        num_anchors = int(self.get_anchor.shape[0])
        payload = {
            "time_step": int(time_step),
            "latent_dim": int(self.temporal_latent_dim),
            "temporal_block_dim": int(self.temporal_block_dim),
            "temporal_num_times": int(self.temporal_num_times),
        }
        active_state = self.temporal_state if self.temporal_state is not None else self.base_temporal_state
        if active_state is not None:
            start, end = self._state_current_block_slice(active_state, int(time_step))
            if end > start:
                payload["latent_block"] = self._pack_sparse_latent_rows(
                    active_state.latent[:, start:end],
                    None,
                )
                payload["latent_block_dim"] = int(end - start)
        if self.temporal_last_opacity_max is not None:
            payload["last_opacity_max"] = self.temporal_last_opacity_max.detach().cpu()
        else:
            payload["last_opacity_max"] = torch.zeros((num_anchors, 1), dtype=torch.float32).cpu()
        if self.temporal_anchor_birth_timestep is not None:
            payload["birth_timestep"] = self.temporal_anchor_birth_timestep.detach().cpu()
        else:
            payload["birth_timestep"] = torch.zeros((num_anchors,), dtype=torch.long).cpu()
        if self.temporal_anchor_death_timestep is not None:
            payload["death_timestep"] = self.temporal_anchor_death_timestep.detach().cpu()
        else:
            payload["death_timestep"] = torch.full((num_anchors,), 2**31 - 1, dtype=torch.long).cpu()
        if self.temporal_local_mask is not None:
            payload["local_mask"] = self.temporal_local_mask.detach().cpu()
        else:
            payload["local_mask"] = torch.zeros((num_anchors,), dtype=torch.bool).cpu()
        if self.temporal_local_parent_ids is not None:
            payload["local_parent_ids"] = self.temporal_local_parent_ids.detach().cpu()
        else:
            payload["local_parent_ids"] = torch.full((num_anchors,), -1, dtype=torch.long).cpu()
        if self.temporal_anomaly_score is not None:
            payload["anomaly_score"] = self.temporal_anomaly_score.detach().cpu()
        else:
            payload["anomaly_score"] = torch.zeros((num_anchors,), dtype=torch.float32).cpu()
        if self.temporal_anomaly_count is not None:
            payload["anomaly_count"] = self.temporal_anomaly_count.detach().cpu()
        else:
            payload["anomaly_count"] = torch.zeros((num_anchors,), dtype=torch.int32).cpu()
        if self.temporal_anomaly_candidate_mask is not None:
            payload["anomaly_candidate_mask"] = self.temporal_anomaly_candidate_mask.detach().cpu()
        else:
            payload["anomaly_candidate_mask"] = torch.zeros((num_anchors,), dtype=torch.bool).cpu()
        if self.temporal_split_suppression_mask is not None:
            payload["split_suppression_mask"] = self.temporal_split_suppression_mask.detach().cpu()
        else:
            payload["split_suppression_mask"] = torch.zeros((num_anchors,), dtype=torch.bool).cpu()
        payload["phase3_active"] = bool(self.temporal_phase3_active)
        payload["phase3_start_iter"] = int(self.temporal_phase3_start_iter)
        payload["base_anchor_count"] = int(self.temporal_base_anchor_count)
        return payload

    def load_temporal_checkpoint_payload(self, payload, training_args, time_step: int):
        self.setup_temporal_adaptation(training_args, time_step=time_step)
        if not isinstance(payload, dict):
            return
        self.current_time_step = int(time_step)
        payload_num_times = payload.get("temporal_num_times", None)
        payload_block_dim = payload.get("temporal_block_dim", None)
        if payload_num_times is not None:
            self.temporal_num_times = max(int(payload_num_times), 1)
        if payload_block_dim is not None:
            self.temporal_block_dim = max(int(payload_block_dim), 1)
        latent_block = payload.get("latent_block", None)
        latent_tensor = payload.get("latent", None)
        active_state = self.temporal_state if self.temporal_state is not None else self.base_temporal_state
        if active_state is not None:
            if latent_block is not None:
                latent_block = self._unpack_sparse_latent_rows(
                    latent_block,
                    expected_num_anchors=int(active_state.latent.shape[0]),
                    expected_latent_dim=int(payload.get("latent_block_dim", self.temporal_block_dim)),
                    device="cuda",
                    dtype=active_state.latent.dtype,
                )
                if latent_block is not None:
                    start, end = self._state_current_block_slice(active_state, int(time_step))
                    active_width = int(active_state.latent.shape[1])
                    if active_width == int(latent_block.shape[1]):
                        with torch.no_grad():
                            active_state.latent.copy_(latent_block)
                        self.temporal_compact_time_step = int(time_step)
                    elif end > start and int(latent_block.shape[1]) == int(end - start):
                        with torch.no_grad():
                            active_state.latent[:, start:end].copy_(latent_block)
            elif latent_tensor is not None:
                latent_tensor = self._unpack_sparse_latent_rows(
                    latent_tensor,
                    expected_num_anchors=int(active_state.latent.shape[0]),
                    expected_latent_dim=int(active_state.latent.shape[1]),
                    device="cuda",
                    dtype=active_state.latent.dtype,
                )
                if latent_tensor is not None:
                    with torch.no_grad():
                        active_state.latent.copy_(latent_tensor)
        last_opacity_max = payload.get("last_opacity_max", None)
        if last_opacity_max is not None:
            last_opacity_max = last_opacity_max.to(device="cuda", dtype=torch.float32)
            if (
                self.temporal_last_opacity_max is not None
                and int(last_opacity_max.shape[0]) == int(self.temporal_last_opacity_max.shape[0])
            ):
                self.temporal_last_opacity_max = last_opacity_max
        birth_tensor = payload.get("birth_timestep", None)
        if birth_tensor is not None:
            birth_tensor = birth_tensor.to(device="cuda", dtype=torch.long).reshape(-1)
            if (
                self.temporal_anchor_birth_timestep is not None
                and int(birth_tensor.shape[0]) == int(self.temporal_anchor_birth_timestep.shape[0])
            ):
                self.temporal_anchor_birth_timestep = birth_tensor
        death_tensor = payload.get("death_timestep", None)
        if death_tensor is not None:
            death_tensor = death_tensor.to(device="cuda", dtype=torch.long).reshape(-1)
            if (
                self.temporal_anchor_death_timestep is not None
                and int(death_tensor.shape[0]) == int(self.temporal_anchor_death_timestep.shape[0])
            ):
                self.temporal_anchor_death_timestep = death_tensor
        local_mask = payload.get("local_mask", None)
        if local_mask is not None:
            local_mask = local_mask.to(device="cuda", dtype=torch.bool).reshape(-1)
            if self.temporal_local_mask is not None and int(local_mask.shape[0]) == int(self.temporal_local_mask.shape[0]):
                self.temporal_local_mask = local_mask
        local_parent_ids = payload.get("local_parent_ids", None)
        if local_parent_ids is not None:
            local_parent_ids = local_parent_ids.to(device="cuda", dtype=torch.long).reshape(-1)
            if (
                self.temporal_local_parent_ids is not None
                and int(local_parent_ids.shape[0]) == int(self.temporal_local_parent_ids.shape[0])
            ):
                self.temporal_local_parent_ids = local_parent_ids
        anomaly_score = payload.get("anomaly_score", None)
        if anomaly_score is not None:
            anomaly_score = anomaly_score.to(device="cuda", dtype=torch.float32).reshape(-1)
            if self.temporal_anomaly_score is not None and int(anomaly_score.shape[0]) == int(self.temporal_anomaly_score.shape[0]):
                self.temporal_anomaly_score = anomaly_score
        anomaly_count = payload.get("anomaly_count", None)
        if anomaly_count is not None:
            anomaly_count = anomaly_count.to(device="cuda", dtype=torch.int32).reshape(-1)
            if self.temporal_anomaly_count is not None and int(anomaly_count.shape[0]) == int(self.temporal_anomaly_count.shape[0]):
                self.temporal_anomaly_count = anomaly_count
        anomaly_candidate_mask = payload.get("anomaly_candidate_mask", None)
        if anomaly_candidate_mask is not None:
            anomaly_candidate_mask = anomaly_candidate_mask.to(device="cuda", dtype=torch.bool).reshape(-1)
            if (
                self.temporal_anomaly_candidate_mask is not None
                and int(anomaly_candidate_mask.shape[0]) == int(self.temporal_anomaly_candidate_mask.shape[0])
            ):
                self.temporal_anomaly_candidate_mask = anomaly_candidate_mask
        split_suppression_mask = payload.get("split_suppression_mask", None)
        if split_suppression_mask is not None:
            split_suppression_mask = split_suppression_mask.to(device="cuda", dtype=torch.bool).reshape(-1)
            if (
                self.temporal_split_suppression_mask is not None
                and int(split_suppression_mask.shape[0]) == int(self.temporal_split_suppression_mask.shape[0])
            ):
                self.temporal_split_suppression_mask = split_suppression_mask
        self.temporal_phase3_active = bool(payload.get("phase3_active", False))
        self.temporal_phase3_start_iter = int(payload.get("phase3_start_iter", -1))
        self.temporal_base_anchor_count = int(payload.get("base_anchor_count", self.get_anchor.shape[0]))
        self.N_base_anchors = int(self.temporal_base_anchor_count)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]
        if self.mlp_temporal_bank is not None:
            l.append(
                {
                    'params': self.mlp_temporal_bank.parameters(),
                    'lr': training_args.mlp_temporal_bank_lr_init,
                    "name": "mlp_temporal_bank",
                }
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.mlp_temporal_bank is not None:
            self.mlp_temporal_bank_scheduler_args = get_expon_lr_func(
                lr_init=training_args.mlp_temporal_bank_lr_init,
                lr_final=training_args.mlp_temporal_bank_lr_final,
                lr_delay_mult=training_args.mlp_temporal_bank_lr_delay_mult,
                max_steps=training_args.mlp_temporal_bank_lr_max_steps,
            )
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        self.temporal_latent_scheduler_args = get_expon_lr_func(
            lr_init=training_args.latent_lr,
            lr_final=training_args.latent_lr,
            lr_delay_mult=1.0,
            max_steps=max(1, int(getattr(training_args, "iterations", 1))),
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "base_temporal_latent":
                lr = self.temporal_latent_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.mlp_temporal_bank is not None and param_group["name"] == "mlp_temporal_bank":
                lr = self.mlp_temporal_bank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                group['name'] == 'base_temporal_latent':
                continue
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


    # statis grad information to guide liftting. 
    def training_statis(
        self,
        viewspace_point_tensor,
        opacity,
        update_filter,
        offset_selection_mask,
        anchor_visible_mask,
        active_anchor_mask=None,
    ):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        n_anchors = int(self.get_anchor.shape[0])
        if anchor_visible_mask.dtype in (torch.int32, torch.int64, torch.long):
            visible_anchor_ids = anchor_visible_mask.to(device=self.get_anchor.device, dtype=torch.long).reshape(-1)
            visible_anchor_ids = visible_anchor_ids[
                (visible_anchor_ids >= 0) & (visible_anchor_ids < n_anchors)
            ]
            anchor_visible_mask = torch.zeros((n_anchors,), device=self.get_anchor.device, dtype=torch.bool)
            if visible_anchor_ids.numel() > 0:
                anchor_visible_mask[visible_anchor_ids] = True
        else:
            anchor_visible_mask = anchor_visible_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
            visible_anchor_ids = torch.where(anchor_visible_mask)[0]
        if int(temp_opacity.shape[0]) != int(visible_anchor_ids.numel()):
            return
        if active_anchor_mask is not None:
            active_anchor_mask = active_anchor_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
            if int(active_anchor_mask.shape[0]) == int(anchor_visible_mask.shape[0]):
                visible_keep_mask = active_anchor_mask[visible_anchor_ids]
                visible_anchor_ids = visible_anchor_ids[visible_keep_mask]
                temp_opacity = temp_opacity[visible_keep_mask]
        if visible_anchor_ids.numel() == 0:
            return
        self.opacity_accum[visible_anchor_ids] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[visible_anchor_ids] += 1

        # update neural gaussian statis
        visible_offset_ids = torch.where(
            anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        )[0]
        selected_offset_ids = visible_offset_ids[offset_selection_mask.reshape(-1)]
        selected_update_filter = update_filter.reshape(-1)
        selected_grad_source = viewspace_point_tensor.grad
        if active_anchor_mask is not None and int(active_anchor_mask.shape[0]) == int(anchor_visible_mask.shape[0]):
            repeated_active_mask = active_anchor_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).reshape(-1)
            selected_keep_mask = repeated_active_mask[selected_offset_ids]
            selected_offset_ids = selected_offset_ids[selected_keep_mask]
            selected_update_filter = selected_update_filter[selected_keep_mask]
            selected_grad_source = selected_grad_source[selected_keep_mask]
        if selected_offset_ids.numel() == 0 or not bool(selected_update_filter.any()):
            return

        updated_offset_ids = selected_offset_ids[selected_update_filter]
        grad_norm = torch.norm(selected_grad_source[selected_update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[updated_offset_ids] += grad_norm
        self.offset_denom[updated_offset_ids] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                group['name'] == 'base_temporal_latent':
                continue

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

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._prune_base_temporal_latent(valid_points_mask)
        self._prune_temporal_metadata(valid_points_mask)

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            candidate_offset_ids = torch.where(candidate_mask)[0]
            if candidate_offset_ids.numel() == 0:
                continue
            candidate_anchor_ids = torch.div(candidate_offset_ids, self.n_offsets, rounding_mode='floor')
            candidate_local_offset_ids = candidate_offset_ids % self.n_offsets
            selected_xyz = (
                self.get_anchor[candidate_anchor_ids]
                + self._offset[candidate_anchor_ids, candidate_local_offset_ids]
                * self.get_scaling[candidate_anchor_ids, :3]
            )
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                self.temporal_split_event_count += int(candidate_anchor.shape[0])
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat[candidate_anchor_ids]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._append_temporal_metadata(
                    int(candidate_anchor.shape[0]),
                    local_mask_value=bool(self.temporal_phase3_active or self.temporal_local_growth_active),
                )
                


    def adjust_anchor(
        self,
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        min_opacity=0.005,
        active_anchor_mask=None,
        anomaly_anchor_mask=None,
        suppress_growth_mask=None,
        death_timestep=None,
        suppress_neighbor_radius: float = 0.0,
        real_prune_temporal: bool = False,
    ):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        n_anchors = int(self.get_anchor.shape[0])
        anomaly_mask = None
        if anomaly_anchor_mask is not None:
            anomaly_mask = anomaly_anchor_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
            if int(anomaly_mask.shape[0]) != n_anchors:
                anomaly_mask = None
        growth_suppress_mask = None
        if suppress_growth_mask is not None:
            growth_suppress_mask = suppress_growth_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
            if int(growth_suppress_mask.shape[0]) != n_anchors:
                growth_suppress_mask = None
        if active_anchor_mask is not None:
            active_anchor_mask = active_anchor_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1)
            if int(active_anchor_mask.shape[0]) == int(self.get_anchor.shape[0]):
                repeated_active_mask = active_anchor_mask.unsqueeze(1).repeat([1, self.n_offsets]).reshape(-1)
                if int(repeated_active_mask.shape[0]) == int(offset_mask.shape[0]):
                    offset_mask = torch.logical_and(offset_mask, repeated_active_mask)
        if anomaly_mask is not None:
            repeated_anomaly_mask = anomaly_mask.unsqueeze(1).repeat([1, self.n_offsets]).reshape(-1)
            if int(repeated_anomaly_mask.shape[0]) == int(offset_mask.shape[0]):
                offset_mask = torch.logical_and(offset_mask, repeated_anomaly_mask)
        if growth_suppress_mask is not None:
            repeated_suppress_mask = growth_suppress_mask.unsqueeze(1).repeat([1, self.n_offsets]).reshape(-1)
            if int(repeated_suppress_mask.shape[0]) == int(offset_mask.shape[0]):
                offset_mask = torch.logical_and(offset_mask, ~repeated_suppress_mask)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        if active_anchor_mask is not None and int(active_anchor_mask.shape[0]) == int(prune_mask.shape[0]):
            prune_mask = torch.logical_and(prune_mask, active_anchor_mask)
        if anomaly_mask is not None and int(anomaly_mask.shape[0]) == int(prune_mask.shape[0]):
            prune_mask = torch.logical_and(prune_mask, anomaly_mask)
        if (
            death_timestep is not None
            and self.temporal_split_suppression_mask is not None
            and int(self.temporal_split_suppression_mask.shape[0]) == int(prune_mask.shape[0])
        ):
            prune_mask = torch.logical_or(
                prune_mask,
                torch.logical_and(
                    self.temporal_split_suppression_mask.to(device=self.get_anchor.device, dtype=torch.bool).reshape(-1),
                    anchors_mask,
                ),
            )

        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        if prune_mask.shape[0]>0:
            if bool(prune_mask.any()):
                death_count = int(prune_mask.sum().item())
                self.temporal_death_event_count += death_count
                if bool(real_prune_temporal):
                    keep_mask = ~prune_mask
                    if int(self.offset_denom.shape[0]) == int(prune_mask.shape[0]) * int(self.n_offsets):
                        self.offset_denom = self.offset_denom.view([-1, self.n_offsets])[keep_mask].reshape([-1, 1])
                    if int(self.offset_gradient_accum.shape[0]) == int(prune_mask.shape[0]) * int(self.n_offsets):
                        self.offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[keep_mask].reshape([-1, 1])
                    if int(self.opacity_accum.shape[0]) == int(prune_mask.shape[0]):
                        self.opacity_accum = self.opacity_accum[keep_mask]
                    if int(self.anchor_demon.shape[0]) == int(prune_mask.shape[0]):
                        self.anchor_demon = self.anchor_demon[keep_mask]
                    self.prune_anchor(prune_mask)
                    self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
                    return
                repeated_prune_mask = prune_mask.unsqueeze(1).repeat([1, self.n_offsets]).reshape(-1)
                if int(repeated_prune_mask.shape[0]) == int(self.offset_denom.shape[0]):
                    self.offset_denom[repeated_prune_mask] = 0
                    self.offset_gradient_accum[repeated_prune_mask] = 0
                self.opacity_accum[prune_mask] = 0
                self.anchor_demon[prune_mask] = 0
                self.mark_temporal_anchor_death(
                    prune_mask,
                    int(self.current_time_step if death_timestep is None else death_timestep),
                )
                self.update_temporal_split_suppression_from_deaths(
                    prune_mask,
                    suppress_neighbor_radius,
                )

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def get_mlp_checkpoint_payload(self):
        if (
            self.base_temporal_state is not None
            and int(self.base_temporal_state.latent.shape[0]) != int(self.get_anchor.shape[0])
        ):
            raise RuntimeError(
                "base_temporal_state rows do not match anchor rows while saving checkpoint: "
                f"{int(self.base_temporal_state.latent.shape[0])} vs {int(self.get_anchor.shape[0])}"
            )
        payload = {
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'temporal_history_mlps': {
                key: module.state_dict() for key, module in self.temporal_history_mlps.items()
            },
            'temporal_opacity_mlps': {
                key: module.state_dict() for key, module in self.temporal_opacity_mlps.items()
            },
            'temporal_cov_mlps': {
                key: module.state_dict() for key, module in self.temporal_cov_mlps.items()
            },
            'temporal_color_mlps': {
                key: module.state_dict() for key, module in self.temporal_color_mlps.items()
            },
        }
        if self.use_feat_bank:
            payload['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            payload['appearance'] = self.embedding_appearance.state_dict()
        return payload

    def load_mlp_checkpoint_payload(self, checkpoint):
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.temporal_enabled and 'base_temporal_latent' in checkpoint:
            latent_payload = checkpoint['base_temporal_latent']
            self.temporal_num_times = max(int(latent_payload.get('num_times', self.temporal_num_times)), 1)
            self.temporal_block_dim = max(int(latent_payload.get('chunk_dim', self.temporal_block_dim)), 1)
            latent_tensor = self._unpack_sparse_latent_rows(
                latent_payload.get('latent', None),
                expected_num_anchors=int(self.get_anchor.shape[0]),
                expected_latent_dim=int(latent_payload.get('latent_dim', self.temporal_latent_dim)),
                device=self.get_anchor.device,
                dtype=self.get_anchor.dtype,
            )
            if latent_tensor is not None:
                self.temporal_latent_dim = int(latent_tensor.shape[1])
                self.base_temporal_state = self._build_temporal_state_from_tensor(latent_tensor)
        for kind, field_name in (
            ("history", "temporal_history_mlps"),
            ("opacity", "temporal_opacity_mlps"),
            ("cov", "temporal_cov_mlps"),
            ("color", "temporal_color_mlps"),
        ):
            payload_dict = checkpoint.get(field_name, {})
            if not isinstance(payload_dict, dict):
                continue
            for key in sorted(payload_dict.keys(), key=lambda value: int(value)):
                time_step = int(key)
                if time_step <= 0:
                    continue
                self._ensure_temporal_modules_for_time(time_step)
                module = None
                if kind == "history":
                    module = self.temporal_history_mlps[key] if key in self.temporal_history_mlps else None
                else:
                    module_dict = self._get_temporal_property_dict(kind)
                    module = module_dict[key] if key in module_dict else None
                if module is not None:
                    module.load_state_dict(payload_dict[key])
        if self.use_feat_bank and 'feature_bank_mlp' in checkpoint:
            self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
        if self.appearance_dim > 0 and 'appearance' in checkpoint:
            appearance_state = checkpoint.get('appearance', {})
            weight = appearance_state.get('embedding.weight', None)
            if weight is None:
                weight = appearance_state.get('weight', None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                target_cams = int(self.embedding_appearance.embedding.weight.shape[0]) if self.embedding_appearance is not None else int(weight.shape[0])
                if self.embedding_appearance is None or int(self.embedding_appearance.embedding.weight.shape[0]) != target_cams:
                    self.embedding_appearance = Embedding(target_cams, self.appearance_dim).cuda()
                current_weight = self.embedding_appearance.embedding.weight.data
                loaded_weight = weight.to(current_weight.device)
                copy_rows = min(current_weight.shape[0], loaded_weight.shape[0])
                current_weight[:copy_rows].copy_(loaded_weight[:copy_rows])
                if current_weight.shape[0] > loaded_weight.shape[0]:
                    current_weight[loaded_weight.shape[0]:].zero_()
            else:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'unite':
            torch.save(self.get_mlp_checkpoint_payload(), os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError("Split MLP checkpoint export is not supported for per-time temporal MLPs.")


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.load_mlp_checkpoint_payload(checkpoint)
        else:
            raise NotImplementedError("Split MLP checkpoint import is not supported for per-time temporal MLPs.")
