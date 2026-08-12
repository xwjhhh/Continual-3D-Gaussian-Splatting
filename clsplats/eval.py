#!/usr/bin/env python3
"""
Evaluate a CL-Splats checkpoint on test split views.

Supported evaluation paths in this simplified version:
  - normal 3DGS / pure-3dgs checkpoints
  - public CL-Splats checkpoints via their native gsplat render chain
  - 3dgs_hash / idea3_hash checkpoints via scene_eval_bank + stage2_color_state
  - IRC-GS / Scaffold-GS checkpoints via Scaffold-GS anchor + MLP render chain
  - 4DGS wrapper checkpoints via the original 4DGaussians render chain
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


def _sanitize_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = str(os.environ.get(name, "")).strip()
        if value and (not value.isdigit() or int(value) < 1):
            os.environ[name] = "1"


_sanitize_thread_env()

import omegaconf
import numpy as np
from PIL import Image as PILImage
from plyfile import PlyData
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from clsplats.dataset import CLSplatsDataset
from clsplats.models.irc_gs import _load_scaffold_mlp_payload
from clsplats.representation.model_factory import (
    create_model as create_representation_model,
    get_available_models as get_available_representations,
)
from clsplats.rendering import render
from clsplats.utils.hexplane_utils import (
    init_grid_param,
    interpolate_ms_features,
    normalize_aabb,
)
from clsplats.utils.loss_utils import l1_loss, ssim

DEFAULT_TEMPORAL_LATENT_DIM = 8  # per-timestep temporal latent block dim

try:
    from clsplats.utils.deformation_utils import (
        build_spatial_kplanes_config,
        normalize_spatial_kplanes_config,
    )
except ImportError:
    def build_spatial_kplanes_config(feat_dim: int, resolution: int) -> Dict[str, Any]:
        res = int(resolution)
        return {
            "grid_dimensions": 2,
            "input_coordinate_dim": 3,
            "output_coordinate_dim": int(feat_dim),
            "resolution": [res, res, res],
        }

    def normalize_spatial_kplanes_config(kplanes_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(kplanes_config, dict):
            raise TypeError(
                "Spatial tri-plane kplanes_config must be a dict, "
                f"got {type(kplanes_config).__name__}."
            )
        grid_dimensions = int(kplanes_config.get("grid_dimensions", 2))
        input_coordinate_dim = int(kplanes_config.get("input_coordinate_dim", 3))
        output_coordinate_dim = int(kplanes_config.get("output_coordinate_dim", 32))
        resolution = [int(v) for v in kplanes_config.get("resolution", [])]
        if grid_dimensions != 2:
            raise ValueError(
                "Spatial tri-plane field expects grid_dimensions=2 for xy/xz/yz planes, "
                f"got {grid_dimensions}."
            )
        if input_coordinate_dim != 3:
            raise ValueError(
                "Spatial tri-plane field expects input_coordinate_dim=3 for xyz inputs, "
                f"got {input_coordinate_dim}."
            )
        if len(resolution) != input_coordinate_dim:
            raise ValueError(
                "Spatial tri-plane field expects resolution length to match "
                f"input_coordinate_dim ({input_coordinate_dim}), got {resolution}."
            )
        return {
            "grid_dimensions": grid_dimensions,
            "input_coordinate_dim": input_coordinate_dim,
            "output_coordinate_dim": output_coordinate_dim,
            "resolution": resolution,
        }


_SCAFFOLD_RUNTIME: Optional[Dict[str, Any]] = None
_FOURDGS_RUNTIME: Optional[Dict[str, Any]] = None
_FOURDGS_EVAL_CONTEXT_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


class _EvalSpatialTriPlaneField(nn.Module):
    """Evaluation-time spatial tri-plane field for 3dgs_hash stage2 color."""

    def __init__(
        self,
        bounds: float,
        feat_dim: Optional[int],
        resolution: Optional[int],
        multires: Tuple[int, ...],
        kplanes_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        aabb = torch.tensor(
            [[bounds, bounds, bounds], [-bounds, -bounds, -bounds]],
            dtype=torch.float32,
        )
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        if kplanes_config is None:
            if feat_dim is None or resolution is None:
                raise ValueError(
                    "_EvalSpatialTriPlaneField requires kplanes_config or both "
                    "feat_dim and resolution."
                )
            kplanes_config = build_spatial_kplanes_config(
                feat_dim=int(feat_dim),
                resolution=int(resolution),
            )
        self.grid_config = [normalize_spatial_kplanes_config(dict(kplanes_config))]
        self.multires = tuple(int(v) for v in multires)
        self.concat_features = True
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res_mul in self.multires:
            config = self.grid_config[0].copy()
            config["resolution"] = [int(r) * int(res_mul) for r in config["resolution"][:3]]
            gp = init_grid_param(
                grid_nd=int(config["grid_dimensions"]),
                in_dim=int(config["input_coordinate_dim"]),
                out_dim=int(config["output_coordinate_dim"]),
                reso=config["resolution"],
            )
            self.feat_dim += int(gp[-1].shape[1])
            self.grids.append(gp)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        pts = normalize_aabb(pts, self.aabb).reshape(-1, 3)
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,
            grid_dimensions=int(self.grid_config[0]["grid_dimensions"]),
            concat_features=self.concat_features,
            num_levels=None,
        )
        if len(features) < 1:
            return torch.zeros((0, 1), device=pts.device, dtype=torch.float32)
        return features


class _EvalStage2ColorMLP(nn.Module):
    """Evaluation-time color head matching current 3dgs_hash stage2 color branch."""

    def __init__(
        self,
        bounds: float,
        plane_feat_dim: Optional[int],
        plane_resolution: Optional[int],
        multires: Tuple[int, ...],
        hidden_dim: int,
        depth: int,
        kplanes_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.grid = _EvalSpatialTriPlaneField(
            bounds=float(bounds),
            feat_dim=None if plane_feat_dim is None else int(plane_feat_dim),
            resolution=None if plane_resolution is None else int(plane_resolution),
            multires=tuple(int(v) for v in multires),
            kplanes_config=kplanes_config,
        )
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.feature_reducer = self._build_feature_reducer(
            in_dim=int(self.grid.feat_dim),
            hidden_dim=int(self.hidden_dim),
            depth=max(1, int(self.depth)),
        )
        self.color_mlp = self._build_color_mlp(
            hidden_dim=int(self.hidden_dim),
            depth=max(1, int(self.depth)),
        )

    @staticmethod
    def _build_feature_reducer(in_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
        layers = [nn.Linear(int(in_dim), int(hidden_dim))]
        for _ in range(max(0, int(depth) - 1)):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(int(hidden_dim), int(hidden_dim)))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_color_mlp(hidden_dim: int, depth: int) -> nn.Sequential:
        layers = [
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim) + 3, int(hidden_dim)),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(int(hidden_dim), int(hidden_dim)))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), 3),
                nn.Sigmoid(),
            ]
        )
        return nn.Sequential(*layers)

    def forward(self, pts: torch.Tensor, ob_view: torch.Tensor) -> torch.Tensor:
        reduced_feat = self.feature_reducer(self.grid(pts))
        return self.color_mlp(torch.cat([reduced_feat, ob_view], dim=-1))


class _GaussianExplicitView:
    """Read-only Gaussian view built from explicit tensors."""

    def __init__(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        active_sh_degree: int,
        anchor_ids: Optional[torch.Tensor] = None,
    ):
        self._xyz = xyz
        self._features = features
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        self.active_sh_degree = int(active_sh_degree)
        self.anchor_ids = anchor_ids

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation


def _cfg_get(container: Any, key: str, default):
    if container is None:
        return default
    try:
        value = container.get(key, default)
    except Exception:
        return default
    return default if value is None else value


def _resolve_keyed_value(container: Any, timestep: int):
    if not isinstance(container, dict):
        return None
    for key in (str(int(timestep)), int(timestep)):
        if key in container:
            return container[key]
    return None


def _load_module_state_dict_shape_safe(
    module: nn.Module,
    payload: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    current_state = module.state_dict()
    safe_state = {}
    loaded = 0
    skipped = 0
    for key, value in payload.items():
        if key not in current_state:
            skipped += 1
            continue
        current_tensor = current_state[key]
        if tuple(current_tensor.shape) != tuple(value.shape):
            skipped += 1
            continue
        safe_state[key] = value.to(device=current_tensor.device, dtype=current_tensor.dtype)
        loaded += 1
    if safe_state:
        module.load_state_dict(safe_state, strict=False)
    return loaded, skipped


def _infer_anchor_ids_from_xyz(
    xyz: torch.Tensor,
    anchor_xyz_bank: Optional[torch.Tensor],
    chunk: int = 16384,
) -> Optional[torch.Tensor]:
    if not isinstance(xyz, torch.Tensor) or not isinstance(anchor_xyz_bank, torch.Tensor):
        return None
    if xyz.ndim != 2 or int(xyz.shape[1]) != 3:
        return None
    if anchor_xyz_bank.ndim != 2 or int(anchor_xyz_bank.shape[1]) != 3:
        return None
    if int(xyz.shape[0]) <= 0 or int(anchor_xyz_bank.shape[0]) <= 0:
        return None
    xyz = xyz.to(dtype=torch.float32)
    anchor_xyz_bank = anchor_xyz_bank.to(device=xyz.device, dtype=torch.float32)
    out = []
    for start in range(0, int(xyz.shape[0]), int(chunk)):
        end = min(start + int(chunk), int(xyz.shape[0]))
        dists = torch.cdist(xyz[start:end], anchor_xyz_bank)
        out.append(torch.argmin(dists, dim=1).to(dtype=torch.long))
    if not out:
        return None
    return torch.cat(out, dim=0)


def _build_3dgs_hash_render_view(
    checkpoint: Dict[str, Any],
    device: str,
    timestep: int,
    fallback_model,
) -> Tuple[Optional[object], Optional[int], Optional[Dict[str, int]]]:
    """Restore current 3dgs_hash / idea3_hash eval-bank view."""
    idea_state = checkpoint.get("idea3_state", None)
    if not isinstance(idea_state, dict):
        idea_state = checkpoint.get("idea1_state", None)
    if not isinstance(idea_state, dict):
        return None, None, None

    base = idea_state.get("base_scene_snapshot", None)
    eval_bank = idea_state.get("scene_eval_bank", None)
    payload = _resolve_keyed_value(eval_bank, int(timestep))
    if not isinstance(base, dict) or not isinstance(payload, dict):
        return None, None, None

    required_base = ["xyz", "features_dc", "features_rest", "rotation", "scaling"]
    if any(base.get(k, None) is None for k in required_base):
        return None, None, None
    required_eval = ["base_ids", "base_opacity", "base_scaling"]
    if any(payload.get(k, None) is None for k in required_eval):
        return None, None, None

    xyz_all = torch.as_tensor(base["xyz"], device="cpu", dtype=torch.float32).reshape(-1, 3)
    f_dc_all = torch.as_tensor(base["features_dc"], device="cpu", dtype=torch.float32)
    f_rest_all = torch.as_tensor(base["features_rest"], device="cpu", dtype=torch.float32)
    rotation_raw_all = torch.as_tensor(base["rotation"], device="cpu", dtype=torch.float32).reshape(-1, 4)
    scaling_default_raw_all = torch.as_tensor(base["scaling"], device="cpu", dtype=torch.float32).reshape(-1, 3)
    base_anchor_ids_all = base.get("anchor_ids", None)
    if base_anchor_ids_all is not None:
        base_anchor_ids_all = torch.as_tensor(base_anchor_ids_all, device="cpu", dtype=torch.long).reshape(-1)

    anchor_state = idea_state.get("anchor_state", None)
    anchor_xyz_bank = None
    if isinstance(anchor_state, dict) and anchor_state.get("anchor_xyz", None) is not None:
        anchor_xyz_bank = torch.as_tensor(
            anchor_state["anchor_xyz"],
            device="cpu",
            dtype=torch.float32,
        ).reshape(-1, 3)

    n = int(xyz_all.shape[0])
    if n <= 0:
        return None, 0, None
    if (
        int(f_dc_all.shape[0]) != n
        or int(f_rest_all.shape[0]) != n
        or int(rotation_raw_all.shape[0]) != n
        or int(scaling_default_raw_all.shape[0]) != n
    ):
        return None, None, None

    base_ids = torch.as_tensor(payload["base_ids"], device="cpu", dtype=torch.long).reshape(-1)
    base_opacity_raw = torch.as_tensor(payload["base_opacity"], device="cpu", dtype=torch.float32).reshape(-1, 1)
    base_scaling_raw = torch.as_tensor(payload["base_scaling"], device="cpu", dtype=torch.float32).reshape(-1, 3)
    base_sh_ids = payload.get("base_sh_ids", None)
    if base_sh_ids is not None:
        base_sh_ids = torch.as_tensor(base_sh_ids, device="cpu", dtype=torch.long).reshape(-1)
    base_features_dc = payload.get("base_features_dc", None)
    if base_features_dc is not None:
        base_features_dc = torch.as_tensor(base_features_dc, device="cpu", dtype=torch.float32)
    base_features_rest = payload.get("base_features_rest", None)
    if base_features_rest is not None:
        base_features_rest = torch.as_tensor(base_features_rest, device="cpu", dtype=torch.float32)

    base_features_dc_full = f_dc_all.clone()
    base_features_rest_full = f_rest_all.clone()
    base_opacity_raw_full = torch.full((n, 1), -20.0, device="cpu", dtype=torch.float32)
    base_scaling_raw_full = scaling_default_raw_all.clone()
    base_active_mask = torch.zeros(n, device="cpu", dtype=torch.bool)

    m_base = min(
        int(base_ids.shape[0]),
        int(base_opacity_raw.shape[0]),
        int(base_scaling_raw.shape[0]),
    )
    if m_base > 0:
        base_ids = base_ids[:m_base]
        base_opacity_raw = base_opacity_raw[:m_base]
        base_scaling_raw = base_scaling_raw[:m_base]
        valid_base = torch.logical_and(base_ids >= 0, base_ids < n)
        base_ids = base_ids[valid_base]
        base_opacity_raw = base_opacity_raw[valid_base]
        base_scaling_raw = base_scaling_raw[valid_base]
        if int(base_ids.shape[0]) > 0:
            base_opacity_raw_full[base_ids] = base_opacity_raw
            base_scaling_raw_full[base_ids] = base_scaling_raw
            base_active_mask[base_ids] = True

    if base_sh_ids is None:
        base_sh_ids = base_ids
    if isinstance(base_sh_ids, torch.Tensor):
        sh_counts = [int(base_sh_ids.shape[0])]
        if isinstance(base_features_dc, torch.Tensor):
            sh_counts.append(int(base_features_dc.shape[0]))
        if isinstance(base_features_rest, torch.Tensor):
            sh_counts.append(int(base_features_rest.shape[0]))
        m_sh = min(sh_counts) if sh_counts else 0
        if m_sh > 0:
            base_sh_ids = base_sh_ids[:m_sh]
            if isinstance(base_features_dc, torch.Tensor):
                base_features_dc = base_features_dc[:m_sh]
            if isinstance(base_features_rest, torch.Tensor):
                base_features_rest = base_features_rest[:m_sh]
            valid_sh = torch.logical_and(base_sh_ids >= 0, base_sh_ids < n)
            base_sh_ids = base_sh_ids[valid_sh]
            if isinstance(base_features_dc, torch.Tensor):
                base_features_dc = base_features_dc[valid_sh]
            if isinstance(base_features_rest, torch.Tensor):
                base_features_rest = base_features_rest[valid_sh]
            if int(base_sh_ids.shape[0]) > 0:
                if isinstance(base_features_dc, torch.Tensor):
                    base_features_dc_full[base_sh_ids] = base_features_dc
                if isinstance(base_features_rest, torch.Tensor):
                    base_features_rest_full[base_sh_ids] = base_features_rest

    base_features_all = torch.cat([base_features_dc_full, base_features_rest_full], dim=1)

    local_required = (
        "local_xyz",
        "local_features_dc",
        "local_features_rest",
        "local_opacity",
        "local_scaling",
        "local_rotation",
    )
    local_anchor_ids_all = payload.get("local_anchor_ids", None)
    if local_anchor_ids_all is not None:
        local_anchor_ids_all = torch.as_tensor(local_anchor_ids_all, device="cpu", dtype=torch.long).reshape(-1)
    has_local = all(payload.get(k, None) is not None for k in local_required)
    if has_local:
        local_xyz_all = torch.as_tensor(payload["local_xyz"], device="cpu", dtype=torch.float32)
        local_fdc_all = torch.as_tensor(payload["local_features_dc"], device="cpu", dtype=torch.float32)
        local_frest_all = torch.as_tensor(payload["local_features_rest"], device="cpu", dtype=torch.float32)
        local_opacity_all = torch.as_tensor(payload["local_opacity"], device="cpu", dtype=torch.float32).reshape(-1, 1)
        local_scaling_all = torch.as_tensor(payload["local_scaling"], device="cpu", dtype=torch.float32).reshape(-1, 3)
        local_rotation_all = torch.as_tensor(payload["local_rotation"], device="cpu", dtype=torch.float32).reshape(-1, 4)

        m_local = min(
            int(local_xyz_all.shape[0]),
            int(local_fdc_all.shape[0]),
            int(local_frest_all.shape[0]),
            int(local_opacity_all.shape[0]),
            int(local_scaling_all.shape[0]),
            int(local_rotation_all.shape[0]),
        )
        if m_local > 0:
            local_xyz = local_xyz_all[:m_local]
            local_features = torch.cat([local_fdc_all[:m_local], local_frest_all[:m_local]], dim=1)
            local_opacity_raw = local_opacity_all[:m_local]
            local_scaling_raw = local_scaling_all[:m_local]
            local_rotation_raw = local_rotation_all[:m_local]
            local_anchor_ids = None
            if isinstance(local_anchor_ids_all, torch.Tensor) and int(local_anchor_ids_all.shape[0]) >= m_local:
                local_anchor_ids = local_anchor_ids_all[:m_local]
        else:
            local_xyz = torch.empty((0, 3), dtype=torch.float32, device="cpu")
            local_features = torch.empty(
                (0, int(base_features_all.shape[1]), int(base_features_all.shape[2])),
                dtype=torch.float32,
                device="cpu",
            )
            local_opacity_raw = torch.empty((0, 1), dtype=torch.float32, device="cpu")
            local_scaling_raw = torch.empty((0, 3), dtype=torch.float32, device="cpu")
            local_rotation_raw = torch.empty((0, 4), dtype=torch.float32, device="cpu")
            local_anchor_ids = torch.empty((0,), dtype=torch.long, device="cpu")
    else:
        local_xyz = torch.empty((0, 3), dtype=torch.float32, device="cpu")
        local_features = torch.empty(
            (0, int(base_features_all.shape[1]), int(base_features_all.shape[2])),
            dtype=torch.float32,
            device="cpu",
        )
        local_opacity_raw = torch.empty((0, 1), dtype=torch.float32, device="cpu")
        local_scaling_raw = torch.empty((0, 3), dtype=torch.float32, device="cpu")
        local_rotation_raw = torch.empty((0, 4), dtype=torch.float32, device="cpu")
        local_anchor_ids = torch.empty((0,), dtype=torch.long, device="cpu")

    local_total = int(local_xyz.shape[0])
    raw_trainable_local_count = payload.get("trainable_local_count", None)
    trainable_local_count = 0
    if isinstance(raw_trainable_local_count, torch.Tensor) and int(raw_trainable_local_count.numel()) > 0:
        try:
            trainable_local_count = int(raw_trainable_local_count.reshape(-1)[0].item())
        except Exception:
            trainable_local_count = 0
    elif raw_trainable_local_count is not None:
        try:
            trainable_local_count = int(raw_trainable_local_count)
        except Exception:
            trainable_local_count = 0
    else:
        trainable_local_count = int(local_total)
    trainable_local_count = max(0, min(int(trainable_local_count), int(local_total)))
    historical_local_count = int(local_total - trainable_local_count)

    xyz = torch.cat([xyz_all, local_xyz], dim=0).to(device)
    features = torch.cat([base_features_all, local_features], dim=0).to(device)
    rotation_raw = torch.cat([rotation_raw_all, local_rotation_raw], dim=0).to(device)
    scaling_raw = torch.cat([base_scaling_raw_full, local_scaling_raw], dim=0).to(device)
    opacity_raw = torch.cat([base_opacity_raw_full, local_opacity_raw], dim=0).to(device)
    anchor_ids = None
    if (
        isinstance(base_anchor_ids_all, torch.Tensor)
        and int(base_anchor_ids_all.shape[0]) == n
        and int(local_anchor_ids.shape[0]) == int(local_xyz.shape[0])
    ):
        anchor_ids = torch.cat([base_anchor_ids_all, local_anchor_ids], dim=0).to(device)
    if anchor_ids is None:
        anchor_ids = _infer_anchor_ids_from_xyz(xyz, anchor_xyz_bank)

    if hasattr(fallback_model, "rotation_activation"):
        rotation = fallback_model.rotation_activation(rotation_raw)
    else:
        rotation = F.normalize(rotation_raw, dim=-1)
    if hasattr(fallback_model, "scaling_activation"):
        scaling = fallback_model.scaling_activation(scaling_raw)
    else:
        scaling = torch.exp(scaling_raw)
    if hasattr(fallback_model, "opacity_activation"):
        opacity = fallback_model.opacity_activation(opacity_raw)
    else:
        opacity = torch.sigmoid(opacity_raw)

    active_sh_degree = int(getattr(fallback_model, "active_sh_degree", 0))
    raw_degree = base.get("active_sh_degree", None)
    if raw_degree is not None:
        raw_degree_t = torch.as_tensor(raw_degree).reshape(-1)
        if raw_degree_t.numel() > 0:
            active_sh_degree = int(raw_degree_t[0].item())

    split_meta = {
        "rows_total": int(xyz.shape[0]),
        "base_like_rows": int(n + historical_local_count),
        "local_trainable_rows": int(trainable_local_count),
    }
    visible = int(base_active_mask.sum().item()) + int(local_xyz.shape[0])
    if visible == 0:
        return None, 0, split_meta

    return (
        _GaussianExplicitView(
            xyz,
            features,
            opacity,
            scaling,
            rotation,
            active_sh_degree,
            anchor_ids=anchor_ids,
        ),
        visible,
        split_meta,
    )


def _build_stage2_color_eval_override(
    checkpoint: Dict[str, Any],
    device: torch.device,
    camera,
    render_view,
) -> Optional[torch.Tensor]:
    if not hasattr(render_view, "get_xyz"):
        return None
    xyz = render_view.get_xyz
    if not isinstance(xyz, torch.Tensor):
        return None

    idea_state = checkpoint.get("idea3_state", None)
    if not isinstance(idea_state, dict):
        idea_state = checkpoint.get("idea1_state", None)
    if not isinstance(idea_state, dict):
        return None

    stage2_color_state = idea_state.get("stage2_color_state", None)
    if not isinstance(stage2_color_state, dict):
        return None
    module_state = stage2_color_state.get("module", None)
    if not isinstance(module_state, dict):
        return None

    train_cfg = _cfg_get(_cfg_get(checkpoint, "config", {}), "train", {}) or {}
    bounds = float(_cfg_get(train_cfg, "idea3_stage2_hexplane_bounds", _cfg_get(train_cfg, "stage2_hexplane_bounds", 1.6)))
    plane_feat_dim = int(_cfg_get(train_cfg, "idea3_stage2_hexplane_feat_dim", _cfg_get(train_cfg, "stage2_hexplane_feat_dim", 32)))
    plane_resolution = int(_cfg_get(train_cfg, "idea3_stage2_hexplane_resolution", _cfg_get(train_cfg, "stage2_hexplane_resolution", 64)))
    hidden_dim = int(_cfg_get(train_cfg, "idea3_stage2_hexplane_mlp_width", _cfg_get(train_cfg, "stage2_hexplane_mlp_width", 64)))
    depth = int(_cfg_get(train_cfg, "idea3_stage2_hexplane_mlp_depth", _cfg_get(train_cfg, "stage2_hexplane_mlp_depth", 1)))
    kplanes_config = _cfg_get(
        train_cfg,
        "idea3_stage2_kplanes_config",
        _cfg_get(train_cfg, "stage2_kplanes_config", None),
    )
    if isinstance(kplanes_config, dict):
        kplanes_config = normalize_spatial_kplanes_config(dict(kplanes_config))
        plane_feat_dim = int(kplanes_config["output_coordinate_dim"])
        plane_resolution = int(kplanes_config["resolution"][0])
    else:
        kplanes_config = build_spatial_kplanes_config(
            feat_dim=int(plane_feat_dim),
            resolution=int(plane_resolution),
        )
    multires_cfg = _cfg_get(
        train_cfg,
        "idea3_stage2_multires",
        _cfg_get(
            train_cfg,
            "stage2_multires",
            _cfg_get(train_cfg, "idea3_stage2_hexplane_multires", _cfg_get(train_cfg, "stage2_hexplane_multires", (1, 2, 4, 8))),
        ),
    )
    try:
        multires = tuple(int(v) for v in multires_cfg)
    except Exception:
        multires = (1, 2, 4, 8)

    module = _EvalStage2ColorMLP(
        bounds=bounds,
        plane_feat_dim=plane_feat_dim,
        plane_resolution=plane_resolution,
        multires=multires,
        hidden_dim=hidden_dim,
        depth=depth,
        kplanes_config=kplanes_config,
    ).to(device)
    safe_payload = {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in module_state.items()
    }
    loaded, _ = _load_module_state_dict_shape_safe(module, safe_payload)
    if loaded <= 0:
        return None

    module.eval()
    with torch.no_grad():
        xyz = xyz.to(device=device, dtype=torch.float32)
        cam_center = camera.camera_center.to(device=device, dtype=torch.float32).view(1, 3)
        ob_view = xyz - cam_center
        ob_dist = ob_view.norm(dim=1, keepdim=True).clamp_min(1e-8)
        ob_view = ob_view / ob_dist
        rgb = module(xyz, ob_view)
        if int(rgb.shape[0]) != int(xyz.shape[0]) or int(rgb.shape[-1]) != 3:
            return None
    return rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CL-Splats checkpoints.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset root path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt path.")
    parser.add_argument("--timestep", type=int, default=None, help="Timestep to evaluate.")
    parser.add_argument(
        "--timesteps",
        type=str,
        default=None,
        help="Evaluate multiple timesteps in one process, e.g. 0,1,2 or 0-9.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--white-background", action="store_true", help="Use white background.")
    parser.add_argument("--prefer-undist", action="store_true", default=True)
    parser.add_argument("--prefer-dist", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--allow-resize-mismatch", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--no-save-gt-images", action="store_true", help="Do not save GT test images during eval.")
    parser.add_argument("--eval-train-split", action="store_true", help="Also report metrics on train-split views.")
    parser.add_argument("--benchmark-fps", action="store_true", help="Measure render throughput on test views.")
    parser.add_argument("--benchmark-warmup", type=int, default=5, help="Warmup renders before FPS timing.")
    parser.add_argument("--benchmark-repeat", type=int, default=10, help="Number of timed passes over test views.")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--traceback", action="store_true", help="Print full traceback on failure.")
    return parser


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")
    return device_arg


def resolve_ply_path(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> Path:
    candidates = []
    raw_ply = checkpoint.get("ply_path", None)
    if raw_ply:
        raw_p = Path(str(raw_ply))
        if raw_p.is_absolute():
            candidates.append(raw_p)
        else:
            candidates.append((Path.cwd() / raw_p).resolve())
            candidates.append((checkpoint_path.parent / raw_p).resolve())
    candidates.append(checkpoint_path.with_suffix(".ply"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve PLY path from checkpoint. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def _ensure_scaffold_optional_deps() -> None:
    try:
        import colorama  # noqa: F401
    except ModuleNotFoundError:
        import types

        class _ColorStub:
            def __getattr__(self, _name: str) -> str:
                return ""

        colorama_stub = types.ModuleType("colorama")
        colorama_stub.Fore = _ColorStub()
        colorama_stub.Style = _ColorStub()

        def _noop(*_args, **_kwargs):
            return None

        colorama_stub.init = _noop
        sys.modules["colorama"] = colorama_stub


def _load_scaffold_runtime() -> Dict[str, Any]:
    global _SCAFFOLD_RUNTIME
    if _SCAFFOLD_RUNTIME is not None:
        return _SCAFFOLD_RUNTIME

    scaffold_root = Path(__file__).resolve().parents[1] / "Scaffold-GS-main"
    if not scaffold_root.is_dir():
        raise FileNotFoundError(
            "Scaffold-GS-main not found. Expected path: "
            f"{str(scaffold_root)}"
        )
    scaffold_root_str = str(scaffold_root)
    if scaffold_root_str not in sys.path:
        sys.path.insert(0, scaffold_root_str)

    _ensure_scaffold_optional_deps()

    renderer_module = importlib.import_module("gaussian_renderer")
    gaussian_model_module = importlib.import_module("scene.gaussian_model")
    _SCAFFOLD_RUNTIME = {
        "prefilter_voxel": getattr(renderer_module, "prefilter_voxel"),
        "render": getattr(renderer_module, "render"),
        "GaussianModel": getattr(gaussian_model_module, "GaussianModel"),
    }
    return _SCAFFOLD_RUNTIME


def _create_scaffold_gaussians(model_cfg: Dict[str, Any]):
    runtime = _load_scaffold_runtime()
    gaussian_model_cls = runtime["GaussianModel"]
    temporal_enabled = int(model_cfg.get("num_times", 1)) > 1
    static_view_stream = bool(model_cfg.get("static_view_stream", False))
    if static_view_stream:
        appearance_dim = 0
    elif "appearance_dim" in model_cfg:
        appearance_dim = int(model_cfg.get("appearance_dim", 32))
    else:
        appearance_dim = 0 if temporal_enabled else 32
    return gaussian_model_cls(
        feat_dim=int(model_cfg.get("feat_dim", 32)),
        n_offsets=int(model_cfg.get("n_offsets", 10)),
        voxel_size=float(model_cfg.get("voxel_size", 0.001)),
        update_depth=int(model_cfg.get("update_depth", 3)),
        update_init_factor=int(model_cfg.get("update_init_factor", 16)),
        update_hierachy_factor=int(model_cfg.get("update_hierachy_factor", 4)),
        use_feat_bank=bool(model_cfg.get("use_feat_bank", False)),
        appearance_dim=appearance_dim,
        ratio=int(model_cfg.get("ratio", 1)),
        add_opacity_dist=bool(model_cfg.get("add_opacity_dist", False)),
        add_cov_dist=bool(model_cfg.get("add_cov_dist", False)),
        add_color_dist=bool(model_cfg.get("add_color_dist", False)),
        temporal_latent_dim=int(model_cfg.get("temporal_latent_dim", DEFAULT_TEMPORAL_LATENT_DIM)),
        temporal_num_times=(1 if static_view_stream else int(model_cfg.get("num_times", 1))),
        temporal_mode=str(model_cfg.get("temporal_mode", "per_anchor")),
    )


def _apply_checkpoint_temporal_layout_defaults(
    checkpoint: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    effective_model_cfg = dict(model_cfg)

    if "appearance_dim" not in effective_model_cfg and checkpoint.get("temporal_enabled", False):
        effective_model_cfg["appearance_dim"] = 0

    ckpt_cfg = checkpoint.get("config", {}) or {}
    ckpt_train_cfg = ckpt_cfg.get("train", {}) if isinstance(ckpt_cfg, dict) else {}
    if "num_times" not in effective_model_cfg and isinstance(ckpt_train_cfg, dict):
        effective_model_cfg["num_times"] = int(ckpt_train_cfg.get("num_times", 1))

    static_view_stream = bool(
        checkpoint.get(
            "static_view_stream",
            ckpt_train_cfg.get("static_view_stream", False)
            if isinstance(ckpt_train_cfg, dict)
            else False,
        )
    )
    effective_model_cfg["static_view_stream"] = static_view_stream
    if static_view_stream:
        effective_model_cfg["appearance_dim"] = 0
        return effective_model_cfg

    mlp_state = checkpoint.get("mlp_state", None)
    latent_payload = checkpoint.get("base_temporal_latent", None)
    if latent_payload is None and isinstance(mlp_state, dict):
        latent_payload = mlp_state.get("base_temporal_latent", None)
    if isinstance(latent_payload, dict):
        chunk_dim = latent_payload.get("chunk_dim", None)
        payload_num_times = latent_payload.get("num_times", None)
        if chunk_dim is not None:
            effective_model_cfg["temporal_latent_dim"] = int(chunk_dim)
        if payload_num_times is not None:
            effective_model_cfg["num_times"] = int(payload_num_times)

    per_time_opacity = mlp_state.get("temporal_opacity_mlps", {}) if isinstance(mlp_state, dict) else {}
    if isinstance(per_time_opacity, dict) and per_time_opacity:
        max_time = max(int(key) for key in per_time_opacity.keys())
        effective_model_cfg["num_times"] = max(int(effective_model_cfg.get("num_times", 1)), max_time + 1)
    opacity_state = mlp_state.get("opacity_mlp", {}) if isinstance(mlp_state, dict) else {}
    first_layer = None
    temporal_head_layer = None
    if isinstance(per_time_opacity, dict) and per_time_opacity:
        latest_key = str(max(int(key) for key in per_time_opacity.keys()))
        latest_state = per_time_opacity.get(latest_key, {})
        if isinstance(latest_state, dict):
            temporal_head_layer = latest_state.get("0.weight", None)
    if temporal_head_layer is not None:
        first_layer = temporal_head_layer
    elif isinstance(opacity_state, dict):
        first_layer = opacity_state.get("0.weight", None)
    if hasattr(first_layer, "shape") and len(first_layer.shape) == 2:
        feat_dim = int(effective_model_cfg.get("feat_dim", 32))
        opacity_dist_dim = 1 if bool(effective_model_cfg.get("add_opacity_dist", False)) else 0
        inferred_temporal_block_dim = int(first_layer.shape[1]) - feat_dim - 3 - opacity_dist_dim
        if inferred_temporal_block_dim >= 0:
            num_times = max(int(effective_model_cfg.get("num_times", 1)), 1)
            if temporal_head_layer is not None:
                if inferred_temporal_block_dim > 0:
                    effective_model_cfg["temporal_latent_dim"] = int(inferred_temporal_block_dim)
                    effective_model_cfg["num_times"] = max(num_times, 2)
            elif inferred_temporal_block_dim == 0:
                effective_model_cfg["temporal_latent_dim"] = 0
                effective_model_cfg["num_times"] = 1
            elif "temporal_latent_dim" not in effective_model_cfg:
                # Base t0 MLP in the new per-time design only sees the current block z_t,
                # so its temporal input width equals block_dim, not num_times * block_dim.
                effective_model_cfg["temporal_latent_dim"] = inferred_temporal_block_dim

    return effective_model_cfg


def _build_scaffold_temporal_training_args(checkpoint: Dict[str, Any], model_cfg: Dict[str, Any]):
    train_cfg = {}
    ckpt_cfg = checkpoint.get("config", {})
    if isinstance(ckpt_cfg, dict):
        train_cfg = ckpt_cfg.get("train", {}) or {}
    return SimpleNamespace(
        latent_lr=float(train_cfg.get("latent_lr", 1e-3)),
        mlp_temporal_bank_lr_init=float(train_cfg.get("mlp_temporal_bank_lr_init", 0.01)),
        mlp_opacity_lr_init=float(train_cfg.get("mlp_opacity_lr_init", 0.002)),
        mlp_cov_lr_init=float(train_cfg.get("mlp_cov_lr_init", 0.004)),
        mlp_color_lr_init=float(train_cfg.get("mlp_color_lr_init", 0.008)),
    )


def _lookup_temporal_payload(temporal_payloads: Dict[Any, Any], timestep: int) -> Optional[Dict[str, Any]]:
    payload = temporal_payloads.get(int(timestep), None)
    if payload is None:
        payload = temporal_payloads.get(str(int(timestep)), None)
    return payload if isinstance(payload, dict) else None


def _lookup_lifetime_metadata_payload(
    temporal_payloads: Dict[Any, Any],
    timestep: int,
) -> Optional[Dict[str, Any]]:
    exact_payload = _lookup_temporal_payload(temporal_payloads, timestep)
    if exact_payload is not None:
        return exact_payload

    keyed_payloads = []
    for key, payload in temporal_payloads.items():
        if not isinstance(payload, dict):
            continue
        try:
            payload_time = int(payload.get("time_step", key))
        except Exception:
            continue
        keyed_payloads.append((payload_time, payload))
    if not keyed_payloads:
        return None

    later_payloads = [(time_step, payload) for time_step, payload in keyed_payloads if time_step >= int(timestep)]
    if later_payloads:
        return min(later_payloads, key=lambda item: item[0])[1]
    return max(keyed_payloads, key=lambda item: item[0])[1]


def _build_sfgs_eval_model(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: str,
    timestep: int,
):
    if device != "cuda":
        raise RuntimeError(
        "IRC-GS / Scaffold-GS evaluation currently requires CUDA because the "
            "underlying Scaffold-GS modules construct CUDA tensors directly."
        )

    effective_model_cfg = _apply_checkpoint_temporal_layout_defaults(
        checkpoint=checkpoint,
        model_cfg=model_cfg,
    )
    logger.info(
        "[Eval] IRC-GS effective temporal layout: "
        f"num_times={int(effective_model_cfg.get('num_times', 1))}, "
        f"temporal_latent_dim={int(effective_model_cfg.get('temporal_latent_dim', DEFAULT_TEMPORAL_LATENT_DIM))}, "
        f"appearance_dim={int(effective_model_cfg.get('appearance_dim', 32)) if 'appearance_dim' in effective_model_cfg else 'default'}"
    )

    gaussians = _create_scaffold_gaussians(effective_model_cfg)
    ply_path = resolve_ply_path(checkpoint_path, checkpoint)
    mlp_dir = None

    gaussians.load_ply_sparse_gaussian(str(ply_path))

    mlp_state = checkpoint.get("mlp_state", None)
    if not isinstance(mlp_state, dict):
        raise KeyError(
            "Scaffold-GS checkpoint is missing embedded 'mlp_state'. "
            "Legacy _mlp sidecar directories are no longer supported."
        )
    if gaussians.appearance_dim > 0:
        appearance_state = mlp_state.get("appearance", {})
        weight = None
        if isinstance(appearance_state, dict):
            weight = appearance_state.get("weight", None)
            if weight is None:
                weight = appearance_state.get("embedding.weight", None)
            if weight is None:
                for value in appearance_state.values():
                    if hasattr(value, "shape") and len(value.shape) == 2:
                        weight = value
                        break
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            gaussians.set_appearance(int(weight.shape[0]))
    _load_scaffold_mlp_payload(gaussians, mlp_state)

    temporal_enabled = bool(checkpoint.get("temporal_enabled", int(effective_model_cfg.get("num_times", 1)) > 1))
    temporal_adapter_enabled = bool(
        checkpoint.get(
            "temporal_adapter_enabled",
            temporal_enabled and not bool(effective_model_cfg.get("static_view_stream", False)),
        )
    )
    temporal_payloads = checkpoint.get("temporal_adapter_payloads", {}) or {}
    if temporal_enabled:
        temporal_payload = _lookup_temporal_payload(temporal_payloads, int(timestep))
        lifetime_payload = _lookup_lifetime_metadata_payload(temporal_payloads, int(timestep))
        gaussians.current_time_step = int(timestep)
        if temporal_adapter_enabled:
            temporal_args = _build_scaffold_temporal_training_args(checkpoint, effective_model_cfg)
            if int(timestep) <= 0:
                logger.info(
                    "[Eval] Timestep 0 uses base Scaffold state only; skip temporal adapter restore."
                )
            elif temporal_payload is not None and hasattr(gaussians, "load_temporal_checkpoint_payload"):
                gaussians.load_temporal_checkpoint_payload(temporal_payload, temporal_args, time_step=int(timestep))
                logger.info(
                    f"[Eval] Loaded embedded temporal adapter for timestep {int(timestep)} from checkpoint payload"
                )
            elif int(timestep) > 0:
                logger.warning(
                    f"[Eval] Missing embedded temporal adapter for timestep {int(timestep)}; "
                    "rendering with base Scaffold temporal metadata only."
                )
        else:
            logger.info(
                f"[Eval] Static view-stream timestep {int(timestep)} uses its isolated "
                "property-MLP head without a temporal latent."
            )
        birth_timestep = None
        death_timestep = None
        lifetime_source = "none"
        if isinstance(lifetime_payload, dict):
            birth_timestep = lifetime_payload.get("birth_timestep", None)
            death_timestep = lifetime_payload.get("death_timestep", None)
            if birth_timestep is not None or death_timestep is not None:
                lifetime_source = "payload"
        if birth_timestep is None:
            birth_timestep = checkpoint.get("anchor_birth_timestep", None)
            if birth_timestep is not None and lifetime_source == "none":
                lifetime_source = "checkpoint"
        if death_timestep is None:
            death_timestep = checkpoint.get("anchor_death_timestep", None)
            if death_timestep is not None and lifetime_source == "none":
                lifetime_source = "checkpoint"
        logger.info(
            f"[Eval] Temporal lifetime metadata source for timestep {int(timestep)}: {lifetime_source}"
        )
        if birth_timestep is not None:
            birth_timestep = torch.as_tensor(birth_timestep, device=device, dtype=torch.long).reshape(-1)
            n_anchors = int(gaussians.get_anchor.shape[0])
            if int(birth_timestep.shape[0]) == n_anchors:
                gaussians.temporal_anchor_birth_timestep = birth_timestep
                logger.info(
                    f"[Eval] Restored anchor birth timesteps for timestep {int(timestep)}"
                )
        if death_timestep is not None:
            death_timestep = torch.as_tensor(death_timestep, device=device, dtype=torch.long).reshape(-1)
            n_anchors = int(gaussians.get_anchor.shape[0])
            if int(death_timestep.shape[0]) == n_anchors:
                gaussians.temporal_anchor_death_timestep = death_timestep
                logger.info(
                    f"[Eval] Restored anchor death timesteps for timestep {int(timestep)}"
                )
        local_mask = checkpoint.get("temporal_local_mask", None)
        if local_mask is not None:
            local_mask = torch.as_tensor(local_mask, device=device, dtype=torch.bool).reshape(-1)
            n_anchors = int(gaussians.get_anchor.shape[0])
            if int(local_mask.shape[0]) == n_anchors:
                gaussians.temporal_local_mask = local_mask
        local_parent_ids = checkpoint.get("temporal_local_parent_ids", None)
        if local_parent_ids is not None:
            local_parent_ids = torch.as_tensor(local_parent_ids, device=device, dtype=torch.long).reshape(-1)
            n_anchors = int(gaussians.get_anchor.shape[0])
            if int(local_parent_ids.shape[0]) == n_anchors:
                gaussians.temporal_local_parent_ids = local_parent_ids
        split_suppression_mask = checkpoint.get("temporal_split_suppression_mask", None)
        if split_suppression_mask is not None:
            split_suppression_mask = torch.as_tensor(split_suppression_mask, device=device, dtype=torch.bool).reshape(-1)
            n_anchors = int(gaussians.get_anchor.shape[0])
            if int(split_suppression_mask.shape[0]) == n_anchors:
                gaussians.temporal_split_suppression_mask = split_suppression_mask
        if hasattr(gaussians, "get_temporal_render_mask"):
            render_mask = gaussians.get_temporal_render_mask(time_step=int(timestep))
            gaussians._eval_temporal_visible_anchors = int(render_mask.sum().item())
            gaussians._eval_temporal_total_anchors = int(render_mask.numel())
            logger.info(
                f"[Eval] IRC-GS temporal render mask for timestep {int(timestep)}: "
                f"visible_anchors={int(render_mask.sum().item())}/{int(render_mask.numel())}"
            )
    # Scaffold-GS uses module.training as a renderer branch switch.  Keep the
    # same forward path as training/visualization; evaluate under no_grad.
    gaussians.train()
    return gaussians, ply_path, mlp_dir


def _render_sfgs_view(camera, gaussians, bg_color: torch.Tensor) -> torch.Tensor:
    runtime = _load_scaffold_runtime()
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    visible_mask = runtime["prefilter_voxel"](camera, gaussians, pipe, bg_color)
    render_pkg = runtime["render"](
        camera,
        gaussians,
        pipe,
        bg_color,
        visible_mask=visible_mask,
    )
    return render_pkg["render"]


def _is_4dgs_model(model_name: str, checkpoint: Dict[str, Any]) -> bool:
    if model_name in {"4dgs", "4dgs_all_timesteps", "four_d_gs"}:
        return True
    trainer_type = str(checkpoint.get("trainer_type", "")).strip().lower()
    return trainer_type in {"4dgs_all_timesteps", "4dgs", "four_d_gs"}


def _resolve_4dgs_root(checkpoint: Dict[str, Any], model_cfg: Dict[str, Any]) -> Path:
    fourdgs_cfg = model_cfg.get("fourdgs", {}) if isinstance(model_cfg, dict) else {}
    configured = fourdgs_cfg.get("root", None) if isinstance(fourdgs_cfg, dict) else None
    candidates = []
    if configured:
        candidates.append(Path(str(configured)).expanduser())
    candidates.append(Path(__file__).resolve().parents[1] / "4DGaussians-master")
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "train.py").is_file():
            return candidate
    raise FileNotFoundError(
        "4DGaussians-master not found for eval. Set model.fourdgs.root in the "
        "4DGS checkpoint config, or place 4DGaussians-master next to clsplats."
    )


def _load_4dgs_runtime(root: Path) -> Dict[str, Any]:
    global _FOURDGS_RUNTIME
    root = root.resolve()
    if _FOURDGS_RUNTIME is not None and _FOURDGS_RUNTIME.get("root") == root:
        return _FOURDGS_RUNTIME

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    # 4DGS uses top-level packages named scene/utils/gaussian_renderer, which
    # can collide with Scaffold-GS in the same Python process. Reload those
    # top-level modules from the 4DGS root for this runtime.
    for module_name in list(sys.modules.keys()):
        if module_name in {"arguments", "gaussian_renderer", "scene", "utils"} or module_name.startswith(
            ("arguments.", "gaussian_renderer.", "scene.", "utils.")
        ):
            sys.modules.pop(module_name, None)

    arguments_module = importlib.import_module("arguments")
    scene_module = importlib.import_module("scene")
    camera_module = importlib.import_module("scene.cameras")
    renderer_module = importlib.import_module("gaussian_renderer")
    gaussian_model_module = importlib.import_module("scene.gaussian_model")

    _FOURDGS_RUNTIME = {
        "root": root,
        "Scene": getattr(scene_module, "Scene"),
        "Camera": getattr(camera_module, "Camera"),
        "render": getattr(renderer_module, "render"),
        "GaussianModel": getattr(gaussian_model_module, "GaussianModel"),
        "ModelHiddenParams": getattr(arguments_module, "ModelHiddenParams"),
    }
    return _FOURDGS_RUNTIME


def _find_latest_4dgs_iteration_dir(model_path: Path) -> Optional[Path]:
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.is_dir():
        return None
    candidates = [
        path for path in point_cloud_dir.iterdir()
        if path.is_dir()
        and re.fullmatch(r"iteration_\d+", path.name)
        and (path / "point_cloud.ply").is_file()
    ]
    if not candidates:
        return None

    def _iter_num(path: Path) -> int:
        digits = "".join(ch for ch in path.name if ch.isdigit())
        return int(digits) if digits else -1

    return max(candidates, key=_iter_num)


def _resolve_4dgs_time(source_dir: Path, timestep: int) -> float:
    raw_times = []
    for name in ("transforms_train.json", "transforms_test.json"):
        path = source_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for frame in payload.get("frames", []):
            if isinstance(frame, dict) and frame.get("time", None) is not None:
                try:
                    raw_times.append(float(frame["time"]))
                except Exception:
                    pass
    if not raw_times:
        return 0.0
    unique_times = sorted(set(raw_times))
    index = max(0, min(int(timestep), len(unique_times) - 1))
    max_time = max(unique_times)
    if abs(max_time) <= 1e-12:
        return 0.0
    return float(unique_times[index] / max_time)


def _load_4dgs_cfg_args(model_path: Path) -> argparse.Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        return argparse.Namespace()
    try:
        text = cfg_path.read_text(encoding="utf-8")
        payload = eval(text, {"Namespace": argparse.Namespace}, {})
        if isinstance(payload, argparse.Namespace):
            return payload
    except Exception:
        logger.warning(f"Failed to load 4DGS cfg_args from {cfg_path}; using eval defaults.")
    return argparse.Namespace()


def _namespace_cache_items(namespace: argparse.Namespace) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in vars(namespace).items()))


def _valid_4dgs_normalization(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    center = value.get("center", None)
    if center is None:
        return False
    try:
        center_array = np.asarray(center, dtype=np.float64)
        scale = float(value.get("scale", 1.0))
    except (TypeError, ValueError):
        return False
    return bool(
        center_array.shape == (3,)
        and np.isfinite(center_array).all()
        and math.isfinite(scale)
        and scale > 0.0
    )


def _resolve_4dgs_normalization(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    model_cfg: Dict[str, Any],
    source_dir: Path,
) -> Dict[str, Any]:
    normalization = checkpoint.get("normalization", {})
    if _valid_4dgs_normalization(normalization):
        return dict(normalization)

    fourdgs_cfg = model_cfg.get("fourdgs", {}) if isinstance(model_cfg, dict) else {}
    if isinstance(fourdgs_cfg, dict) and not bool(fourdgs_cfg.get("normalize_scene", True)):
        return dict(normalization) if isinstance(normalization, dict) else {}

    sidecar_path = source_dir / "normalization.json"
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            sidecar = None
        if _valid_4dgs_normalization(sidecar):
            logger.warning(
                "Recovered missing 4DGS normalization from dataset sidecar: "
                f"{sidecar_path}"
            )
            return dict(sidecar)

    def _checkpoint_timestep(path: Path) -> int:
        match = re.fullmatch(r"checkpoint_t(\d+)\.pt", path.name)
        return int(match.group(1)) if match else 10**9

    sibling_checkpoints = sorted(
        checkpoint_path.parent.glob("checkpoint_t*.pt"),
        key=_checkpoint_timestep,
    )
    for candidate in sibling_checkpoints:
        if candidate.resolve() == checkpoint_path.resolve():
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        candidate_normalization = payload.get("normalization", {}) if isinstance(payload, dict) else {}
        if _valid_4dgs_normalization(candidate_normalization):
            logger.warning(
                "Recovered missing 4DGS normalization from sibling checkpoint: "
                f"{candidate}"
            )
            return dict(candidate_normalization)

    logger.warning(
        "4DGS checkpoint has no valid normalization metadata. Custom camera renders "
        "may be misaligned if the training input was normalized."
    )
    return dict(normalization) if isinstance(normalization, dict) else {}


def _build_4dgs_eval_model(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: str,
    timestep: int,
):
    if device != "cuda":
        raise RuntimeError(
            "4DGS evaluation requires CUDA because the original 4DGS renderer "
            "and deformation modules construct CUDA tensors directly."
        )

    root = _resolve_4dgs_root(checkpoint, model_cfg)
    runtime = _load_4dgs_runtime(root)

    source_dir = checkpoint.get("source_dir", None)
    model_path = checkpoint.get("model_path", None)
    if source_dir is None:
        # Bridge checkpoints live inside the model directory, while native
        # 4DGS rendering receives that model directory itself.  Normalize both
        # forms to the generated-input layout used by the bridge.  Older runs
        # may place _4dgs_input beside the model directory, so accept both.
        model_root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        source_candidates = (
            model_root / "_4dgs_input" / model_root.name,
            model_root.parent / "_4dgs_input" / model_root.name,
            model_root / "_4dgs_input",
        )
        source_dir = next(
            (candidate for candidate in source_candidates if candidate.is_dir()),
            source_candidates[0],
        )
    if model_path is None:
        model_path = checkpoint_path.parent
    source_dir = Path(str(source_dir)).expanduser().resolve()
    model_path = Path(str(model_path)).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"4DGS eval source_dir does not exist: {source_dir}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"4DGS eval model_path does not exist: {model_path}")

    normalization = _resolve_4dgs_normalization(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        source_dir=source_dir,
    )
    if _valid_4dgs_normalization(normalization):
        logger.info(
            "4DGS custom-camera normalization: "
            f"center={normalization['center']}, scale={float(normalization['scale']):.8f}"
        )

    cfg_args = _load_4dgs_cfg_args(model_path)
    scene_args = SimpleNamespace(
        sh_degree=int(getattr(cfg_args, "sh_degree", model_cfg.get("sh_degree", 3) if isinstance(model_cfg, dict) else 3)),
        source_path=str(source_dir),
        model_path=str(model_path),
        images=str(getattr(cfg_args, "images", "images") or "images"),
        resolution=int(getattr(cfg_args, "resolution", -1) or -1),
        white_background=bool(getattr(cfg_args, "white_background", True)),
        data_device="cuda",
        eval=bool(getattr(cfg_args, "eval", True)),
        render_process=False,
        add_points=False,
        extension=str(getattr(cfg_args, "extension", checkpoint.get("image_extension", ".png")) or ".png"),
        llffhold=int(getattr(cfg_args, "llffhold", 8) or 8),
    )
    latest_iteration_dir = _find_latest_4dgs_iteration_dir(model_path)
    expected_iteration = int(getattr(cfg_args, "iterations", 0) or 0)
    if expected_iteration > 0:
        expected_iteration_dir = model_path / "point_cloud" / f"iteration_{expected_iteration}"
        if not (expected_iteration_dir / "point_cloud.ply").is_file():
            latest_name = (
                latest_iteration_dir.name
                if latest_iteration_dir is not None
                else "none"
            )
            raise FileNotFoundError(
                "4DGS training is incomplete for comparison rendering: expected "
                f"{expected_iteration_dir}/point_cloud.ply from cfg_args, "
                f"but the latest available fine model is {latest_name}."
            )
        iteration_dir = expected_iteration_dir
    else:
        iteration_dir = latest_iteration_dir
    if iteration_dir is None:
        raise FileNotFoundError(
            "4DGS fine-stage model is not available under "
            f"{model_path}/point_cloud (expected iteration_N/point_cloud.ply). "
            "The run may still be in coarse training or may not have completed."
        )
    ply_path = iteration_dir / "point_cloud.ply"

    cache_key = (
        str(root),
        str(source_dir),
        str(model_path),
        None if iteration_dir is None else str(iteration_dir),
        int(scene_args.sh_degree),
        str(scene_args.images),
        int(scene_args.resolution),
        bool(scene_args.white_background),
        str(scene_args.extension),
        int(scene_args.llffhold),
        _namespace_cache_items(cfg_args),
        str(device),
    )
    context = _FOURDGS_EVAL_CONTEXT_CACHE.get(cache_key)
    if context is None:
        hyper_parser = argparse.ArgumentParser()
        hyper_group = runtime["ModelHiddenParams"](hyper_parser)
        hyper_defaults = hyper_parser.parse_args([])
        for key, value in vars(cfg_args).items():
            if hasattr(hyper_defaults, key):
                setattr(hyper_defaults, key, value)
        hyper = hyper_group.extract(hyper_defaults)

        gaussians = runtime["GaussianModel"](scene_args.sh_degree, hyper)
        iteration_match = re.fullmatch(r"iteration_(\d+)", iteration_dir.name)
        load_iteration = int(iteration_match.group(1)) if iteration_match else -1
        scene = runtime["Scene"](
            scene_args,
            gaussians,
            load_iteration=load_iteration,
            shuffle=False,
        )
        if hasattr(gaussians, "eval"):
            gaussians.eval()

        context = {
            "root": root,
            "runtime": runtime,
            "gaussians": gaussians,
            "scene": scene,
            "source_dir": source_dir,
            "model_path": model_path,
            "dataset_type": getattr(scene, "dataset_type", "blender"),
            "normalization": normalization,
            "white_background": bool(scene_args.white_background),
            "cache_key": cache_key,
        }
        _FOURDGS_EVAL_CONTEXT_CACHE.clear()
        _FOURDGS_EVAL_CONTEXT_CACHE[cache_key] = context
    else:
        context["normalization"] = normalization
        logger.info(
            "Reusing cached 4DGS eval scene: "
            f"model_path={model_path}, iteration={iteration_dir.name if iteration_dir else 'unknown'}"
        )

    target_time = _resolve_4dgs_time(source_dir, int(timestep))
    context["time"] = target_time
    scene = context["scene"]
    context["test_camera_indices"] = _4dgs_indices_for_time(scene.getTestCameras(), target_time)
    context["train_camera_indices"] = _4dgs_indices_for_time(scene.getTrainCameras(), target_time)
    return context, ply_path


def _4dgs_indices_for_time(camera_dataset, target_time: float) -> list[int]:
    raw_dataset = getattr(camera_dataset, "dataset", camera_dataset)
    indexed_times = []
    for idx in range(len(raw_dataset)):
        item = raw_dataset[idx]
        if hasattr(item, "time"):
            indexed_times.append((idx, float(item.time)))
    if not indexed_times:
        return list(range(len(camera_dataset)))

    eps = 1e-6
    matches = [idx for idx, time_value in indexed_times if abs(time_value - target_time) <= eps]
    if matches:
        return matches

    nearest = min(indexed_times, key=lambda pair: abs(pair[1] - target_time))[1]
    return [idx for idx, time_value in indexed_times if abs(time_value - nearest) <= eps]


def _select_4dgs_scene_camera(context: Dict[str, Any], split: str, pair_index: int):
    if split == "train":
        camera_dataset = context["scene"].getTrainCameras()
        indices = context.get("train_camera_indices", [])
    else:
        camera_dataset = context["scene"].getTestCameras()
        indices = context.get("test_camera_indices", [])

    if pair_index < len(indices):
        return camera_dataset[indices[pair_index]]
    return None


def _camera_rt_from_clsplats_camera(camera) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(camera, "R") and hasattr(camera, "T"):
        return np.asarray(camera.R, dtype=np.float64), np.asarray(camera.T, dtype=np.float64)

    if not hasattr(camera, "world_view_transform"):
        raise AttributeError(
            "Cannot convert camera to 4DGS format: missing R/T and world_view_transform."
        )
    w2c = camera.world_view_transform.detach().cpu().numpy().T.astype(np.float64)
    return w2c[:3, :3].T, w2c[:3, 3]


def _apply_4dgs_camera_normalization(
    R: np.ndarray,
    T: np.ndarray,
    normalization: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    center = normalization.get("center", None) if isinstance(normalization, dict) else None
    scale = float(normalization.get("scale", 1.0)) if isinstance(normalization, dict) else 1.0
    if center is None:
        return R, T

    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,):
        return R, T

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(R, dtype=np.float64).T
    w2c[:3, 3] = np.asarray(T, dtype=np.float64)

    c2w = np.linalg.inv(w2c)
    c2w[:3, 3] = (c2w[:3, 3] - center_arr) * scale
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :3].T, w2c[:3, 3]


def _to_4dgs_camera(camera, gt_image: torch.Tensor, context: Dict[str, Any]):
    camera_cls = context["runtime"]["Camera"]
    image_cpu = gt_image.detach().clamp(0.0, 1.0).cpu()
    R, T = _camera_rt_from_clsplats_camera(camera)
    R, T = _apply_4dgs_camera_normalization(R, T, context.get("normalization", {}))
    return camera_cls(
        colmap_id=int(getattr(camera, "uid", 0)),
        R=R,
        T=T,
        FoVx=float(camera.FoVx),
        FoVy=float(camera.FoVy),
        image=image_cpu,
        gt_alpha_mask=None,
        image_name=str(getattr(camera, "image_name", "eval_view")),
        uid=int(getattr(camera, "uid", 0)),
        data_device="cuda",
        time=float(context["time"]),
        mask=None,
    )


def _render_4dgs_view(
    camera,
    gt_image: torch.Tensor,
    context: Dict[str, Any],
    bg_color: torch.Tensor,
    *,
    pair_index: int = 0,
    split: str = "test",
) -> torch.Tensor:
    runtime = context["runtime"]
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    # Prefer the camera object that 4DGS created from the exported transforms.
    # This keeps evaluation on the same imported camera convention used during
    # training.  Fall back to converting the CL-Splats camera if ordering differs.
    fourdgs_camera = _select_4dgs_scene_camera(context, split=split, pair_index=pair_index)
    if fourdgs_camera is None:
        fourdgs_camera = _to_4dgs_camera(camera, gt_image, context)
    render_pkg = runtime["render"](
        fourdgs_camera,
        context["gaussians"],
        pipe,
        bg_color,
        stage="fine",
        cam_type=context["dataset_type"],
    )
    return render_pkg["render"]


def _normalize_model_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _is_hash_model(model_name: str) -> bool:
    return model_name in {
        "3dgs_hash",
        "pure_3dgs_idea3_hash",
        "idea3_hash",
    }


def _is_official_cl_splats_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    return (
        str(checkpoint.get("trainer_type", "")).strip().lower()
        == "official_cl_splats"
    )


def _load_official_cl_splats_ply(
    ply_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a public CL-Splats PLY without changing its gsplat layout."""
    vertices = PlyData.read(str(ply_path))["vertex"]
    names = {str(prop.name) for prop in vertices.properties}
    required = {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"Official CL-Splats PLY is missing required fields: {missing}; path={ply_path}"
        )

    means = np.stack([vertices[name] for name in ("x", "y", "z")], axis=-1).astype(np.float32)
    log_scales = np.stack(
        [vertices[name] for name in ("scale_0", "scale_1", "scale_2")], axis=-1
    ).astype(np.float32)
    quats = np.stack(
        [vertices[name] for name in ("rot_0", "rot_1", "rot_2", "rot_3")], axis=-1
    ).astype(np.float32)
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)
    sh0 = np.stack(
        [vertices[name] for name in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=-1
    ).astype(np.float32)[:, None, :]

    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    if len(rest_names) % 3 != 0:
        raise ValueError(
            f"Official CL-Splats PLY has invalid SH field count={len(rest_names)}; path={ply_path}"
        )
    if rest_names:
        rest = np.stack([vertices[name] for name in rest_names], axis=-1).astype(np.float32)
        shn = rest.reshape(means.shape[0], -1, 3)
    else:
        shn = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    colors = np.concatenate((sh0, shn), axis=1)
    num_sh_coeffs = int(colors.shape[1])
    sh_degree = math.isqrt(num_sh_coeffs) - 1
    if (sh_degree + 1) ** 2 != num_sh_coeffs:
        raise ValueError(
            f"Official CL-Splats PLY has non-square SH coefficient count={num_sh_coeffs}; path={ply_path}"
        )

    return {
        "means": torch.from_numpy(means).to(device),
        "scales": torch.exp(torch.from_numpy(log_scales).to(device)),
        "quats": F.normalize(torch.from_numpy(quats).to(device), dim=-1),
        "opacities": torch.sigmoid(torch.from_numpy(opacity_logits).to(device)),
        "colors": torch.from_numpy(colors).to(device),
        "sh_degree": int(sh_degree),
    }


def _render_official_cl_splats_view(
    camera,
    params: Dict[str, Any],
    bg_color: torch.Tensor,
) -> torch.Tensor:
    """Render a public CL-Splats PLY with the same gsplat path used in training."""
    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise ImportError(
            "Official CL-Splats evaluation requires gsplat in the active environment."
        ) from exc

    means = params["means"]
    device = means.device
    dtype = means.dtype
    width = int(camera.image_width)
    height = int(camera.image_height)

    viewmats = camera.world_view_transform.transpose(0, 1).to(
        device=device, dtype=dtype
    ).unsqueeze(0)
    intrinsics = torch.zeros((1, 3, 3), device=device, dtype=dtype)
    intrinsics[..., 0, 0] = 0.5 * width / math.tan(float(camera.FoVx) * 0.5)
    intrinsics[..., 1, 1] = 0.5 * height / math.tan(float(camera.FoVy) * 0.5)
    intrinsics[..., 0, 2] = (width - 1) * 0.5
    intrinsics[..., 1, 2] = (height - 1) * 0.5
    intrinsics[..., 2, 2] = 1.0

    rendered, _alphas, _info = rasterization(
        means=means,
        quats=params["quats"],
        scales=params["scales"],
        opacities=params["opacities"],
        colors=params["colors"],
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=int(params["sh_degree"]),
        backgrounds=bg_color.to(device=device, dtype=dtype).view(1, 3),
        packed=False,
        distributed=False,
        render_mode="RGB",
    )
    return rendered[0, ..., :3].permute(2, 0, 1).contiguous()


def _ply_field_names(ply_path: Path) -> set[str]:
    try:
        ply = PlyData.read(str(ply_path))
        if len(ply.elements) == 0:
            return set()
        names = ply.elements[0].data.dtype.names
        if names is None:
            return set()
        return {str(name) for name in names}
    except Exception:
        return set()


def _is_sfgs_model(
    model_name: str,
    checkpoint: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> bool:
    if model_name in {
        "irc-gs",
        "scaffold-gs",
    }:
        return True
    if str(checkpoint.get("trainer_type", "")).strip().lower() in {"scaffold_gs_rebuild", "scaffold_gs"}:
        return True
    if checkpoint_path is not None:
        try:
            ply_path = resolve_ply_path(checkpoint_path, checkpoint)
        except FileNotFoundError:
            return False
        field_names = _ply_field_names(ply_path)
        if {"f_anchor_feat_0", "f_offset_0"} & field_names:
            return True
    return False


def _checkpoint_can_restore_timestep_view(checkpoint: Dict[str, Any], timestep: int) -> bool:
    try:
        if int(checkpoint.get("timestep", -1)) == int(timestep):
            return True
    except Exception:
        pass
    for state_key in ("idea3_state", "idea1_state"):
        state = checkpoint.get(state_key, None)
        if not isinstance(state, dict):
            continue
        eval_bank = state.get("scene_eval_bank", None)
        if _resolve_keyed_value(eval_bank, int(timestep)) is not None:
            return True
    return False


def _maybe_swap_to_timestep_checkpoint(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    timestep: int,
) -> Tuple[Path, Dict[str, Any], bool]:
    if _checkpoint_can_restore_timestep_view(checkpoint, int(timestep)):
        return checkpoint_path, checkpoint, False
    if "final" not in checkpoint_path.name.lower():
        return checkpoint_path, checkpoint, False
    fallback_ckpt = checkpoint_path.parent / f"checkpoint_t{int(timestep)}.pt"
    if not fallback_ckpt.exists():
        return checkpoint_path, checkpoint, False
    swapped = torch.load(fallback_ckpt, map_location="cpu", weights_only=False)
    return fallback_ckpt.resolve(), swapped, True


def build_representation_from_checkpoint(checkpoint: Dict[str, Any]):
    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    available = set(get_available_representations())
    representation_name = model_cfg.get("representation", None)
    model_name = model_cfg.get("name", None)

    if representation_name in available:
        chosen_name = representation_name
    elif model_name in available:
        chosen_name = model_name
    else:
        chosen_name = "gaussian"

    representation_cfg = {"name": chosen_name}
    if model_cfg.get("sh_degree", None) is not None:
        representation_cfg["sh_degree"] = model_cfg["sh_degree"]
    if model_cfg.get("optimizer_type", None) is not None:
        representation_cfg["optimizer_type"] = model_cfg["optimizer_type"]
    return create_representation_model(omegaconf.OmegaConf.create(representation_cfg))


def safe_psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((rendered - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def _default_eval_images_dir(checkpoint_path: Path, timestep: int) -> Path:
    return checkpoint_path.parent / "visualizations" / "tests" / f"t{int(timestep)}"


def _tensor_image_to_uint8_array(image: torch.Tensor):
    return (
        image.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if args.prefer_dist and args.prefer_undist:
        raise ValueError("--prefer-dist and --prefer-undist cannot be used together.")
    prefer_undist = not args.prefer_dist
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_name = _normalize_model_name(model_cfg.get("name", None))

    dataset = CLSplatsDataset(
        path=str(dataset_path),
        resolution_scale=args.resolution_scale,
        white_background=args.white_background,
        eval_mode=True,
        device=device,
        split_seed=args.split_seed,
        prefer_undist=prefer_undist,
    )

    timestep = int(checkpoint.get("timestep", 0)) if args.timestep is None else int(args.timestep)
    checkpoint_path, checkpoint, used_timestep_fallback_ckpt = _maybe_swap_to_timestep_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        timestep=int(timestep),
    )
    ckpt_cfg = checkpoint.get("config", {})
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_name = _normalize_model_name(model_cfg.get("name", None))
    is_4dgs_model = _is_4dgs_model(model_name, checkpoint)
    if is_4dgs_model and not model_name:
        model_name = "4dgs"
    is_sfgs_model = False if is_4dgs_model else _is_sfgs_model(model_name, checkpoint, checkpoint_path)
    is_official_cl_splats = _is_official_cl_splats_checkpoint(checkpoint)

    if timestep < 0 or timestep >= dataset.get_num_timesteps():
        raise ValueError(
            f"Invalid timestep {timestep}. Dataset has {dataset.get_num_timesteps()} timesteps."
        )

    save_images = bool(getattr(args, "save_images", False))
    save_gt_images = not bool(getattr(args, "no_save_gt_images", False))
    images_dir: Optional[Path] = None
    gt_images_dir: Optional[Path] = None
    saved_images = 0
    saved_gt_images = 0
    if save_images or save_gt_images:
        images_dir = _default_eval_images_dir(checkpoint_path, timestep)
    if save_images and images_dir is not None:
        _clear_directory(images_dir)
    if save_gt_images and images_dir is not None:
        gt_images_dir = images_dir / "gt"
        _clear_directory(gt_images_dir)

    model = None
    ply_path: Optional[Path] = None
    mlp_dir: Optional[Path] = None
    render_model = None
    render_source = "full_ply"
    render_split_meta: Optional[Dict[str, int]] = None
    render_with_background_only = False
    hash_stage2_color_enabled = False
    temporal_visible_gaussians = None
    temporal_total_gaussians = None

    if is_official_cl_splats:
        ply_path = resolve_ply_path(checkpoint_path, checkpoint)
        render_model = _load_official_cl_splats_ply(
            ply_path=ply_path,
            device=torch.device(device),
        )
        render_source = "official_gsplat_ply"
    elif is_sfgs_model:
        render_model, ply_path, mlp_dir = _build_sfgs_eval_model(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model_cfg=model_cfg,
            device=device,
            timestep=int(timestep),
        )
        render_source = "scaffold_gs"
        temporal_visible_gaussians = getattr(render_model, "_eval_temporal_visible_anchors", None)
        temporal_total_gaussians = getattr(render_model, "_eval_temporal_total_anchors", None)
    elif is_4dgs_model:
        render_model, ply_path = _build_4dgs_eval_model(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model_cfg=model_cfg,
            device=device,
            timestep=int(timestep),
        )
        render_source = "4dgs"
    else:
        ply_path = resolve_ply_path(checkpoint_path, checkpoint)
        model = build_representation_from_checkpoint(checkpoint)
        model.load_ply(str(ply_path))

        render_model = model
        if _is_hash_model(model_name):
            hash_view, hash_visible, hash_split_meta = _build_3dgs_hash_render_view(
                checkpoint=checkpoint,
                device=device,
                timestep=int(timestep),
                fallback_model=model,
            )
            if hash_view is not None or hash_visible == 0:
                render_source = "scene_eval_bank"
                render_split_meta = hash_split_meta
                if int(hash_visible or 0) > 0:
                    render_model = hash_view
                else:
                    render_with_background_only = True
            hash_stage2_color_enabled = True

    test_cameras = dataset.get_test_cameras(timestep)
    test_images = dataset.get_test_images(timestep)
    num_pairs = min(len(test_cameras), len(test_images))
    if num_pairs == 0:
        raise RuntimeError(
            f"No test camera-image pairs available at timestep {timestep}. "
            "Please generate split first and ensure test images can be loaded."
        )

    bg_color = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )
    if is_4dgs_model and isinstance(render_model, dict):
        bg_color = torch.tensor(
            [1, 1, 1] if bool(render_model.get("white_background", args.white_background)) else [0, 0, 0],
            dtype=torch.float32,
            device=device,
        )

    def _evaluate_pairs(
        cameras,
        images,
        *,
        save_outputs: bool = False,
        split_name: str = "test",
    ) -> Dict[str, Any]:
        l1_values = []
        ssim_values = []
        psnr_values = []
        skipped_shape_mismatch = 0
        first_shape_mismatch = None
        saved_count = 0
        saved_gt_count = 0
        pair_count = min(len(cameras), len(images))

        with torch.no_grad():
            for idx in range(pair_count):
                camera = cameras[idx]
                gt_image = images[idx]

                if render_with_background_only:
                    rendered = bg_color.view(3, 1, 1).expand_as(gt_image).clone()
                elif is_official_cl_splats:
                    rendered = _render_official_cl_splats_view(
                        camera, render_model, bg_color
                    )
                elif is_sfgs_model:
                    rendered = _render_sfgs_view(camera, render_model, bg_color)
                elif is_4dgs_model:
                    rendered = _render_4dgs_view(
                        camera,
                        gt_image,
                        render_model,
                        bg_color,
                        pair_index=idx,
                        split=split_name,
                    )
                else:
                    override_color = None
                    if hash_stage2_color_enabled:
                        override_color = _build_stage2_color_eval_override(
                            checkpoint=checkpoint,
                            device=torch.device(device),
                            camera=camera,
                            render_view=render_model,
                        )
                    render_result = render(
                        camera,
                        render_model,
                        bg_color,
                        override_color=override_color,
                    )
                    rendered = render_result["render"]

                if save_outputs and save_images and images_dir is not None:
                    frame = _tensor_image_to_uint8_array(rendered)
                    out_path = images_dir / f"{idx:06d}.png"
                    PILImage.fromarray(frame).save(out_path)
                    saved_count += 1

                if save_outputs and save_gt_images and gt_images_dir is not None:
                    gt_frame = _tensor_image_to_uint8_array(gt_image)
                    gt_out_path = gt_images_dir / f"{idx:06d}.png"
                    PILImage.fromarray(gt_frame).save(gt_out_path)
                    saved_gt_count += 1

                if rendered.shape != gt_image.shape:
                    if args.allow_resize_mismatch:
                        gt_image = F.interpolate(
                            gt_image.unsqueeze(0),
                            size=(rendered.shape[1], rendered.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    else:
                        skipped_shape_mismatch += 1
                        if first_shape_mismatch is None:
                            first_shape_mismatch = {
                                "idx": int(idx),
                                "image_name": str(getattr(camera, "image_name", f"view_{idx}")),
                                "rendered_shape": tuple(int(x) for x in rendered.shape),
                                "gt_shape": tuple(int(x) for x in gt_image.shape),
                                "camera_hw": (
                                    int(getattr(camera, "image_height", -1)),
                                    int(getattr(camera, "image_width", -1)),
                                ),
                            }
                        continue

                l1_values.append(float(l1_loss(rendered, gt_image).item()))
                ssim_values.append(float(ssim(rendered.unsqueeze(0), gt_image.unsqueeze(0)).item()))
                psnr_values.append(float(safe_psnr(rendered, gt_image)))

        metrics = None
        if l1_values:
            metrics = {
                "l1": float(sum(l1_values) / len(l1_values)),
                "ssim": float(sum(ssim_values) / len(ssim_values)),
                "psnr": float(sum(psnr_values) / len(psnr_values)),
            }
        return {
            "metrics": metrics,
            "num_pairs": int(pair_count),
            "num_evaluated_views": int(len(l1_values)),
            "num_skipped_shape_mismatch": int(skipped_shape_mismatch),
            "first_shape_mismatch": first_shape_mismatch,
            "saved_images": int(saved_count),
            "saved_gt_images": int(saved_gt_count),
        }

    test_eval = _evaluate_pairs(test_cameras, test_images, save_outputs=True, split_name="test")
    if not test_eval["metrics"]:
        raise RuntimeError(
            "Evaluation produced no valid views. "
            f"Skipped shape mismatches: {test_eval['num_skipped_shape_mismatch']}, "
            f"first mismatch: {test_eval['first_shape_mismatch']}"
        )
    train_eval = None
    if bool(getattr(args, "eval_train_split", False)):
        train_eval = _evaluate_pairs(
            dataset.get_cameras(timestep),
            dataset.get_images(timestep),
            split_name="train",
        )

    benchmark = None
    if bool(getattr(args, "benchmark_fps", False)):
        warmup = max(0, int(getattr(args, "benchmark_warmup", 5)))
        repeat = max(1, int(getattr(args, "benchmark_repeat", 10)))
        bench_cameras = list(test_cameras[:num_pairs])

        def _render_for_benchmark(camera, camera_index: int):
            if render_with_background_only:
                height = int(getattr(camera, "image_height", 1))
                width = int(getattr(camera, "image_width", 1))
                return bg_color.view(3, 1, 1).expand(3, height, width).clone()
            if is_official_cl_splats:
                return _render_official_cl_splats_view(camera, render_model, bg_color)
            if is_sfgs_model:
                return _render_sfgs_view(camera, render_model, bg_color)
            if is_4dgs_model:
                gt_image = test_images[int(camera_index)]
                return _render_4dgs_view(
                    camera,
                    gt_image,
                    render_model,
                    bg_color,
                    pair_index=int(camera_index),
                    split="test",
                )
            override_color = None
            if hash_stage2_color_enabled:
                override_color = _build_stage2_color_eval_override(
                    checkpoint=checkpoint,
                    device=torch.device(device),
                    camera=camera,
                    render_view=render_model,
                )
            return render(
                camera,
                render_model,
                bg_color,
                override_color=override_color,
            )["render"]

        with torch.no_grad():
            for idx in range(warmup):
                bench_idx = idx % len(bench_cameras)
                _ = _render_for_benchmark(bench_cameras[bench_idx], bench_idx)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            total_frames = 0
            for _pass_idx in range(repeat):
                for bench_idx, camera in enumerate(bench_cameras):
                    _ = _render_for_benchmark(camera, bench_idx)
                    total_frames += 1
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_sec = max(time.perf_counter() - start_time, 1e-12)

        benchmark = {
            "fps": float(total_frames / elapsed_sec),
            "mean_render_ms": float((elapsed_sec / total_frames) * 1000.0),
            "total_frames": int(total_frames),
            "elapsed_sec": float(elapsed_sec),
            "warmup_frames": int(warmup),
            "repeat": int(repeat),
        }

    result = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "ply_path": None if ply_path is None else str(ply_path),
        "mlp_dir": None if mlp_dir is None else str(mlp_dir),
        "timestep": int(timestep),
        "model_name": model_name,
        "split_file_loaded": True,
        "used_timestep_fallback_checkpoint": bool(used_timestep_fallback_ckpt),
        "render_source": render_source,
        "temporal_source": render_source,
        "render_background_only": bool(render_with_background_only),
        "temporal_visible_gaussians": temporal_visible_gaussians,
        "temporal_total_gaussians": temporal_total_gaussians,
        "hash_stage2_color_enabled": bool(hash_stage2_color_enabled),
        "render_split_meta": render_split_meta,
        "num_test_cameras": int(len(test_cameras)),
        "num_test_images_loaded": int(len(test_images)),
        "num_evaluated_views": int(test_eval["num_evaluated_views"]),
        "num_skipped_shape_mismatch": int(test_eval["num_skipped_shape_mismatch"]),
        "first_shape_mismatch": test_eval["first_shape_mismatch"],
        "saved_test_images": int(test_eval["saved_images"]),
        "saved_test_images_dir": None if not save_images or images_dir is None else str(images_dir),
        "saved_gt_images": int(test_eval["saved_gt_images"]),
        "saved_gt_images_dir": None if gt_images_dir is None else str(gt_images_dir),
        "metrics": test_eval["metrics"],
        "train_split_metrics": None if train_eval is None else train_eval["metrics"],
        "num_train_cameras": None if train_eval is None else int(train_eval["num_pairs"]),
        "num_train_evaluated_views": None if train_eval is None else int(train_eval["num_evaluated_views"]),
        "benchmark": benchmark,
    }
    return result


def _parse_timesteps_arg(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    timesteps: list[int] = []
    for raw_part in str(value).replace(" ", "").split(","):
        if not raw_part:
            continue
        if "-" in raw_part:
            start_text, end_text = raw_part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            timesteps.extend(range(start, end + step, step))
        else:
            timesteps.append(int(raw_part))
    if not timesteps:
        raise ValueError("--timesteps did not contain any timestep indices.")
    return timesteps


def _write_eval_json(result: Any, output_json: str) -> None:
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics json: {output_path}")


def _print_eval_result(result: Dict[str, Any]) -> None:
    print("=" * 72)
    print("CL-Splats Evaluation")
    print("=" * 72)
    print(f"Dataset path           : {result['dataset_path']}")
    print(f"Checkpoint             : {result['checkpoint_path']}")
    print(f"PLY path               : {result['ply_path']}")
    if result.get("mlp_dir"):
        print(f"MLP dir                : {result['mlp_dir']}")
    print(f"Model                  : {result['model_name']}")
    print(f"Timestep               : {result['timestep']}")
    print(f"Render source          : {result['render_source']}")
    print(f"Background-only render : {result['render_background_only']}")
    if result.get("temporal_visible_gaussians") is not None:
        print(
            "Temporal visible      : "
            f"{result['temporal_visible_gaussians']}/{result.get('temporal_total_gaussians')}"
        )
    print(f"Hash stage2 color      : {result['hash_stage2_color_enabled']}")
    print(f"Test cameras           : {result['num_test_cameras']}")
    print(f"Test images loaded     : {result['num_test_images_loaded']}")
    print(f"Evaluated views        : {result['num_evaluated_views']}")
    if result.get("train_split_metrics") is not None:
        train_metrics = result["train_split_metrics"]
        print(f"Train evaluated views  : {result['num_train_evaluated_views']}")
        print(
            "Train split metrics    : "
            f"L1={train_metrics['l1']:.6f}, "
            f"SSIM={train_metrics['ssim']:.6f}, "
            f"PSNR={train_metrics['psnr']:.4f}"
        )
    print(f"Skipped shape mismatch : {result['num_skipped_shape_mismatch']}")
    if result["saved_test_images"] > 0:
        print(f"Saved test renders     : {result['saved_test_images']}")
        print(f"Saved renders dir      : {result['saved_test_images_dir']}")
    if result["saved_gt_images"] > 0:
        print(f"Saved GT images        : {result['saved_gt_images']}")
        print(f"Saved GT dir           : {result['saved_gt_images_dir']}")
    print("-" * 72)
    print(f"L1   : {result['metrics']['l1']:.6f}")
    print(f"SSIM : {result['metrics']['ssim']:.6f}")
    print(f"PSNR : {result['metrics']['psnr']:.4f}")
    if result.get("benchmark") is not None:
        benchmark = result["benchmark"]
        print("-" * 72)
        print(f"FPS              : {benchmark['fps']:.2f}")
        print(f"Mean render time : {benchmark['mean_render_ms']:.3f} ms")
        print(f"Benchmark frames : {benchmark['total_frames']}")
    print("=" * 72)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        timesteps = _parse_timesteps_arg(args.timesteps)
        if timesteps is None:
            result = evaluate(args)
            _print_eval_result(result)
            if args.output_json:
                _write_eval_json(result, args.output_json)
            return

        results = []
        for timestep in timesteps:
            run_args = argparse.Namespace(**vars(args))
            run_args.timestep = int(timestep)
            run_args.timesteps = None
            print("=" * 72)
            print(f"[RUN] timestep={int(timestep)}")
            print("=" * 72)
            result = evaluate(run_args)
            results.append(result)
            _print_eval_result(result)
    except Exception as exc:
        print(f"Error: {exc}")
        if getattr(args, "traceback", False):
            traceback.print_exc()
        raise SystemExit(1)

    if args.output_json:
        _write_eval_json({"results": results}, args.output_json)


if __name__ == "__main__":
    main()
