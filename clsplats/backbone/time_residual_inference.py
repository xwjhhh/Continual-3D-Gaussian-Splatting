from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_residual_cnn import TimeResidualRefiner


def _cfg_get(container: Any, key: str, default):
    if container is None:
        return default
    try:
        value = container.get(key, default)
    except Exception:
        return default
    return default if value is None else value


def _resolve_keyed_value(container: Any, timestep: int):
    if container is None:
        return None
    key_candidates = (str(int(timestep)), int(timestep))
    for key in key_candidates:
        try:
            if key in container:
                return container[key]
        except Exception:
            continue
    return None


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
    return bool(default)


def _infer_hidden_dim(payload: Any) -> Optional[int]:
    if payload is None:
        return None
    weight = _cfg_get(payload, "residual_net.net.0.weight", None)
    if weight is None:
        weight = _cfg_get(payload, "residual_net.0.weight", None)
    if weight is None:
        return None
    if not isinstance(weight, torch.Tensor):
        weight = torch.as_tensor(weight)
    if weight.ndim < 1 or int(weight.shape[0]) <= 0:
        return None
    return int(weight.shape[0])


def _infer_in_channels(payload: Any) -> Optional[int]:
    if payload is None:
        return None
    weight = _cfg_get(payload, "residual_net.net.0.weight", None)
    if weight is None:
        weight = _cfg_get(payload, "residual_net.0.weight", None)
    if weight is None:
        return None
    if not isinstance(weight, torch.Tensor):
        weight = torch.as_tensor(weight)
    if weight.ndim < 2 or int(weight.shape[1]) <= 0:
        return None
    return int(weight.shape[1])


def _load_module_state_dict_shape_safe(
    module: torch.nn.Module,
    payload: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    if module is None or not isinstance(payload, dict):
        return 0, 0
    current = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in payload.items():
        if key not in current:
            skipped += 1
            continue
        dst = current[key]
        if tuple(dst.shape) != tuple(value.shape):
            skipped += 1
            continue
        filtered[key] = value
    if filtered:
        module.load_state_dict(filtered, strict=False)
    return int(len(filtered)), int(skipped)


def _align_input_channels(x: torch.Tensor, expected_in: int) -> torch.Tensor:
    """Align [B,C,H,W] to expected channel count by trim/pad."""
    if int(x.shape[1]) == int(expected_in):
        return x
    if int(x.shape[1]) > int(expected_in):
        return x[:, : int(expected_in)]
    pad = torch.zeros(
        (
            int(x.shape[0]),
            int(expected_in) - int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
        ),
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=1)


def load_time_residual_refiner(
    checkpoint: Dict[str, Any],
    timestep: int,
    device: str,
) -> Tuple[Optional[TimeResidualRefiner], Dict[str, Any]]:
    """
    Load a frozen per-time residual CNN from checkpoint when available.

    Enabled only when:
      - idea3_state/idea1_state has `time_residual_nets[timestep]`
      - and (if update_steps exists for this timestep) update_steps > 0
    """
    meta: Dict[str, Any] = {
        "enabled": False,
        "reason": "not_found",
        "timestep": int(timestep),
        "update_steps": 0,
        "hidden_dim": None,
        "clamp_output": False,
    }

    idea_state = _cfg_get(checkpoint, "idea3_state", None)
    if idea_state is None:
        idea_state = _cfg_get(checkpoint, "idea1_state", None)
    if idea_state is None:
        meta["reason"] = "missing_idea_state"
        return None, meta

    residual_bank = _cfg_get(idea_state, "time_residual_nets", None)
    payload = _resolve_keyed_value(residual_bank, timestep)
    if payload is None:
        meta["reason"] = "missing_time_residual_branch"
        return None, meta

    update_steps_bank = _cfg_get(idea_state, "time_residual_update_steps", None)
    raw_steps = _resolve_keyed_value(update_steps_bank, timestep)
    if raw_steps is not None:
        try:
            steps = int(raw_steps)
        except Exception:
            steps = 0
        meta["update_steps"] = max(0, steps)
        if steps <= 0:
            meta["reason"] = "time_residual_branch_not_trained"
            return None, meta

    cfg = _cfg_get(checkpoint, "config", {}) or {}
    train_cfg = _cfg_get(cfg, "train", {}) or {}
    hidden_dim = _infer_hidden_dim(payload)
    if hidden_dim is None:
        hidden_dim = _cfg_get(
            train_cfg,
            "idea3_color_hidden_dim",
            _cfg_get(train_cfg, "residual_cnn_hidden_dim", None),
        )
    if hidden_dim is None:
        hidden_dim = 32
    hidden_dim = max(1, int(hidden_dim))
    in_channels = _infer_in_channels(payload)
    if in_channels is None:
        in_channels = 13
    in_channels = max(1, int(in_channels))

    clamp_output = _to_bool(
        _cfg_get(
            train_cfg,
            "idea3_color_clamp_output",
            _cfg_get(train_cfg, "residual_cnn_clamp_output", False),
        ),
        False,
    )

    net = TimeResidualRefiner(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
    state_payload: Dict[str, torch.Tensor] = {}
    try:
        items = payload.items()
    except Exception:
        meta["reason"] = "invalid_time_residual_state"
        return None, meta
    for key, value in items:
        if isinstance(value, torch.Tensor):
            state_payload[key] = value.detach().to(device=device)
        else:
            state_payload[key] = torch.as_tensor(value, device=device)
    loaded_n, skipped_n = _load_module_state_dict_shape_safe(net, state_payload)
    if loaded_n <= 0:
        meta["reason"] = "state_shape_mismatch_no_compatible_keys"
        return None, meta

    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)

    meta["enabled"] = True
    if skipped_n > 0:
        meta["reason"] = f"loaded_partial(loaded={loaded_n},skipped={skipped_n})"
    else:
        meta["reason"] = "loaded"
    meta["hidden_dim"] = int(hidden_dim)
    meta["clamp_output"] = bool(clamp_output)
    return net, meta


def build_residual_condition_maps(
    camera,
    render_result: Dict[str, torch.Tensor],
    scene_extent: float,
) -> Dict[str, torch.Tensor]:
    base_rgb = render_result["render"]
    _, h, w = base_rgb.shape
    dtype = base_rgb.dtype
    device = base_rgb.device

    invdepth = render_result.get("invdepths", None)
    if invdepth is None:
        invdepth = torch.zeros((1, h, w), device=device, dtype=dtype)
    else:
        invdepth = invdepth.to(device=device, dtype=dtype)
        if invdepth.dim() == 2:
            invdepth = invdepth.unsqueeze(0)
        invdepth = invdepth[:1]

    alpha = (invdepth > 1e-8).to(dtype=dtype)

    yy = torch.arange(h, device=device, dtype=dtype)
    xx = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    fx = 0.5 * float(w) / max(math.tan(float(camera.FoVx) * 0.5), 1e-6)
    fy = 0.5 * float(h) / max(math.tan(float(camera.FoVy) * 0.5), 1e-6)
    cx = 0.5 * float(w - 1)
    cy = 0.5 * float(h - 1)

    x_cam = (grid_x - cx) / max(float(fx), 1e-6)
    y_cam = (grid_y - cy) / max(float(fy), 1e-6)
    z_cam = torch.ones_like(x_cam)
    ray_cam = torch.stack([x_cam, y_cam, z_cam], dim=0)
    ray_cam = ray_cam / (ray_cam.norm(dim=0, keepdim=True) + 1e-8)

    view_inv = torch.inverse(camera.world_view_transform.to(device=device, dtype=dtype))
    rot_c2w = view_inv[:3, :3]
    ray_world = ray_cam.reshape(3, -1).T @ rot_c2w
    ray_world = ray_world.T.reshape(3, h, w)
    ray_world = ray_world / (ray_world.norm(dim=0, keepdim=True) + 1e-8)

    depth = torch.where(invdepth > 1e-8, 1.0 / (invdepth + 1e-8), torch.zeros_like(invdepth))
    cam_center = camera.camera_center.to(device=device, dtype=dtype).view(3, 1, 1)
    xyz_world = cam_center + ray_world * depth
    scene_scale = max(float(scene_extent), 1e-6)
    xyz_world = xyz_world / scene_scale

    dx = F.pad(xyz_world[:, :, 2:] - xyz_world[:, :, :-2], (1, 1, 0, 0)) * 0.5
    dy = F.pad(xyz_world[:, 2:, :] - xyz_world[:, :-2, :], (0, 0, 1, 1)) * 0.5
    normal_world = torch.cross(dx, dy, dim=0)
    normal_world = normal_world / (normal_world.norm(dim=0, keepdim=True) + 1e-8)
    normal_world = normal_world * alpha

    return {
        "base_rgb": base_rgb,
        "xyz_world": xyz_world,
        "normal_world": normal_world,
        "ray_world": ray_world,
        "alpha": alpha,
    }


def predict_time_residual_correction(
    camera,
    render_result: Dict[str, torch.Tensor],
    residual_refiner: TimeResidualRefiner,
    scene_extent: float,
    clamp_output: bool = False,
) -> Dict[str, torch.Tensor]:
    cond = build_residual_condition_maps(camera, render_result, scene_extent)
    x = TimeResidualRefiner.build_input(
        cond["base_rgb"].unsqueeze(0),
        cond["xyz_world"].unsqueeze(0),
        cond["normal_world"].unsqueeze(0),
        cond["ray_world"].unsqueeze(0),
        cond["alpha"].unsqueeze(0),
    )
    first_conv = None
    try:
        first_conv = residual_refiner.residual_net.net[0]
    except Exception:
        first_conv = None
    if isinstance(first_conv, nn.Conv2d):
        x = _align_input_channels(x, int(first_conv.in_channels))
    residual = residual_refiner.residual_net(x).squeeze(0)
    corrected = cond["base_rgb"] + residual
    if bool(clamp_output):
        corrected = corrected.clamp(0.0, 1.0)
    return {
        "corrected": corrected,
        "residual": residual,
        "base_rgb": cond["base_rgb"],
        "alpha": cond["alpha"],
    }
