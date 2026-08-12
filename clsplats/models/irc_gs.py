"""
Scaffold-GS trainer bridge for the CL-Splats data pipeline.

This module bridges CL-Splats and Scaffold-GS training.

By default it keeps a single Scaffold-GS model across timesteps so training
can continue from the previous checkpoint or timestep state. The legacy
per-timestep full-rebuild behavior is still available through
`model.rebuild_every_timestep=true`.

Optimization strategy:
  - anchor/offset Gaussian representation
  - Scaffold-GS renderer + voxel prefilter
  - Scaffold-GS learning-rate schedule
  - Scaffold-GS anchor growing / pruning strategy

The built-in CL-Splats evaluator is still based on vanilla 3DGS tensors, so
this trainer opts out of that path and focuses on training/checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import math
import os
import random
import re
import shutil
import sys
import types

from loguru import logger
import omegaconf
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image as PILImage

try:
    import faiss  # type: ignore
    try:
        import faiss.contrib.torch_utils  # type: ignore  # noqa: F401
    except Exception:
        pass
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

from clsplats.models.cl_splats import CLSplatsTrainer


# Easy-to-edit ablation defaults
# Scaffold-GS anchor 生长/剪枝开始迭代数。
DEFAULT_UPDATE_FROM = 1_500
# Scaffold-GS anchor 生长/剪枝检查间隔。
DEFAULT_UPDATE_INTERVAL = 100
# Scaffold-GS anchor 生长/剪枝停止迭代数。
DEFAULT_UPDATE_UNTIL = 15_000
DEFAULT_MIN_OPACITY = 0.005
# Scaffold-GS 统计中判定 anchor 更新成功的阈值。
DEFAULT_SUCCESS_THRESHOLD = 0.8
# 触发 densify / anchor 生长的梯度阈值。
DEFAULT_DENSIFY_GRAD_THRESHOLD = 0.0002

DEFAULT_T0_IMPORTANCE_ADAPT_ITERS = 5_000
DEFAULT_T0_IMPORTANCE_PRUNE_RATIO = 0.4
DEFAULT_TEMPORAL_IMPORTANCE_PRUNE_RATIO = 0.2
TEMPORAL_TIMESTEP_DTYPE = torch.int8
TEMPORAL_ALIVE_SENTINEL = 127

DEFAULT_TEMPORAL_MASK_OPACITY_THRESHOLD = 0.005
# 每个时间块的 latent 维度（当前实现里也就是 block_dim）。
DEFAULT_TEMPORAL_LATENT_DIM = 8
# 第一阶段 temporal 训练结束迭代数。
DEFAULT_TEMPORAL_STAGE1_UNTIL = 10000
# 第二阶段 temporal 训练结束迭代数。
DEFAULT_TEMPORAL_STAGE2_UNTIL = 25_000
# 刷新 temporal clone 候选集合的迭代间隔。
DEFAULT_TEMPORAL_CLONE_REFRESH_INTERVAL = 500
# 每次统计 clone 候选时采样的训练视角数量。
DEFAULT_TEMPORAL_CLONE_NUM_VIEWS = 30
# 为候选 clone 查找邻近 anchor 时使用的 KNN 数量。
DEFAULT_TEMPORAL_CLONE_KNN = 12
# 从低 PSNR 像素中选取 clone 候选的比例。
DEFAULT_TEMPORAL_CLONE_TOP_LOW_PSNR_PERCENT = 0.3
# 像素级 PSNR 低于该值才参与 temporal clone 候选筛选，数值越低表示误差越大。
DEFAULT_TEMPORAL_CLONE_MAX_PSNR = 22
# 每次刷新最多保留的可见 anchor 候选数量。
DEFAULT_TEMPORAL_CLONE_MAX_VISIBLE_CANDIDATES = 100_000
# 每个视角最多采样的高误差像素数量。
DEFAULT_TEMPORAL_CLONE_MAX_PIXELS_PER_VIEW = 8192
# 单次刷新最多新增的 temporal clone anchor 数量。
DEFAULT_TEMPORAL_CLONE_MAX_NEW_ANCHORS = 10000
# 将 anchor 投影到屏幕时用于邻域命中的半径。
DEFAULT_TEMPORAL_CLONE_SCREEN_RADIUS = 12.0
# 一个 anchor 在多视角累计拿到多少票后，才会被认为值得 clone 成 local anchor。
DEFAULT_TEMPORAL_CLONE_VOTE_THRESHOLD = 3
# 候选 offset 的渲染深度与当前表面深度最多允许差多少，超过就不算“贴在表面上”。
DEFAULT_TEMPORAL_CLONE_DEPTH_TOLERANCE = 0.3
# residual voting / MVRV 是否启用 depth-consistency gate；消融时可显式关闭。
DEFAULT_TEMPORAL_CLONE_DEPTH_GATE_ENABLED = True

# temporal 事件（split / death / suppression）日志的打印间隔。
DEFAULT_TEMPORAL_EVENT_LOG_INTERVAL = 500


def _ensure_scaffold_path() -> Path:
    current_dir = Path(__file__).resolve().parent
    scaffold_candidates = (
        current_dir / "Scaffold-GS-main copy",
        current_dir / "Scaffold-GS-main",
        Path(__file__).resolve().parents[2] / "Scaffold-GS-main",
    )
    scaffold_root = next((path for path in scaffold_candidates if path.is_dir()), None)
    if scaffold_root is None:
        raise FileNotFoundError(
            "Scaffold-GS-main not found. Expected one of: "
            + ", ".join(str(path) for path in scaffold_candidates)
        )
    scaffold_root_str = str(scaffold_root)
    if scaffold_root_str not in sys.path:
        sys.path.insert(0, scaffold_root_str)
    return scaffold_root


# 某些 anchor 被判死亡后，对其邻域再做一次抑制的空间半径。
DEFAULT_TEMPORAL_DEATH_SUPPRESS_NEIGHBOR_RADIUS = 0.0


def _mae_threshold_to_psnr(mae_threshold: float) -> float:
    mae_threshold = max(float(mae_threshold), 1e-6)
    return -10.0 * math.log10(mae_threshold * mae_threshold)


def _to_temporal_timestep_tensor(value: Any, device: torch.device | str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    tensor = torch.where(
        tensor >= int(TEMPORAL_ALIVE_SENTINEL),
        torch.full_like(tensor, int(TEMPORAL_ALIVE_SENTINEL)),
        tensor,
    )
    tensor = tensor.clamp(
        min=int(torch.iinfo(TEMPORAL_TIMESTEP_DTYPE).min),
        max=int(torch.iinfo(TEMPORAL_TIMESTEP_DTYPE).max),
    )
    return tensor.to(dtype=TEMPORAL_TIMESTEP_DTYPE)


def _ensure_scaffold_optional_deps() -> None:
    """
    Provide lightweight fallbacks for optional Scaffold-GS dependencies.

    Scaffold-GS imports `colorama` in `scene.dataset_readers` for colored
    progress-bar text only. Training itself does not require it, so we install
    a tiny stub when the package is unavailable.
    """
    try:
        import colorama  # noqa: F401
    except ModuleNotFoundError:
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


_SCAFFOLD_ROOT = _ensure_scaffold_path()
_ensure_scaffold_optional_deps()

from gaussian_renderer import prefilter_voxel as scaffold_prefilter_voxel  # type: ignore  # noqa: E402
from gaussian_renderer import render as scaffold_render  # type: ignore  # noqa: E402
from scene.gaussian_model import GaussianModel as ScaffoldGaussianModel  # type: ignore  # noqa: E402
from utils.loss_utils import l1_loss as scaffold_l1_loss  # type: ignore  # noqa: E402
from utils.loss_utils import ssim as scaffold_ssim  # type: ignore  # noqa: E402
from clsplats.models.irc_modules.consolidation import ConsolidationMixin  # noqa: E402
from clsplats.models.irc_modules.inheritance import InheritanceMixin  # noqa: E402
from clsplats.models.irc_modules.revision import RevisionMixin  # noqa: E402


def _get_scaffold_mlp_payload(gaussians: ScaffoldGaussianModel) -> Dict[str, Any]:
    if hasattr(gaussians, "get_mlp_checkpoint_payload"):
        return gaussians.get_mlp_checkpoint_payload()

    payload: Dict[str, Any] = {
        "opacity_mlp": gaussians.mlp_opacity.state_dict(),
        "cov_mlp": gaussians.mlp_cov.state_dict(),
        "color_mlp": gaussians.mlp_color.state_dict(),
        "temporal_history_mlps": {
            key: module.state_dict() for key, module in getattr(gaussians, "temporal_history_mlps", {}).items()
        },
        "temporal_opacity_mlps": {
            key: module.state_dict() for key, module in getattr(gaussians, "temporal_opacity_mlps", {}).items()
        },
        "temporal_cov_mlps": {
            key: module.state_dict() for key, module in getattr(gaussians, "temporal_cov_mlps", {}).items()
        },
        "temporal_color_mlps": {
            key: module.state_dict() for key, module in getattr(gaussians, "temporal_color_mlps", {}).items()
        },
    }
    if getattr(gaussians, "use_feat_bank", False):
        payload["feature_bank_mlp"] = gaussians.mlp_feature_bank.state_dict()
    if getattr(gaussians, "appearance_dim", 0) > 0 and getattr(gaussians, "embedding_appearance", None) is not None:
        payload["appearance"] = gaussians.embedding_appearance.state_dict()
    return payload


def _load_scaffold_mlp_payload(gaussians: ScaffoldGaussianModel, checkpoint: Dict[str, Any]) -> None:
    if hasattr(gaussians, "load_mlp_checkpoint_payload"):
        gaussians.load_mlp_checkpoint_payload(checkpoint)
        return

    gaussians.mlp_opacity.load_state_dict(checkpoint["opacity_mlp"])
    gaussians.mlp_cov.load_state_dict(checkpoint["cov_mlp"])
    gaussians.mlp_color.load_state_dict(checkpoint["color_mlp"])
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
            if time_step <= 0 or not hasattr(gaussians, "_ensure_temporal_modules_for_time"):
                continue
            gaussians._ensure_temporal_modules_for_time(time_step)
            module = None
            if kind == "history":
                module = gaussians.temporal_history_mlps[key] if key in gaussians.temporal_history_mlps else None
            else:
                module_dict = getattr(gaussians, f"temporal_{kind}_mlps", {})
                module = module_dict[key] if key in module_dict else None
            if module is not None:
                module.load_state_dict(payload_dict[key])
    if getattr(gaussians, "use_feat_bank", False) and "feature_bank_mlp" in checkpoint:
        gaussians.mlp_feature_bank.load_state_dict(checkpoint["feature_bank_mlp"])
    if getattr(gaussians, "appearance_dim", 0) > 0 and "appearance" in checkpoint:
        appearance_state = checkpoint.get("appearance", {})
        weight = appearance_state.get("embedding.weight", None) if isinstance(appearance_state, dict) else None
        if weight is None and isinstance(appearance_state, dict):
            weight = appearance_state.get("weight", None)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            target_cams = int(weight.shape[0])
            if getattr(gaussians, "embedding_appearance", None) is None or int(gaussians.embedding_appearance.embedding.weight.shape[0]) != target_cams:
                gaussians.set_appearance(target_cams)
            current_weight = gaussians.embedding_appearance.embedding.weight.data
            loaded_weight = weight.to(current_weight.device)
            copy_rows = min(current_weight.shape[0], loaded_weight.shape[0])
            current_weight[:copy_rows].copy_(loaded_weight[:copy_rows])
            if current_weight.shape[0] > loaded_weight.shape[0]:
                current_weight[loaded_weight.shape[0]:].zero_()
        else:
            gaussians.embedding_appearance.load_state_dict(checkpoint["appearance"])


@dataclass
class ScaffoldTrainingArgs:
    # 当前时间步的总训练迭代数。
    iterations: int = 30_000

    # anchor 位置学习率的初始值。
    position_lr_init: float = 0.0
    # anchor 位置学习率的最终值。
    position_lr_final: float = 0.0
    # anchor 位置学习率 warmup / delay 的倍率。
    position_lr_delay_mult: float = 0.01
    # anchor 位置学习率从初始值衰减到最终值的步数。
    position_lr_max_steps: int = 30_000

    # offset 学习率的初始值。
    offset_lr_init: float = 0.01
    # offset 学习率的最终值。
    offset_lr_final: float = 0.0001
    # offset 学习率 warmup / delay 的倍率。
    offset_lr_delay_mult: float = 0.01
    # offset 学习率从初始值衰减到最终值的步数。
    offset_lr_max_steps: int = 30_000

    # anchor 特征向量的学习率。
    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    # anchor scaling 参数的学习率。
    scaling_lr: float = 0.007
    # anchor rotation 参数的学习率。
    rotation_lr: float = 0.002

    # opacity MLP 学习率的初始值。
    mlp_opacity_lr_init: float = 0.002
    # opacity MLP 学习率的最终值。
    mlp_opacity_lr_final: float = 0.00002
    # opacity MLP 学习率 warmup / delay 的倍率。
    mlp_opacity_lr_delay_mult: float = 0.01
    # opacity MLP 学习率从初始值衰减到最终值的步数。
    mlp_opacity_lr_max_steps: int = 30_000

    # covariance MLP 学习率的初始值。
    mlp_cov_lr_init: float = 0.004
    # covariance MLP 学习率的最终值。
    mlp_cov_lr_final: float = 0.004
    # covariance MLP 学习率 warmup / delay 的倍率。
    mlp_cov_lr_delay_mult: float = 0.01
    # covariance MLP 学习率从初始值衰减到最终值的步数。
    mlp_cov_lr_max_steps: int = 30_000

    # color MLP 学习率的初始值。
    mlp_color_lr_init: float = 0.008
    # color MLP 学习率的最终值。
    mlp_color_lr_final: float = 0.00005
    # color MLP 学习率 warmup / delay 的倍率。
    mlp_color_lr_delay_mult: float = 0.01
    # color MLP 学习率从初始值衰减到最终值的步数。
    mlp_color_lr_max_steps: int = 30_000

    # feature bank MLP 学习率的初始值。
    mlp_featurebank_lr_init: float = 0.01
    # feature bank MLP 学习率的最终值。
    mlp_featurebank_lr_final: float = 0.00001
    # feature bank MLP 学习率 warmup / delay 的倍率。
    mlp_featurebank_lr_delay_mult: float = 0.01
    # feature bank MLP 学习率从初始值衰减到最终值的步数。
    mlp_featurebank_lr_max_steps: int = 30_000

    # temporal feature bank MLP 学习率的初始值。
    mlp_temporal_bank_lr_init: float = 0.01
    # temporal feature bank MLP 学习率的最终值。
    mlp_temporal_bank_lr_final: float = 0.00001
    # temporal feature bank MLP 学习率 warmup / delay 的倍率。
    mlp_temporal_bank_lr_delay_mult: float = 0.01
    # temporal feature bank MLP 学习率从初始值衰减到最终值的步数。
    mlp_temporal_bank_lr_max_steps: int = 30_000

    # appearance embedding 学习率的初始值。
    appearance_lr_init: float = 0.05
    # appearance embedding 学习率的最终值。
    appearance_lr_final: float = 0.0005
    # appearance embedding 学习率 warmup / delay 的倍率。
    appearance_lr_delay_mult: float = 0.01
    # appearance embedding 学习率从初始值衰减到最终值的步数。
    appearance_lr_max_steps: int = 30_000

    # Scaffold-GS 中控制密集化触发区域比例的参数。
    percent_dense: float = 0.01
    # SSIM 损失在图像重建损失中的权重。
    lambda_dssim: float = 0.2
    # scaling 正则项权重，用于约束 Gaussian 尺度。
    scaling_reg_weight: float = 0.01

    # 从该迭代开始累计 anchor 生长/剪枝所需统计量。
    start_stat: int = 500
    # 从该迭代开始执行 anchor 生长/剪枝更新。
    update_from: int = DEFAULT_UPDATE_FROM
    # anchor 生长/剪枝更新的迭代间隔。
    update_interval: int = DEFAULT_UPDATE_INTERVAL
    # 到该迭代后停止执行 anchor 生长/剪枝更新。
    update_until: int = DEFAULT_UPDATE_UNTIL

    min_opacity: float = DEFAULT_MIN_OPACITY
    # Scaffold-GS anchor 生长/剪枝统计中的成功率阈值。
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    # 梯度超过该阈值的区域会被作为 densify 候选。
    densify_grad_threshold: float = DEFAULT_DENSIFY_GRAD_THRESHOLD

    # 只在 t=0 使用 LightGaussian 风格的重要性剪枝，剪枝对象是 Scaffold-GS anchor。
    t0_importance_prune_enabled: bool = True
    # 剪枝触发迭代；默认在 t=0 原始训练结束时触发。
    t0_importance_prune_iteration: int = -1
    # 剪枝后继续适配的步数，这段时间不再允许 grow/prune。
    t0_importance_adapt_iters: int = DEFAULT_T0_IMPORTANCE_ADAPT_ITERS
    # 删除重要性最低的 anchor 比例。
    t0_importance_prune_ratio: float = DEFAULT_T0_IMPORTANCE_PRUNE_RATIO
    # 重要性分数类型：offset_contribution / important_score / v_important_score / count / opacity。
    t0_importance_score_type: str = "offset_contribution"
    # v_important_score 中体积项的幂。
    t0_importance_v_pow: float = 0.1
    # t>0 时对当前 timestep 的 local anchor 进行重要性剪枝。
    temporal_importance_prune_enabled: bool = True
    # t>0 local anchor 剪枝比例，独立于 t=0 的剪枝比例。
    temporal_importance_prune_ratio: float = DEFAULT_TEMPORAL_IMPORTANCE_PRUNE_RATIO

    # LoRA 低秩分解的 rank。
    lora_rank: int = 4
    # LoRA 缩放系数 alpha。
    lora_alpha: float = 4.0
    # temporal latent 向量的学习率。
    latent_lr: float = 1e-3
    # feature bank LoRA 参数的学习率。
    bank_lora_lr: float = 5e-4
    # opacity MLP LoRA 参数的学习率。
    opacity_lora_lr: float = 5e-4
    # covariance MLP LoRA 参数的学习率。
    cov_lora_lr: float = 2e-4
    # color MLP LoRA 参数的学习率。
    color_lora_lr: float = 1e-3
    # temporal latent 正则项权重。
    lambda_latent: float = 1e-4
    # LoRA 参数正则项权重。
    lambda_lora: float = 1e-5
    # anchor 死亡后，对周边邻域施加额外 suppression 的半径。
    temporal_death_suppress_neighbor_radius: float = DEFAULT_TEMPORAL_DEATH_SUPPRESS_NEIGHBOR_RADIUS
    temporal_mask_opacity_threshold: float = DEFAULT_TEMPORAL_MASK_OPACITY_THRESHOLD
    # temporal 训练前半段视为结构调整阶段；这期间允许 local clone 做结构生长/筛选。
    temporal_structure_until: int = 15_000
    # temporal 训练第一阶段结束迭代数。
    temporal_stage1_until: int = DEFAULT_TEMPORAL_STAGE1_UNTIL
    # temporal 训练第二阶段结束迭代数。
    temporal_stage2_until: int = DEFAULT_TEMPORAL_STAGE2_UNTIL
    # temporal clone 候选集合的刷新间隔。
    temporal_clone_refresh_interval: int = DEFAULT_TEMPORAL_CLONE_REFRESH_INTERVAL
    # 每次刷新 clone 候选时采样的视角数量。
    temporal_clone_num_views: int = DEFAULT_TEMPORAL_CLONE_NUM_VIEWS
    # 为高误差像素查找附近 anchor 时使用的 KNN 数量。
    temporal_clone_knn: int = DEFAULT_TEMPORAL_CLONE_KNN
    # 从 PSNR 最低的像素中取多少比例参与 clone 投票。
    temporal_clone_top_low_psnr_percent: float = DEFAULT_TEMPORAL_CLONE_TOP_LOW_PSNR_PERCENT
    # 参与 clone 投票的最大像素级 PSNR，低于该值才视为坏像素。
    temporal_clone_max_psnr: float = DEFAULT_TEMPORAL_CLONE_MAX_PSNR
    # 每次刷新最多参与投票的可见 anchor 候选数。
    temporal_clone_max_visible_candidates: int = DEFAULT_TEMPORAL_CLONE_MAX_VISIBLE_CANDIDATES
    # 每个视角最多采样多少个高误差像素。
    temporal_clone_max_pixels_per_view: int = DEFAULT_TEMPORAL_CLONE_MAX_PIXELS_PER_VIEW
    # 单次刷新最多新增多少个 local clone anchor。
    temporal_clone_max_new_anchors: int = DEFAULT_TEMPORAL_CLONE_MAX_NEW_ANCHORS
    # 屏幕空间中判定 anchor 命中高误差像素的半径。
    temporal_clone_screen_radius: float = DEFAULT_TEMPORAL_CLONE_SCREEN_RADIUS
    # bootstrap / 多视角累计投票超过该阈值后，anchor 才会被选为 clone parent。
    temporal_clone_vote_threshold: float = DEFAULT_TEMPORAL_CLONE_VOTE_THRESHOLD
    # 是否要求 residual voting 命中的候选 anchor 满足 depth-consistency gate。
    temporal_clone_depth_gate_enabled: bool = DEFAULT_TEMPORAL_CLONE_DEPTH_GATE_ENABLED
    # 候选 offset 与当前像素表面深度的允许误差。
    temporal_clone_depth_tolerance: float = DEFAULT_TEMPORAL_CLONE_DEPTH_TOLERANCE

class EveryTimeRebuildTrainer(
    InheritanceMixin,
    RevisionMixin,
    ConsolidationMixin,
    CLSplatsTrainer,
):
    """
    Scaffold-GS trainer for CL-Splats timesteps.

    By default the trainer keeps the current Scaffold-GS state and continues
    optimization on later timesteps. Set `model.rebuild_every_timestep=true`
    to restore the original full-rebuild behavior that reinitializes from each
    timestep's COLMAP point cloud.
    """

    supports_builtin_eval = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self._scaffold_root = _SCAFFOLD_ROOT
        self._pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
        )
        self._current_scene_key = "t0"
        self.scene_final_gaussian_counts: Dict[str, int] = {}
        self.temporal_adapter_payloads: Dict[int, Dict[str, Any]] = {}
        self._viewpoint_stack: Optional[List[Any]] = None
        self._temporal_clone_phase_active = False
        self._temporal_clone_bootstrap_done = False
        self._temporal_event_log_interval = DEFAULT_TEMPORAL_EVENT_LOG_INTERVAL
        self._temporal_bootstrap_vote_stats: Dict[str, int] = {}
        self._temporal_bootstrap_selected_anchor_ids = torch.empty((0,), dtype=torch.long)
        self._temporal_bootstrap_selected_anchor_mask = torch.empty((0,), dtype=torch.bool)
        self._temporal_votes_current_to_prev = torch.empty((0,), dtype=torch.float32)
        self._temporal_votes_prev_to_current = torch.empty((0,), dtype=torch.float32)
        self._temporal_votes_total = torch.empty((0,), dtype=torch.float32)
        self._faiss_gpu_resources = None
        self._faiss_knn_info_emitted = False
        self._faiss_knn_warning_emitted = False
        self._last_temporal_event_counters: Dict[str, int] = {
            "split_events": 0,
            "death_events": 0,
            "suppression_events": 0,
        }
        self._t0_importance_pruned = False
        self._t0_importance_prune_stats: Dict[str, Any] = {}
        self._temporal_importance_pruned = False
        self._temporal_importance_prune_stats: Dict[str, Any] = {}
        self._t0_visualization_view_indices: Optional[List[int]] = None
        self.gaussians = self._create_scaffold_gaussians()
        logger.info(
            "Initialized Scaffold-GS bridge trainer "
            f"(scaffold_root={str(self._scaffold_root)})"
        )

    def _model_cfg(self) -> Dict[str, Any]:
        model_cfg = self.cfg.get("model", {})
        return {} if model_cfg is None else model_cfg

    def _train_cfg(self) -> Dict[str, Any]:
        train_cfg = self.cfg.get("train", {})
        return {} if train_cfg is None else train_cfg

    def _output_dir_path(self) -> Path:
        return Path(str(self.cfg.get("output_dir", "outputs"))).expanduser().resolve()


    @staticmethod
    def _default_visualization_view_indices(num_views: int) -> List[int]:
        if int(num_views) <= 0:
            return []
        return sorted(set([0, int(num_views) // 2, max(0, int(num_views) - 1)]))

    def _load_t0_visualization_view_indices_from_disk(self, num_t0_cameras: int) -> Optional[List[int]]:
        t0_vis_dir = self._output_dir_path() / "visualizations" / "t0"
        if not t0_vis_dir.is_dir():
            return None

        pattern = re.compile(r"^iter(\d+)_view(\d+)_compare\.png$")
        views_by_iter: Dict[int, set[int]] = {}
        for path in t0_vis_dir.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            view_idx = int(match.group(2))
            views_by_iter.setdefault(iteration, set()).add(view_idx)

        if not views_by_iter:
            return None

        latest_iteration = max(views_by_iter)
        view_indices = sorted(views_by_iter[latest_iteration])
        valid_view_indices = [idx for idx in view_indices if idx < int(num_t0_cameras)]
        invalid_view_indices = [idx for idx in view_indices if idx >= int(num_t0_cameras)]
        if invalid_view_indices:
            logger.warning(
                "Ignoring stale t0 visualization view indices from disk that exceed "
                f"loaded t0 camera count ({int(num_t0_cameras)}): {invalid_view_indices}"
            )
        if valid_view_indices:
            logger.info(
                "Using t0 visualization view indices from disk "
                f"(iter {latest_iteration}): {valid_view_indices}"
            )
            return valid_view_indices
        return None


















    def _create_scaffold_gaussians(self) -> ScaffoldGaussianModel:
        model_cfg = self._model_cfg()
        temporal_enabled = self._temporal_enabled()
        if "appearance_dim" in model_cfg:
            appearance_dim = int(model_cfg.get("appearance_dim", 32))
        else:
            appearance_dim = 0 if temporal_enabled else 32
        return ScaffoldGaussianModel(
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
            temporal_num_times=int(self._train_cfg().get("num_times", 1)),
        )

    def _recreate_gaussians_instance(self) -> None:
        old_id = id(self.gaussians)
        self.gaussians = self._create_scaffold_gaussians()
        logger.info(
            "Scaffold trainer: created fresh GaussianModel instance "
            f"(old_model_id={old_id}, new_model_id={id(self.gaussians)})"
        )

    def _has_initialized_anchors(self) -> bool:
        return (
            hasattr(self.gaussians, "get_anchor")
            and isinstance(self.gaussians.get_anchor, torch.Tensor)
            and self.gaussians.get_anchor.numel() > 0
        )

    def _should_rebuild_timestep(self, timestep: int) -> bool:
        if int(timestep) == 0:
            return True
        model_cfg = self._model_cfg()
        return bool(model_cfg.get("rebuild_every_timestep", False))

    def _temporal_enabled(self) -> bool:
        model_cfg = self._model_cfg()
        train_cfg = self._train_cfg()
        if "enable_temporal_lora" in model_cfg:
            return bool(model_cfg.get("enable_temporal_lora", False))
        return int(train_cfg.get("num_times", 1)) > 1

    def _ensure_appearance_module(self, num_cameras: int) -> None:
        if getattr(self.gaussians, "appearance_dim", 0) <= 0:
            return
        embedding = getattr(self.gaussians, "embedding_appearance", None)
        if embedding is None:
            self.gaussians.set_appearance(int(num_cameras))
            return
        embedding_weight = getattr(embedding, "embedding", None)
        if embedding_weight is None:
            return
        current_count = int(embedding_weight.weight.shape[0])
        if current_count != int(num_cameras):
            logger.warning(
                "Camera count changed across timesteps for appearance embedding "
                f"({current_count} -> {int(num_cameras)}); expanding while preserving existing weights."
            )
            old_weight = embedding_weight.weight.detach().clone()
            self.gaussians.set_appearance(int(num_cameras))
            new_embedding = getattr(self.gaussians, "embedding_appearance", None)
            new_embedding_weight = getattr(new_embedding, "embedding", None)
            if new_embedding_weight is None:
                return
            copy_rows = min(current_count, int(num_cameras))
            new_embedding_weight.weight.data[:copy_rows].copy_(old_weight[:copy_rows])

    @staticmethod
    def _infer_scene_key(
        timestep: int,
        source_info: Dict[str, Any] | None = None,
    ) -> str:
        if source_info is not None:
            timestep_root = str(source_info.get("timestep_root", "")).strip()
            if timestep_root:
                scene_key = os.path.basename(os.path.normpath(timestep_root))
                if scene_key:
                    return scene_key
        return f"t{int(timestep)}"

    def _load_timestep_data(self, dataset, timestep: int) -> None:
        self.current_cameras = dataset.get_cameras(timestep)
        self.current_images = dataset.get_images(timestep)
        self.current_depths = None
        self.active_gaussians_mask = None
        self._current_scene_key = self._infer_scene_key(timestep)

        source_info = None
        if hasattr(dataset, "get_timestep_data_source"):
            source_info = dataset.get_timestep_data_source(timestep)
            self._current_scene_key = self._infer_scene_key(timestep, source_info)
            logger.info(
                f"Timestep {timestep} data source: "
                f"root={source_info['timestep_root']}, "
                f"images={source_info['images_path']}, "
                f"sparse={source_info['sparse_model_path']}, "
                f"point_cloud={source_info['point_cloud_path']}, "
                f"sample_image={source_info['sample_image']} "
                f"({source_info['sample_resolution']})"
            )

        logger.info(
            f"Scaffold mode: loaded {len(self.current_cameras)} cameras "
            f"for timestep {timestep} (scene={self._current_scene_key})."
        )

        if timestep == 0:
            self.t0_cameras = self.current_cameras
            self._t0_visualization_view_indices = self._default_visualization_view_indices(
                len(self.t0_cameras)
            )
        elif self.t0_cameras is None:
            try:
                self.t0_cameras = dataset.get_cameras(0)
                self._t0_visualization_view_indices = self._default_visualization_view_indices(
                    len(self.t0_cameras)
                )
                logger.info(
                    f"Loaded {len(self.t0_cameras)} t0 cameras for fixed t>0 Scaffold visualization poses."
                )
            except Exception as exc:
                logger.warning(f"Could not load t0 cameras for fixed Scaffold visualization poses: {exc}")

    def initialize_from_point_cloud(self, pcd, cameras, spatial_lr_scale: float):
        self.scene_extent = float(spatial_lr_scale)
        self.gaussians = self._create_scaffold_gaussians()
        self._t0_importance_pruned = False
        self._t0_importance_prune_stats = {}
        self.gaussians.set_appearance(len(cameras))
        self.gaussians.create_from_pcd(pcd, spatial_lr_scale)
        logger.info(
            "Initialized Scaffold-GS model from point cloud "
            f"(anchors={int(self.gaussians.get_anchor.shape[0])})"
        )

    def setup_training(self, iterations: int = None):
        train_cfg = self._train_cfg()
        if iterations is None:
            iterations = int(train_cfg.get("iterations", 30000))
        total_iters = int(iterations)
        default_training_args = ScaffoldTrainingArgs(iterations=total_iters)

        self.training_args = ScaffoldTrainingArgs(
            iterations=total_iters,
            position_lr_init=float(train_cfg.get("position_lr_init", 0.0)),
            position_lr_final=float(train_cfg.get("position_lr_final", 0.0)),
            position_lr_delay_mult=float(train_cfg.get("position_lr_delay_mult", 0.01)),
            position_lr_max_steps=int(train_cfg.get("position_lr_max_steps", total_iters)),
            offset_lr_init=float(train_cfg.get("offset_lr_init", 0.01)),
            offset_lr_final=float(train_cfg.get("offset_lr_final", 0.0001)),
            offset_lr_delay_mult=float(train_cfg.get("offset_lr_delay_mult", 0.01)),
            offset_lr_max_steps=int(train_cfg.get("offset_lr_max_steps", total_iters)),
            feature_lr=float(train_cfg.get("feature_lr", 0.0075)),
            opacity_lr=float(train_cfg.get("opacity_lr", 0.02)),
            scaling_lr=float(train_cfg.get("scaling_lr", 0.007)),
            rotation_lr=float(train_cfg.get("rotation_lr", 0.002)),
            mlp_opacity_lr_init=float(train_cfg.get("mlp_opacity_lr_init", 0.002)),
            mlp_opacity_lr_final=float(train_cfg.get("mlp_opacity_lr_final", 0.00002)),
            mlp_opacity_lr_delay_mult=float(train_cfg.get("mlp_opacity_lr_delay_mult", 0.01)),
            mlp_opacity_lr_max_steps=int(train_cfg.get("mlp_opacity_lr_max_steps", total_iters)),
            mlp_cov_lr_init=float(train_cfg.get("mlp_cov_lr_init", 0.004)),
            mlp_cov_lr_final=float(train_cfg.get("mlp_cov_lr_final", 0.004)),
            mlp_cov_lr_delay_mult=float(train_cfg.get("mlp_cov_lr_delay_mult", 0.01)),
            mlp_cov_lr_max_steps=int(train_cfg.get("mlp_cov_lr_max_steps", total_iters)),
            mlp_color_lr_init=float(train_cfg.get("mlp_color_lr_init", 0.008)),
            mlp_color_lr_final=float(train_cfg.get("mlp_color_lr_final", 0.00005)),
            mlp_color_lr_delay_mult=float(train_cfg.get("mlp_color_lr_delay_mult", 0.01)),
            mlp_color_lr_max_steps=int(train_cfg.get("mlp_color_lr_max_steps", total_iters)),
            mlp_featurebank_lr_init=float(train_cfg.get("mlp_featurebank_lr_init", 0.01)),
            mlp_featurebank_lr_final=float(train_cfg.get("mlp_featurebank_lr_final", 0.00001)),
            mlp_featurebank_lr_delay_mult=float(train_cfg.get("mlp_featurebank_lr_delay_mult", 0.01)),
            mlp_featurebank_lr_max_steps=int(train_cfg.get("mlp_featurebank_lr_max_steps", total_iters)),
            mlp_temporal_bank_lr_init=float(train_cfg.get("mlp_temporal_bank_lr_init", 0.01)),
            mlp_temporal_bank_lr_final=float(train_cfg.get("mlp_temporal_bank_lr_final", 0.00001)),
            mlp_temporal_bank_lr_delay_mult=float(train_cfg.get("mlp_temporal_bank_lr_delay_mult", 0.01)),
            mlp_temporal_bank_lr_max_steps=int(train_cfg.get("mlp_temporal_bank_lr_max_steps", total_iters)),
            appearance_lr_init=float(train_cfg.get("appearance_lr_init", 0.05)),
            appearance_lr_final=float(train_cfg.get("appearance_lr_final", 0.0005)),
            appearance_lr_delay_mult=float(train_cfg.get("appearance_lr_delay_mult", 0.01)),
            appearance_lr_max_steps=int(train_cfg.get("appearance_lr_max_steps", total_iters)),
            percent_dense=float(train_cfg.get("percent_dense", 0.01)),
            lambda_dssim=float(train_cfg.get("lambda_dssim", 0.2)),
            scaling_reg_weight=float(train_cfg.get("scaling_reg_weight", 0.01)),
            start_stat=int(train_cfg.get("start_stat", 500)),
            update_from=int(train_cfg.get("update_from", DEFAULT_UPDATE_FROM)),
            update_interval=int(train_cfg.get("update_interval", DEFAULT_UPDATE_INTERVAL)),
            update_until=int(train_cfg.get("update_until", DEFAULT_UPDATE_UNTIL)),
            min_opacity=float(train_cfg.get("min_opacity", DEFAULT_MIN_OPACITY)),
            success_threshold=float(train_cfg.get("success_threshold", DEFAULT_SUCCESS_THRESHOLD)),
            densify_grad_threshold=float(train_cfg.get("densify_grad_threshold", DEFAULT_DENSIFY_GRAD_THRESHOLD)),
            t0_importance_prune_enabled=bool(train_cfg.get("t0_importance_prune_enabled", True)),
            t0_importance_prune_iteration=int(train_cfg.get("t0_importance_prune_iteration", -1)),
            t0_importance_adapt_iters=int(train_cfg.get("t0_importance_adapt_iters", DEFAULT_T0_IMPORTANCE_ADAPT_ITERS)),
            t0_importance_prune_ratio=float(train_cfg.get("t0_importance_prune_ratio", DEFAULT_T0_IMPORTANCE_PRUNE_RATIO)),
            t0_importance_score_type=str(train_cfg.get("t0_importance_score_type", "offset_contribution")),
            t0_importance_v_pow=float(train_cfg.get("t0_importance_v_pow", 0.1)),
            temporal_importance_prune_enabled=bool(train_cfg.get("temporal_importance_prune_enabled", True)),
            temporal_importance_prune_ratio=float(
                train_cfg.get("temporal_importance_prune_ratio", DEFAULT_TEMPORAL_IMPORTANCE_PRUNE_RATIO)
            ),
            lora_rank=int(train_cfg.get("lora_rank", 4)),
            lora_alpha=float(train_cfg.get("lora_alpha", 4.0)),
            latent_lr=float(train_cfg.get("latent_lr", 1e-3)),
            bank_lora_lr=float(train_cfg.get("bank_lora_lr", 5e-4)),
            opacity_lora_lr=float(train_cfg.get("opacity_lora_lr", 5e-4)),
            cov_lora_lr=float(train_cfg.get("cov_lora_lr", 2e-4)),
            color_lora_lr=float(train_cfg.get("color_lora_lr", 1e-3)),
            lambda_latent=float(train_cfg.get("lambda_latent", 1e-4)),
            lambda_lora=float(train_cfg.get("lambda_lora", 1e-5)),
            temporal_death_suppress_neighbor_radius=float(train_cfg.get("temporal_death_suppress_neighbor_radius", DEFAULT_TEMPORAL_DEATH_SUPPRESS_NEIGHBOR_RADIUS)),
            temporal_mask_opacity_threshold=float(train_cfg.get("temporal_mask_opacity_threshold", DEFAULT_TEMPORAL_MASK_OPACITY_THRESHOLD)),
            temporal_structure_until=int(train_cfg.get("temporal_structure_until", 15000)),
            temporal_stage1_until=int(train_cfg.get("temporal_stage1_until", DEFAULT_TEMPORAL_STAGE1_UNTIL)),
            temporal_stage2_until=int(train_cfg.get("temporal_stage2_until", DEFAULT_TEMPORAL_STAGE2_UNTIL)),
            temporal_clone_refresh_interval=int(train_cfg.get("temporal_clone_refresh_interval", DEFAULT_TEMPORAL_CLONE_REFRESH_INTERVAL)),
            temporal_clone_num_views=int(train_cfg.get("temporal_clone_num_views", DEFAULT_TEMPORAL_CLONE_NUM_VIEWS)),
            temporal_clone_knn=int(train_cfg.get("temporal_clone_knn", DEFAULT_TEMPORAL_CLONE_KNN)),
            temporal_clone_top_low_psnr_percent=float(
                train_cfg.get(
                    "temporal_clone_top_low_psnr_percent",
                    train_cfg.get("temporal_clone_top_error_percent", DEFAULT_TEMPORAL_CLONE_TOP_LOW_PSNR_PERCENT),
                )
            ),
            temporal_clone_max_psnr=float(
                train_cfg.get(
                    "temporal_clone_max_psnr",
                    _mae_threshold_to_psnr(train_cfg["temporal_clone_min_error"])
                    if "temporal_clone_min_error" in train_cfg
                    else DEFAULT_TEMPORAL_CLONE_MAX_PSNR,
                )
            ),
            temporal_clone_max_visible_candidates=int(train_cfg.get("temporal_clone_max_visible_candidates", DEFAULT_TEMPORAL_CLONE_MAX_VISIBLE_CANDIDATES)),
            temporal_clone_max_pixels_per_view=int(train_cfg.get("temporal_clone_max_pixels_per_view", DEFAULT_TEMPORAL_CLONE_MAX_PIXELS_PER_VIEW)),
            temporal_clone_max_new_anchors=int(train_cfg.get("temporal_clone_max_new_anchors", DEFAULT_TEMPORAL_CLONE_MAX_NEW_ANCHORS)),
            temporal_clone_screen_radius=float(train_cfg.get("temporal_clone_screen_radius", DEFAULT_TEMPORAL_CLONE_SCREEN_RADIUS)),
            temporal_clone_vote_threshold=float(train_cfg.get("temporal_clone_vote_threshold", DEFAULT_TEMPORAL_CLONE_VOTE_THRESHOLD)),
            temporal_clone_depth_gate_enabled=bool(
                train_cfg.get("temporal_clone_depth_gate_enabled", DEFAULT_TEMPORAL_CLONE_DEPTH_GATE_ENABLED)
            ),
            temporal_clone_depth_tolerance=float(train_cfg.get("temporal_clone_depth_tolerance", DEFAULT_TEMPORAL_CLONE_DEPTH_TOLERANCE)),
        )
        self.gaussians.train()
        self.gaussians.training_setup(self.training_args)
        if self._temporal_enabled() and int(self.timestep) > 0:
            self.gaussians.setup_base_temporal_state(self.training_args)
            logger.info(
                f"Temporal base latent active at timestep {int(self.timestep)}: optimize per-anchor latent "
                f"(dim={int(getattr(self.gaussians, 'temporal_latent_dim', 0))}) together with base Scaffold-GS."
            )
        return self.training_args

















    def prepare_timestep(self, timestep: int, dataset=None) -> None:
        self.timestep = int(timestep)
        self._viewpoint_stack = None
        if int(self.timestep) == 0:
            self._t0_importance_pruned = False
            self._t0_importance_prune_stats = {}
        self._temporal_importance_pruned = False
        self._temporal_importance_prune_stats = {}
        self._temporal_clone_phase_active = False
        self._temporal_clone_bootstrap_done = False
        self._temporal_bootstrap_vote_stats = {}
        self._set_temporal_bootstrap_selected_anchors(None, 0)
        logger.info(f"Preparing timestep {self.timestep} with Scaffold-GS training")

        if dataset is not None:
            self._load_timestep_data(dataset, self.timestep)
            cameras = self.current_cameras or dataset.get_cameras(self.timestep)
            should_rebuild = self._should_rebuild_timestep(self.timestep)
            has_initialized = self._has_initialized_anchors()

            if should_rebuild or not has_initialized:
                scene_info = dataset.get_scene_info(self.timestep)
                if scene_info is None or scene_info.point_cloud is None:
                    raise RuntimeError(
                        f"No point cloud found for timestep {self.timestep}; "
                        "Scaffold-GS rebuild requires per-timestep point cloud."
                    )

                prev_count = int(self.gaussians.get_anchor.shape[0]) if has_initialized else 0
                self._recreate_gaussians_instance()
                spatial_lr_scale = float(
                    (scene_info.nerf_normalization or {}).get("radius", 1.0)
                )
                self.scene_extent = spatial_lr_scale
                self.gaussians.set_appearance(len(cameras))
                self.gaussians.create_from_pcd(scene_info.point_cloud, spatial_lr_scale)
                rebuilt_count = int(self.gaussians.get_anchor.shape[0])
                logger.info(
                    f"Timestep {self.timestep}: rebuilt Scaffold-GS anchors "
                    f"(scene={self._current_scene_key}, prev_anchors={prev_count}, "
                    f"rebuilt_anchors={rebuilt_count}, scene_extent={spatial_lr_scale:.6f})."
                )
            else:
                self._ensure_appearance_module(len(cameras))
                logger.info(
                    f"Timestep {self.timestep}: reusing Scaffold-GS anchors "
                    f"(scene={self._current_scene_key}, anchors={int(self.gaussians.get_anchor.shape[0])})."
                )
            if (
                self._temporal_enabled()
                and int(self.timestep) > 0
                and hasattr(self.gaussians, "reset_temporal_anomaly_state")
            ):
                self.gaussians.reset_temporal_anomaly_state()
            if self._temporal_enabled() and int(self.timestep) > 0:
                logger.info(
                    f"Timestep {int(self.timestep)}: stage 2 will use periodic residual voting."
                )

    def _next_training_view(self):
        if not self.current_cameras:
            raise RuntimeError("No cameras loaded for current timestep.")
        if self._viewpoint_stack is None or len(self._viewpoint_stack) == 0:
            self._viewpoint_stack = list(self.current_cameras)
        idx = random.randint(0, len(self._viewpoint_stack) - 1)
        return self._viewpoint_stack.pop(idx)

    def _lookup_gt_image(self, camera) -> torch.Tensor:
        if hasattr(camera, "original_image") and camera.original_image is not None:
            gt_image = camera.original_image
        elif self.current_images is not None:
            try:
                camera_index = self.current_cameras.index(camera)
            except ValueError as exc:
                raise RuntimeError("Could not locate camera inside current camera list.") from exc
            gt_image = self.current_images[camera_index]
        else:
            raise RuntimeError("Ground-truth image is unavailable for the selected camera.")
        if gt_image.shape[0] > 3:
            gt_image = gt_image[:3]
        return gt_image.to(device=self.device, dtype=torch.float32)






    def _train_step(self, iteration: int, camera, gt_image) -> Dict[str, float]:
        if self.training_args is None:
            raise RuntimeError("setup_training() must be called before _train_step().")

        self.gaussians.update_learning_rate(iteration)
        voxel_visible_mask = scaffold_prefilter_voxel(
            camera,
            self.gaussians,
            self._pipe,
            self.bg_color,
        )
        t0_prune_iter = int(getattr(self.training_args, "t0_importance_prune_iteration", -1))
        if t0_prune_iter < 0:
            t0_prune_iter = int(getattr(self.training_args, "_resolved_t0_importance_prune_iteration", self.training_args.update_until))
        structure_allowed = not (
            int(self.timestep) == 0
            and bool(getattr(self.training_args, "t0_importance_prune_enabled", True))
            and int(iteration) >= int(t0_prune_iter)
        )
        retain_grad = bool(structure_allowed and 0 <= int(iteration) < int(self.training_args.update_until))
        render_pkg = scaffold_render(
            camera,
            self.gaussians,
            self._pipe,
            self.bg_color,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
        )

        rendered_image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        offset_selection_mask = render_pkg["selection_mask"]
        scaling = render_pkg["scaling"]
        opacity = render_pkg["neural_opacity"]

        if rendered_image.shape != gt_image.shape:
            raise RuntimeError(
                "Render/GT image size mismatch: "
                f"timestep={self.timestep}, camera={getattr(camera, 'image_name', 'unknown')}, "
                f"rendered_shape={tuple(rendered_image.shape)}, gt_shape={tuple(gt_image.shape)}."
            )

        loss_l1 = scaffold_l1_loss(rendered_image, gt_image)
        loss_ssim = 1.0 - scaffold_ssim(rendered_image, gt_image)
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (
            (1.0 - self.training_args.lambda_dssim) * loss_l1
            + self.training_args.lambda_dssim * loss_ssim
            + self.training_args.scaling_reg_weight * scaling_reg
        )
        latent_reg = torch.tensor(0.0, device=loss.device)
        loss.backward()

        with torch.no_grad():
            skip_optimizer_step = False
            if (
                structure_allowed
                and
                int(iteration) < int(self.training_args.update_until)
                and int(iteration) > int(self.training_args.start_stat)
            ):
                self.gaussians.training_statis(
                    viewspace_point_tensor,
                    opacity,
                    visibility_filter,
                    offset_selection_mask,
                    voxel_visible_mask,
                )
                if (
                    int(iteration) > int(self.training_args.update_from)
                    and int(iteration) % int(self.training_args.update_interval) == 0
                ):
                    self.gaussians.adjust_anchor(
                        check_interval=int(self.training_args.update_interval),
                        success_threshold=float(self.training_args.success_threshold),
                        grad_threshold=float(self.training_args.densify_grad_threshold),
                        min_opacity=float(self.training_args.min_opacity),
                    )
            elif int(iteration) == int(self.training_args.update_until):
                for attr_name in ("opacity_accum", "offset_gradient_accum", "offset_denom"):
                    if hasattr(self.gaussians, attr_name):
                        delattr(self.gaussians, attr_name)
                torch.cuda.empty_cache()
            if (
                int(self.timestep) == 0
                and bool(getattr(self.training_args, "t0_importance_prune_enabled", True))
                and not self._t0_importance_pruned
                and int(iteration) == int(t0_prune_iter)
            ):
                self._apply_t0_anchor_importance_prune(iteration)
                skip_optimizer_step = True

            if not skip_optimizer_step:
                self.gaussians.optimizer.step()
                self.gaussians.clamp_trainable_scaling()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": float(loss.item()),
            "loss_l1": float(loss_l1.item()),
            "loss_ssim": float(loss_ssim.item()),
            "scaling_reg": float(scaling_reg.item()),
            "latent_reg": float(latent_reg.item()),
        }

    def _train_step_temporal(self, iteration: int, camera, gt_image) -> Dict[str, float]:
        if self.training_args is None:
            raise RuntimeError("setup_temporal_training() must be called before _train_step_temporal().")
        stage_id = self._get_temporal_stage_id(iteration)
        local_training_active = self._ensure_temporal_local_training_mode(stage_id)
        self._ensure_temporal_bootstrap_for_stage(stage_id)
        structure_active = (
            self._temporal_enabled()
            and int(self.timestep) > 0
            and int(stage_id) == 2
        )
        self.gaussians.update_learning_rate(iteration)
        self.gaussians.update_temporal_learning_rate(iteration, self.training_args)
        voxel_visible_mask = scaffold_prefilter_voxel(
            camera,
            self.gaussians,
            self._pipe,
            self.bg_color,
        )
        render_pkg = scaffold_render(
            camera,
            self.gaussians,
            self._pipe,
            self.bg_color,
            visible_mask=voxel_visible_mask,
            retain_grad=bool(local_training_active and structure_active),
        )
        rendered_image = render_pkg["render"]
        viewspace_point_tensor = render_pkg.get("viewspace_points", None)
        visibility_filter = render_pkg.get("visibility_filter", None)
        offset_selection_mask = render_pkg.get("selection_mask", None)
        visible_anchor_ids = render_pkg.get("visible_anchor_ids", None)
        scaling = render_pkg["scaling"]
        opacity = render_pkg.get("neural_opacity", None)

        if rendered_image.shape != gt_image.shape:
            raise RuntimeError(
                "Render/GT image size mismatch: "
                f"timestep={self.timestep}, camera={getattr(camera, 'image_name', 'unknown')}, "
                f"rendered_shape={tuple(rendered_image.shape)}, gt_shape={tuple(gt_image.shape)}."
            )

        loss_l1 = scaffold_l1_loss(rendered_image, gt_image)
        loss_ssim = 1.0 - scaffold_ssim(rendered_image, gt_image)
        scaling_reg = scaling.prod(dim=1).mean()
        _, temporal_mlp_reg = self.gaussians.get_temporal_regularization_loss()
        latent_reg = torch.tensor(0.0, device=loss_l1.device)
        loss = (
            (1.0 - self.training_args.lambda_dssim) * loss_l1
            + self.training_args.lambda_dssim * loss_ssim
            + self.training_args.scaling_reg_weight * scaling_reg
            + self.training_args.lambda_lora * temporal_mlp_reg
        )
        attribute_mask = None
        stage1_local_mask = None
        stage1_structure_mask = None
        stage2_attribute_mask = None
        stage2_geometry_mask = None
        if local_training_active:
            n_anchors = int(self.gaussians.get_anchor.shape[0])
            render_anchor_mask = (
                self.gaussians.get_temporal_render_mask().to(device=self.device, dtype=torch.bool).reshape(n_anchors)
                if hasattr(self.gaussians, "get_temporal_render_mask")
                else torch.ones((n_anchors,), device=self.device, dtype=torch.bool)
            )
            stage1_local_mask = self._get_stage1_local_anchor_mask()
            stage1_structure_mask = self._get_stage1_structure_eligible_anchor_mask()
            stage2_attribute_mask = self._get_stage2_attribute_anchor_mask()
            stage2_geometry_mask = stage2_attribute_mask
            attribute_mask = render_anchor_mask
            self.gaussians.set_temporal_attribute_gradient_mask(attribute_mask)
        loss.backward()

        with torch.no_grad():
            skip_optimizer_step = False
            self.gaussians.apply_temporal_latent_block_gradient_mask(active_anchor_mask=attribute_mask)
            if local_training_active:
                geometry_mask = stage1_local_mask if structure_active else stage2_geometry_mask
                local_mask = attribute_mask
                structure_mask = stage1_structure_mask if structure_active else attribute_mask
                self.gaussians.apply_temporal_local_gradient_mask(active_anchor_mask=geometry_mask)
                if (
                    structure_active
                    and
                    opacity is not None
                    and viewspace_point_tensor is not None
                    and getattr(viewspace_point_tensor, "grad", None) is not None
                    and visibility_filter is not None
                    and offset_selection_mask is not None
                    and int(iteration) > int(self.training_args.start_stat)
                ):
                    self.gaussians.training_statis(
                        viewspace_point_tensor,
                        opacity,
                        visibility_filter,
                        offset_selection_mask,
                        visible_anchor_ids if visible_anchor_ids is not None else voxel_visible_mask,
                        active_anchor_mask=structure_mask,
                    )
                if (
                    structure_active
                    and
                    int(iteration) > int(self.training_args.update_from)
                    and int(iteration) % int(self.training_args.update_interval) == 0
                ):
                    self.gaussians.adjust_anchor(
                        check_interval=int(self.training_args.update_interval),
                        success_threshold=float(self.training_args.success_threshold),
                        grad_threshold=float(self.training_args.densify_grad_threshold),
                        min_opacity=float(self.training_args.min_opacity),
                        active_anchor_mask=structure_mask,
                        death_timestep=int(self.timestep),
                        suppress_neighbor_radius=float(self.training_args.temporal_death_suppress_neighbor_radius),
                        real_prune_temporal=True,
                    )
                if (
                    structure_active
                    and int(self.training_args.temporal_clone_refresh_interval) > 0
                    and int(iteration) % int(self.training_args.temporal_clone_refresh_interval) == 0
                ):
                    self._refresh_temporal_local_clones(iteration, camera, gt_image)
                if (
                    structure_active
                    and not self._temporal_importance_pruned
                    and int(iteration) == int(self.training_args.temporal_stage2_until)
                ):
                    prune_stats = self._apply_temporal_new_anchor_importance_prune(iteration)
                    skip_optimizer_step = int(prune_stats.get("pruned", 0)) > 0
                if not skip_optimizer_step:
                    self.gaussians.optimizer.step()
                    self.gaussians.clamp_trainable_scaling(active_anchor_mask=local_mask)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.gaussians.temporal_optimizer is not None:
                    if not skip_optimizer_step:
                        self.gaussians.temporal_optimizer.step()
                    self.gaussians.temporal_optimizer.zero_grad(set_to_none=True)
            else:
                self.gaussians.set_temporal_attribute_gradient_mask(None)
                if self.gaussians.temporal_optimizer is not None:
                    self.gaussians.temporal_optimizer.step()
                    self.gaussians.temporal_optimizer.zero_grad(set_to_none=True)
        self.gaussians.set_temporal_attribute_gradient_mask(None)

        masked_anchors = 0
        death_timestep = getattr(self.gaussians, "temporal_anchor_death_timestep", None)
        if death_timestep is not None:
            death_timestep = death_timestep.to(device=self.device, dtype=torch.long).reshape(-1)
            masked_anchors = int((death_timestep <= int(self.timestep)).sum().item())
        return {
            "loss": float(loss.item()),
            "loss_l1": float(loss_l1.item()),
            "loss_ssim": float(loss_ssim.item()),
            "scaling_reg": float(scaling_reg.item()),
            "latent_reg": float(latent_reg.item()),
            "lora_reg": float(temporal_mlp_reg.item()),
            "masked_anchors": float(masked_anchors),
            "stage_id": float(stage_id),
            "stage2_active": float(local_training_active and structure_active),
            "stage3_active": float(local_training_active and int(stage_id) == 3),
        }

    def _render_visualization_view(self, camera) -> torch.Tensor:
        voxel_visible_mask = scaffold_prefilter_voxel(
            camera,
            self.gaussians,
            self._pipe,
            self.bg_color,
        )
        render_pkg = scaffold_render(
            camera,
            self.gaussians,
            self._pipe,
            self.bg_color,
            visible_mask=voxel_visible_mask,
            retain_grad=False,
        )
        return render_pkg["render"]

    def _save_visualization(self, iteration: int) -> None:
        if not self.current_cameras:
            return

        output_dir = self.cfg.get("output_dir", "outputs")
        vis_dir = os.path.join(output_dir, "visualizations", f"t{self.timestep}")
        os.makedirs(vis_dir, exist_ok=True)

        render_cameras = self.current_cameras
        render_images = self.current_images
        using_t0_views = int(self.timestep) > 0 and self.t0_cameras is not None
        if using_t0_views:
            render_cameras = self.t0_cameras
            render_images = None
            disk_view_indices = self._load_t0_visualization_view_indices_from_disk(
                len(render_cameras)
            )
            if disk_view_indices is not None:
                self._t0_visualization_view_indices = disk_view_indices
            else:
                self._t0_visualization_view_indices = self._default_visualization_view_indices(
                    len(render_cameras)
                )
            logger.info("Using t0 camera poses for t>0 Scaffold visualization renders")
        elif int(self.timestep) > 0:
            logger.warning(
                "t0 cameras are unavailable; Scaffold visualization is falling back to current timestep poses."
            )

        num_views = len(render_cameras)
        if using_t0_views and self._t0_visualization_view_indices is not None:
            view_indices = list(self._t0_visualization_view_indices)
        else:
            view_indices = self._default_visualization_view_indices(num_views)
        skipped_view_indices = [idx for idx in view_indices if idx >= num_views]
        if skipped_view_indices:
            logger.warning(
                "Skipping t0 visualization view indices missing from loaded t0 cameras: "
                f"{skipped_view_indices} (num_t0_cameras={num_views})"
            )
        view_indices = [idx for idx in view_indices if idx < num_views]
        train_cfg = self._train_cfg()
        special_view_idx = int(train_cfg.get("vis_special_view_idx", -1))

        for idx in view_indices:
            if idx >= num_views:
                continue
            camera = render_cameras[idx]
            gt_image = render_images[idx] if render_images and idx < len(render_images) else None

            with torch.no_grad():
                rendered = self._render_visualization_view(camera)

            rendered_np = rendered.permute(1, 2, 0).detach().cpu().numpy()
            rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
            h, w = rendered_np.shape[:2]

            gt_np = None
            if gt_image is not None:
                gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
                gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)

            gap = 5
            total_width = w if gt_np is None else (2 * w + gap)
            comparison = np.zeros((h, total_width, 3), dtype=np.uint8)
            comparison[:, :w] = rendered_np
            if gt_np is not None:
                comparison[:, w + gap:w + gap + w] = gt_np

            comp_path = os.path.join(vis_dir, f"iter{iteration:05d}_view{idx}_compare.png")
            PILImage.fromarray(comparison).save(comp_path)

        if 0 <= special_view_idx < num_views:
            camera = render_cameras[special_view_idx]
            with torch.no_grad():
                rendered = self._render_visualization_view(camera)
            rendered_np = rendered.permute(1, 2, 0).detach().cpu().numpy()
            rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
            special_path = os.path.join(vis_dir, f"iter{iteration:05d}_special_view{special_view_idx}.png")
            PILImage.fromarray(rendered_np).save(special_path)

        t0_image_path = train_cfg.get("vis_t0_image_path", None)
        if t0_image_path and self.t0_cameras is not None:
            t0_image_name = os.path.basename(t0_image_path)
            t0_camera = None
            for cam in self.t0_cameras:
                if getattr(cam, "image_name", None) == t0_image_name:
                    t0_camera = cam
                    break
            if t0_camera is not None:
                with torch.no_grad():
                    rendered = self._render_visualization_view(t0_camera)
                rendered_np = rendered.permute(1, 2, 0).detach().cpu().numpy()
                rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
                t0_path = os.path.join(vis_dir, f"iter{iteration:05d}_t0_{t0_image_name}")
                PILImage.fromarray(rendered_np).save(t0_path)

        logger.info(f"Saved Scaffold visualizations to {vis_dir}")

    def train(self) -> Dict[str, Any]:
        if not self.current_cameras:
            raise RuntimeError("No cameras loaded. Call prepare_timestep() first.")
        if not hasattr(self.gaussians, "get_anchor") or self.gaussians.get_anchor.numel() == 0:
            raise RuntimeError("No Scaffold-GS anchors initialized for training.")

        train_cfg = self._train_cfg()
        base_iterations = int(train_cfg.get("iterations", 30000))
        incremental_iterations = int(
            train_cfg.get("incremental_iterations", base_iterations)
        )
        log_interval = max(1, int(train_cfg.get("log_interval", 100)))
        iterations = base_iterations if self.timestep == 0 else incremental_iterations
        if int(self.timestep) == 0 and bool(train_cfg.get("t0_importance_prune_enabled", True)):
            configured_prune_iter = int(train_cfg.get("t0_importance_prune_iteration", -1))
            prune_iter = base_iterations if configured_prune_iter < 0 else configured_prune_iter
            prune_iter = max(1, min(int(prune_iter), int(base_iterations)))
            adapt_iters = max(0, int(train_cfg.get("t0_importance_adapt_iters", DEFAULT_T0_IMPORTANCE_ADAPT_ITERS)))
            iterations = max(int(iterations), int(prune_iter) + int(adapt_iters))
        temporal_mode = self._temporal_enabled() and int(self.timestep) > 0
        self._temporal_event_log_interval = max(
            1,
            int(train_cfg.get("temporal_event_log_interval", DEFAULT_TEMPORAL_EVENT_LOG_INTERVAL)),
        )
        self._reset_temporal_event_logging_state()
        if temporal_mode:
            self.setup_temporal_training(iterations=iterations)
        else:
            self.setup_training(iterations=iterations)
        if int(self.timestep) == 0 and self.training_args is not None:
            configured_prune_iter = int(getattr(self.training_args, "t0_importance_prune_iteration", -1))
            prune_iter = base_iterations if configured_prune_iter < 0 else configured_prune_iter
            prune_iter = max(1, min(int(prune_iter), int(base_iterations)))
            setattr(self.training_args, "_resolved_t0_importance_prune_iteration", int(prune_iter))
            if bool(getattr(self.training_args, "t0_importance_prune_enabled", True)):
                logger.info(
                    "[T0ImportancePrune] enabled: "
                    f"prune_iter={int(prune_iter)}, adapt_iters={max(0, int(getattr(self.training_args, 't0_importance_adapt_iters', DEFAULT_T0_IMPORTANCE_ADAPT_ITERS)))}, "
                    f"total_iterations={int(iterations)}, structural updates disabled after prune_iter."
                )
        self._viewpoint_stack = None
        self.history.setdefault("final_losses", [])
        self.history.setdefault("loss_curves", {})
        self.history.setdefault("timestep_summaries", {})

        losses: List[float] = []
        recent_metrics: List[Dict[str, float]] = []
        loss_curve_points: List[Dict[str, float]] = []
        pbar = tqdm(
            range(1, iterations + 1),
            desc=f"[T{self.timestep}] {'TemporalPrefix' if temporal_mode else 'Scaffold'}",
            dynamic_ncols=True,
            leave=True,
        )
        self._training_pbar = pbar

        try:
            for iteration in pbar:
                camera = self._next_training_view()
                gt_image = self._lookup_gt_image(camera)
                metrics = self._train_step_temporal(iteration, camera, gt_image) if temporal_mode else self._train_step(iteration, camera, gt_image)
                losses.append(metrics["loss"])
                recent_metrics.append(metrics)
                if len(recent_metrics) > log_interval:
                    recent_metrics.pop(0)

                if iteration % log_interval == 0 or iteration == iterations:
                    def _avg_recent(metric_key: str) -> Optional[float]:
                        values = [
                            float(m[metric_key])
                            for m in recent_metrics
                            if metric_key in m and isinstance(m[metric_key], (int, float))
                        ]
                        if not values:
                            return None
                        return float(sum(values) / len(values))

                    avg_loss = sum(m["loss"] for m in recent_metrics) / len(recent_metrics)
                    avg_l1 = sum(m["loss_l1"] for m in recent_metrics) / len(recent_metrics)
                    avg_ssim = _avg_recent("loss_ssim")
                    avg_scaling_reg = _avg_recent("scaling_reg")
                    avg_latent_reg = _avg_recent("latent_reg")
                    avg_lora_reg = _avg_recent("lora_reg")
                    avg_masked_anchors = _avg_recent("masked_anchors")
                    avg_stage = _avg_recent("stage_id")
                    anchors = int(self.gaussians.get_anchor.shape[0])
                    stage_id = int(round(float(metrics.get("stage_id", 1.0))))
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "l1": f"{avg_l1:.4f}",
                            "anchors": anchors,
                            "stage": stage_id,
                        }
                    )
                    curve_point: Dict[str, float] = {
                        "iteration": float(iteration),
                        "loss": float(avg_loss),
                        "loss_l1": float(avg_l1),
                        "anchors": float(anchors),
                        "stage_id": float(stage_id),
                    }
                    if avg_ssim is not None:
                        curve_point["loss_ssim"] = float(avg_ssim)
                    if avg_scaling_reg is not None:
                        curve_point["scaling_reg"] = float(avg_scaling_reg)
                    if avg_latent_reg is not None:
                        curve_point["latent_reg"] = float(avg_latent_reg)
                    if avg_lora_reg is not None:
                        curve_point["lora_reg"] = float(avg_lora_reg)
                    if avg_masked_anchors is not None:
                        curve_point["masked_anchors"] = float(avg_masked_anchors)
                    loss_curve_points.append(curve_point)

                    log_terms = [
                        f"[T{self.timestep}] Iter {iteration}/{iterations}",
                        f"loss={avg_loss:.6f}",
                        f"loss_l1={avg_l1:.6f}",
                        f"anchors={anchors}",
                        f"stage={stage_id}",
                    ]
                    if avg_ssim is not None:
                        log_terms.append(f"loss_ssim={avg_ssim:.6f}")
                    if avg_scaling_reg is not None:
                        log_terms.append(f"scaling_reg={avg_scaling_reg:.6f}")
                    if avg_latent_reg is not None:
                        log_terms.append(f"latent_reg={avg_latent_reg:.6f}")
                    if avg_lora_reg is not None:
                        log_terms.append(f"lora_reg={avg_lora_reg:.6f}")
                    if avg_masked_anchors is not None:
                        log_terms.append(f"masked_anchors={avg_masked_anchors:.2f}")
                    if avg_stage is not None:
                        log_terms.append(f"avg_stage={avg_stage:.2f}")
                    self._log_to_file_only(", ".join(log_terms))
                if temporal_mode and (
                    iteration % int(self._temporal_event_log_interval) == 0
                    or iteration == iterations
                ):
                    self._log_temporal_event_summary(iteration, iterations)

                vis_interval = int(train_cfg.get("vis_interval", 1000))
                if vis_interval > 0 and iteration % vis_interval == 0 and iteration > 0:
                    self._save_visualization(iteration)

            final_anchors = int(self.gaussians.get_anchor.shape[0])
            scene_key = self._current_scene_key or f"t{int(self.timestep)}"
            self.scene_final_gaussian_counts[scene_key] = final_anchors

            stats = {
                "timestep": int(self.timestep),
                "final_loss": losses[-1] if losses else 0.0,
                "avg_loss": sum(losses) / len(losses) if losses else 0.0,
                "num_gaussians": final_anchors,
                "scene_key": scene_key,
                "scene_final_gaussian_counts": dict(self.scene_final_gaussian_counts),
                "local_optimization_enabled": bool(self._temporal_clone_phase_active),
                "temporal_mode": temporal_mode,
                "temporal_clone_phase_active": bool(self._temporal_clone_phase_active),
                "bootstrap_vote_stats": dict(self._temporal_bootstrap_vote_stats),
                "t0_importance_prune_stats": dict(self._t0_importance_prune_stats),
                "temporal_importance_prune_stats": dict(self._temporal_importance_prune_stats),
                "iterations": int(iterations),
                "log_interval": int(log_interval),
                "loss_curve_points": loss_curve_points,
            }
            self.history["losses"].append(stats["avg_loss"])
            self.history["final_losses"].append(stats["final_loss"])
            self.history["num_gaussians"].append(stats["num_gaussians"])
            self.history["loss_curves"][str(self.timestep)] = loss_curve_points
            self.history["timestep_summaries"][str(self.timestep)] = {
                "timestep": int(stats["timestep"]),
                "scene_key": str(scene_key),
                "iterations": int(iterations),
                "log_interval": int(log_interval),
                "avg_loss": float(stats["avg_loss"]),
                "final_loss": float(stats["final_loss"]),
                "num_gaussians": int(stats["num_gaussians"]),
                "temporal_mode": bool(temporal_mode),
                "temporal_clone_phase_active": bool(self._temporal_clone_phase_active),
                "bootstrap_vote_stats": dict(self._temporal_bootstrap_vote_stats),
                "t0_importance_prune_stats": dict(self._t0_importance_prune_stats),
                "temporal_importance_prune_stats": dict(self._temporal_importance_prune_stats),
            }
            self._log_to_file_only(
                f"[T{self.timestep}] Summary: final_loss={stats['final_loss']:.6f}, "
                f"avg_loss={stats['avg_loss']:.6f}, anchors={final_anchors}, "
                f"log_points={len(loss_curve_points)}"
            )

            summary = ", ".join(
                f"{key}:{value}" for key, value in self.scene_final_gaussian_counts.items()
            )
            logger.info(
                "[SceneGaussianStats] "
                f"scene={scene_key}, timestep={self.timestep}, final_anchors={final_anchors}"
            )
            logger.info(
                "[SceneGaussianStats] "
                f"per_scene_final_anchors={{{summary}}}"
            )
            if bool(train_cfg.get("vis_generate_gif", True)) and self.timestep > 0:
                self.generate_training_gif()
            return stats
        finally:
            self._training_pbar = None

    def save_checkpoint(
        self,
        path: str,
        ply_path_override: Optional[str] = None,
        save_external_ply: bool = True,
        compact_for_eval: bool = False,
    ) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if self._temporal_enabled():
            self.temporal_adapter_payloads[int(self.timestep)] = self.gaussians.get_temporal_checkpoint_payload(
                time_step=int(self.timestep)
            )
            self._sync_temporal_adapter_payloads()
            self._log_temporal_lifetime_distribution(
                f"before save_checkpoint t={int(self.timestep)}"
            )
        temporal_adapter_payloads = {
            int(key): dict(value) if isinstance(value, dict) else value
            for key, value in self.temporal_adapter_payloads.items()
        }
        if bool(compact_for_eval):
            for payload in temporal_adapter_payloads.values():
                if not isinstance(payload, dict):
                    continue
                for dropped_key in (
                    "last_opacity_max",
                    "local_mask",
                    "local_parent_ids",
                    "anomaly_score",
                    "anomaly_count",
                    "anomaly_candidate_mask",
                    "split_suppression_mask",
                ):
                    payload.pop(dropped_key, None)
        checkpoint = {
            "timestep": int(self.timestep),
            "history": self.history,
            "config": omegaconf.OmegaConf.to_container(self.cfg, resolve=True),
            "trainer_type": "scaffold_gs",
            "supports_builtin_eval": True,
            "scene_final_gaussian_counts": dict(self.scene_final_gaussian_counts),
            "scene_extent": float(getattr(self, "scene_extent", 1.0)),
            "temporal_enabled": bool(self._temporal_enabled()),
            "temporal_adapter_payloads": temporal_adapter_payloads,
        }
        if self._temporal_enabled():
            birth_timestep = getattr(self.gaussians, "temporal_anchor_birth_timestep", None)
            death_timestep = getattr(self.gaussians, "temporal_anchor_death_timestep", None)
            local_mask = getattr(self.gaussians, "temporal_local_mask", None)
            local_parent_ids = getattr(self.gaussians, "temporal_local_parent_ids", None)
            split_suppression_mask = getattr(self.gaussians, "temporal_split_suppression_mask", None)
            if isinstance(birth_timestep, torch.Tensor):
                checkpoint["anchor_birth_timestep"] = birth_timestep.detach().cpu()
            if isinstance(death_timestep, torch.Tensor):
                checkpoint["anchor_death_timestep"] = death_timestep.detach().cpu()
            if not bool(compact_for_eval) and isinstance(local_mask, torch.Tensor):
                checkpoint["temporal_local_mask"] = local_mask.detach().cpu()
            if not bool(compact_for_eval) and isinstance(local_parent_ids, torch.Tensor):
                checkpoint["temporal_local_parent_ids"] = local_parent_ids.detach().cpu()
            if not bool(compact_for_eval) and isinstance(split_suppression_mask, torch.Tensor):
                checkpoint["temporal_split_suppression_mask"] = split_suppression_mask.detach().cpu()
        ply_path = str(ply_path_override) if ply_path_override is not None else path.replace(".pt", ".ply")
        self.gaussians.save_ply(ply_path)
        checkpoint["ply_path"] = ply_path
        checkpoint["mlp_state"] = _get_scaffold_mlp_payload(self.gaussians)
        torch.save(checkpoint, path)
        legacy_mlp_dir = path.replace(".pt", "_mlp")
        if os.path.isdir(legacy_mlp_dir):
            shutil.rmtree(legacy_mlp_dir, ignore_errors=True)
        logger.info(f"Saved Scaffold-GS checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.timestep = int(checkpoint.get("timestep", 0))
        self.history = checkpoint.get("history", self.history)
        self.scene_final_gaussian_counts = checkpoint.get(
            "scene_final_gaussian_counts",
            self.scene_final_gaussian_counts,
        )
        raw_temporal_payloads = checkpoint.get("temporal_adapter_payloads", {}) or {}
        self.temporal_adapter_payloads = {int(k): v for k, v in raw_temporal_payloads.items()}
        self.scene_extent = float(checkpoint.get("scene_extent", 1.0))
        self.gaussians = self._create_scaffold_gaussians()

        ply_path = checkpoint.get("ply_path", path.replace(".pt", ".ply"))
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Scaffold-GS checkpoint PLY not found: {ply_path}")
        self.gaussians.load_ply_sparse_gaussian(ply_path)

        mlp_state = checkpoint.get("mlp_state", None)
        if not isinstance(mlp_state, dict):
            raise KeyError(
                "Scaffold-GS checkpoint is missing embedded 'mlp_state'. "
                "Legacy _mlp sidecar directories are no longer supported."
            )
        if self.gaussians.appearance_dim > 0:
            appearance_state = mlp_state.get("appearance", {})
            weight = appearance_state.get("embedding.weight", None) if isinstance(appearance_state, dict) else None
            if weight is None and isinstance(appearance_state, dict):
                weight = appearance_state.get("weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                self.gaussians.set_appearance(int(weight.shape[0]))
        _load_scaffold_mlp_payload(self.gaussians, mlp_state)
        temporal_payload = self.temporal_adapter_payloads.get(int(self.timestep), None)
        if (
            bool(checkpoint.get("temporal_enabled", False))
            and int(self.timestep) > 0
            and temporal_payload is not None
        ):
            temporal_args = self.setup_training(iterations=int(self._train_cfg().get("incremental_iterations", self._train_cfg().get("iterations", 30000))))
            self._temporal_clone_phase_active = False
            self.gaussians.load_temporal_checkpoint_payload(
                temporal_payload,
                temporal_args,
                time_step=int(self.timestep),
            )
        if bool(checkpoint.get("temporal_enabled", False)):
            n_anchors = int(self.gaussians.get_anchor.shape[0])
            birth_timestep = checkpoint.get("anchor_birth_timestep", None)
            if birth_timestep is not None:
                birth_timestep = _to_temporal_timestep_tensor(birth_timestep, self.device).reshape(-1)
                if int(birth_timestep.shape[0]) == n_anchors:
                    self.gaussians.temporal_anchor_birth_timestep = birth_timestep
            death_timestep = checkpoint.get("anchor_death_timestep", None)
            if death_timestep is not None:
                death_timestep = _to_temporal_timestep_tensor(death_timestep, self.device).reshape(-1)
                if int(death_timestep.shape[0]) == n_anchors:
                    self.gaussians.temporal_anchor_death_timestep = death_timestep
            local_mask = checkpoint.get("temporal_local_mask", None)
            if local_mask is None and isinstance(temporal_payload, dict):
                local_mask = temporal_payload.get("local_mask", None)
            if local_mask is not None:
                local_mask = torch.as_tensor(local_mask, device=self.device, dtype=torch.bool).reshape(-1)
                if int(local_mask.shape[0]) == n_anchors:
                    self.gaussians.temporal_local_mask = local_mask
            local_parent_ids = checkpoint.get("temporal_local_parent_ids", None)
            if local_parent_ids is None and isinstance(temporal_payload, dict):
                local_parent_ids = temporal_payload.get("local_parent_ids", None)
            if local_parent_ids is not None:
                local_parent_ids = torch.as_tensor(local_parent_ids, device=self.device, dtype=torch.long).reshape(-1)
                if int(local_parent_ids.shape[0]) == n_anchors:
                    self.gaussians.temporal_local_parent_ids = local_parent_ids
            split_suppression_mask = checkpoint.get("temporal_split_suppression_mask", None)
            if split_suppression_mask is None and isinstance(temporal_payload, dict):
                split_suppression_mask = temporal_payload.get("split_suppression_mask", None)
            if split_suppression_mask is not None:
                split_suppression_mask = torch.as_tensor(split_suppression_mask, device=self.device, dtype=torch.bool).reshape(-1)
                if int(split_suppression_mask.shape[0]) == n_anchors:
                    self.gaussians.temporal_split_suppression_mask = split_suppression_mask
        logger.info(f"Loaded Scaffold-GS checkpoint from {path}")

    def save_current_ply(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.gaussians.save_ply(path)


IRCGSTrainer = EveryTimeRebuildTrainer
Pure3DGSTrainer = EveryTimeRebuildTrainer
