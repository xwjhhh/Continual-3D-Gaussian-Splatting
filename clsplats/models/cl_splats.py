"""
CL-Splats Trainer for continual learning of Gaussian Splatting.

This module implements the complete training pipeline that integrates
all components: change detection, lifting, sampling, pruning, and optimization.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import os
import random
import re
from contextlib import suppress

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFile
from loguru import logger
import omegaconf
import wandb

from tqdm import tqdm

from clsplats.dataset.colmap_reader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat, read_intrinsics_text, read_extrinsics_text
from clsplats.utils.graphics_utils import getWorld2View2, getProjectionMatrix

from clsplats.representation.gaussian_model import GaussianModel
from clsplats.representation.model_factory import (
    create_model as create_representation_model,
    get_available_models as get_available_representations,
)
from clsplats.change_detection.dinov2_detector import DinoV2Detector
from clsplats.utils.majority_vote_lifter import MajorityVoteLifter
from clsplats.sampling.gaussian_sampler import GaussianSampler
from clsplats.pruning.sphere_pruner import SpherePruner
from clsplats.utils.loss_utils import combined_loss, l1_loss

# Try to import renderer
try:
    from clsplats.rendering.renderer import render
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    logger.warning("Renderer not available - install diff_gaussian_rasterization")

# Try to import tile mask computation for Local Optimization
try:
    from diff_gaussian_rasterization import compute_tile_mask_from_active_gaussians, SparseGaussianAdam
    TILE_MASK_AVAILABLE = True
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    TILE_MASK_AVAILABLE = False
    SPARSE_ADAM_AVAILABLE = False
    compute_tile_mask_from_active_gaussians = None
    SparseGaussianAdam = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TrainingArgs:
    """Training arguments for GaussianModel.training_setup()"""
    iterations: int = 30000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    exposure_lr_init: float = 0.01
    exposure_lr_final: float = 0.001
    exposure_lr_delay_steps: int = 0
    exposure_lr_delay_mult: float = 0.0


class CLSplatsTrainer:
    """
    Trainer for CL-Splats continual learning pipeline.
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        """Initialize the CL-Splats trainer."""
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestep = 0

        # model.name now represents training method (e.g. cl-splats).
        # Representation is selected via model.representation.
        model_cfg = cfg.get("model", {})
        if model_cfg is None:
            model_cfg = {}
        method_name = model_cfg.get("name", "cl-splats")
        available_representations = get_available_representations()
        legacy_representation_name = (
            method_name if method_name in available_representations else None
        )

        if legacy_representation_name is not None:
            representation_name = legacy_representation_name
            self.method_name = "cl-splats"
            logger.warning(
                "Deprecated config detected: `model.name` is treated as representation. "
                "Please use `model.name=cl-splats` and `model.representation=...`."
            )
        else:
            representation_name = model_cfg.get("representation", "gaussian")
            self.method_name = method_name

        representation_cfg = omegaconf.OmegaConf.create({})
        for key, value in model_cfg.items():
            representation_cfg[key] = value
        representation_cfg["name"] = representation_name

        self.gaussians = create_representation_model(representation_cfg)
        logger.info(
            f"Using method '{self.method_name}' with representation '{representation_name}' "
            f"(available representations: {', '.join(available_representations)})"
        )
        
        # Scene extent for densification
        self.scene_extent = 1.0
        
        # Background color
        self.bg_color = torch.tensor(
            [1, 1, 1] if cfg.get("white_background", False) else [0, 0, 0],
            dtype=torch.float32,
            device=self.device
        )
        
        # Initialize components (lazy loading)
        self._change_detector = None
        self._lifter = None
        self._sampler = None
        self._pruner = None
        
        # Training state
        self.current_cameras = None
        self.current_images = None
        self.current_depths = None
        self.change_masks = None
        
        self.active_gaussians_mask = None
        self.training_args = None
        
        # T0 cameras for fixed visualization/test rendering
        self.t0_cameras = None
        self.t0_visualization_view_indices = None
        
        # History tracking
        self.history = {
            "losses": [],
            "num_gaussians": [],
            "num_pruned": [],
        }
        
        # Log file path for file-only logging
        runtime_cfg = cfg.get("runtime", {})
        if runtime_cfg is None:
            runtime_cfg = {}
        self._log_file_path = runtime_cfg.get(
            "training_log_file",
            os.path.join(cfg.get("output_dir", "outputs"), "training_logs", "train.log"),
        )

    def _log_to_file_only(self, message: str) -> None:
        """Write log message to file only (not console)."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"{timestamp} | INFO     | clsplats.models.cl_splats:train - {message}\n"
        try:
            log_dir = os.path.dirname(self._log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass  # Silently ignore file write errors

    def _emit_progress_message(self, message: str) -> None:
        """Emit progress to console and persist it to training log file."""
        self._log_to_file_only(message)
        if hasattr(self, "_training_pbar") and self._training_pbar is not None:
            tqdm.write(message)
        else:
            logger.info(message)

    @property
    def change_detector(self) -> DinoV2Detector:
        """Lazy initialization of change detector."""
        if self._change_detector is None:
            cd_cfg = self.cfg.get("change_detection", {})
            self._change_detector = DinoV2Detector(cd_cfg)
        return self._change_detector

    @property
    def lifter(self) -> MajorityVoteLifter:
        """Lazy initialization of lifter."""
        if self._lifter is None:
            lifter_cfg = self.cfg.get("lifter", {})
            self._lifter = MajorityVoteLifter(lifter_cfg)
        return self._lifter

    @property
    def sampler(self) -> GaussianSampler:
        """Lazy initialization of sampler."""
        if self._sampler is None:
            sampler_cfg = self.cfg.get("sampling", {})
            self._sampler = GaussianSampler(sampler_cfg, self.lifter)
        return self._sampler

    @property
    def pruner(self) -> SpherePruner:
        """Lazy initialization of pruner."""
        if self._pruner is None:
            pruner_cfg = self.cfg.get("pruning", {})
            self._pruner = SpherePruner(pruner_cfg)
        return self._pruner

    def initialize_from_point_cloud(self, pcd, cameras, spatial_lr_scale: float):
        """Initialize Gaussians from point cloud."""
        self.scene_extent = spatial_lr_scale
        self.gaussians.create_from_pcd(pcd, cameras, spatial_lr_scale)
        logger.info(f"Initialized {self.gaussians.get_xyz.shape[0]} Gaussians")

    def setup_training(self, iterations: int = None):
        """Setup optimizer and training parameters."""
        train_cfg = self.cfg.get("train", {})
        
        # Use provided iterations or default from config (30000)
        total_iters = iterations if iterations is not None else train_cfg.get("iterations", 30000)
        
        self.training_args = TrainingArgs(
            iterations=total_iters,
            position_lr_init=train_cfg.get("lr_position", 0.00016),
            position_lr_final=train_cfg.get("lr_position_final", 0.0000016),
            position_lr_delay_mult=train_cfg.get("lr_delay_mult", 0.01),
            position_lr_max_steps=total_iters,
            feature_lr=train_cfg.get("lr_feature", 0.0025),
            opacity_lr=train_cfg.get("lr_opacity", 0.05),
            scaling_lr=train_cfg.get("lr_scaling", 0.005),
            rotation_lr=train_cfg.get("lr_rotation", 0.001),
            percent_dense=train_cfg.get("percent_dense", 0.01),
            densification_interval=train_cfg.get("densification_interval", 100),
            densify_from_iter=train_cfg.get("densify_from_iter", 500),
            densify_until_iter=train_cfg.get("densify_until_iter", 15000),
            densify_grad_threshold=train_cfg.get("densify_grad_threshold", 0.0002),
        )
        
        # Set up exposure for cameras if available (needed when resuming from checkpoint)
        if hasattr(self, 'current_cameras') and self.current_cameras is not None:
            if not hasattr(self.gaussians, '_exposure') or self.gaussians._exposure is None:
                self.gaussians.set_exposure_for_cameras(self.current_cameras)
        
        self.gaussians.training_setup(self.training_args)

    def prepare_timestep(self, timestep: int, dataset=None) -> None:
        """Prepare for training at a new timestep."""
        train_cfg = self.cfg.get("train", {})
        num_times = train_cfg.get("num_times", 10)
        
        self.timestep = timestep
        logger.info(f"Preparing timestep {timestep}")
        
        # Reset sphere fitting state for new timestep
        self._spheres_fitted = False
        
        if dataset is not None:
            self._load_timestep_data(dataset, timestep)

    def _load_timestep_data(self, dataset, timestep: int) -> None:
        """Load data for the current timestep."""
        self.current_cameras = dataset.get_cameras(timestep)
        self.current_images = dataset.get_images(timestep)

        if hasattr(dataset, "get_timestep_data_source"):
            source_info = dataset.get_timestep_data_source(timestep)
            logger.info(
                f"Timestep {timestep} data source: "
                f"root={source_info['timestep_root']}, "
                f"images={source_info['images_path']}, "
                f"sparse={source_info['sparse_model_path']}, "
                f"point_cloud={source_info['point_cloud_path']}, "
                f"sample_image={source_info['sample_image']} "
                f"({source_info['sample_resolution']})"
            )
        
        logger.info(f"Loaded {len(self.current_cameras)} cameras for timestep {timestep}")
        
        # Save t0 cameras so t>0 visualization/test renders use the same views
        if timestep == 0:
            self.t0_cameras = self.current_cameras
            num_t0_views = len(self.t0_cameras)
            self.t0_visualization_view_indices = self._default_visualization_view_indices(num_t0_views)
        elif self.t0_cameras is None:
            try:
                self.t0_cameras = dataset.get_cameras(0)
                num_t0_views = len(self.t0_cameras)
                self.t0_visualization_view_indices = self._default_visualization_view_indices(num_t0_views)
                logger.info(
                    f"Loaded {len(self.t0_cameras)} t0 cameras for fixed t>0 visualization views"
                )
            except Exception as exc:
                logger.warning(f"Could not load t0 cameras for fixed visualization views: {exc}")
        
        # Detect changes if not the first timestep
        if timestep > 0 and dataset.get_num_timesteps() > 1:
            # Render images from current cameras using the existing model
            # Then compare rendered images with observed images to detect changes
            rendered_images, rendered_depths = self._render_from_cameras(self.current_cameras)
            self.current_depths = rendered_depths
            
            if len(rendered_images) > 0 and len(self.current_images) > 0:
                self.change_masks = self._detect_changes(rendered_images, self.current_images)
                logger.info(f"Detected changes in {len(self.change_masks)} views")
                
                # Compute active Gaussian mask using lifter (2D鈫?D lifting)
                self._compute_active_gaussians_mask()
            else:
                self.change_masks = None
                self.active_gaussians_mask = None
        else:
            self.change_masks = None
            self.current_depths = None
            self.active_gaussians_mask = None
            logger.info("First timestep, no change detection needed")
    
    def _render_from_cameras(self, cameras) -> List[torch.Tensor]:
        """Render images from the current Gaussian model for each camera."""
        from clsplats.rendering import render
        
        rendered_images = []
        rendered_depths = []
        bg_color = torch.tensor([1, 1, 1] if self.cfg.get("white_background", False) else [0, 0, 0], 
                                dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            for camera in cameras:
                result = render(
                    camera,
                    self.gaussians,
                    bg_color,
                )
                rendered_images.append(result["render"])
                # Capture inverse depths (1/z) - result["invdepths"] is [1, H, W]
                rendered_depths.append(result["invdepths"])
        
        return rendered_images, rendered_depths
    
    def _compute_active_gaussians_mask(self) -> None:
        """
        Compute active Gaussian mask by lifting 2D change masks to 3D.
        
        Uses the lifter module to determine which Gaussians are in changed regions
        based on majority voting across multiple views.
        """
        if self.change_masks is None or len(self.change_masks) == 0:
            self.active_gaussians_mask = None
            return
        
        if self.gaussians.get_xyz.shape[0] == 0:
            self.active_gaussians_mask = None
            return
        
        # Get 3D positions of all Gaussians
        xyz = self.gaussians.get_xyz.detach()
        
        # Use lifter to lift 2D change masks to 3D
        self.active_gaussians_mask = self.lifter.lift(
            points_3d=xyz,
            change_masks_2d=self.change_masks,
            cameras=self.current_cameras,
            depth_maps=self.current_depths
        )
        
        num_active = self.active_gaussians_mask.sum().item()
        total = self.active_gaussians_mask.shape[0]
        msg = f"Local Optimization: {num_active}/{total} Gaussians active ({100*num_active/total:.1f}%)"
        self._emit_progress_message(msg)

    def _detect_changes(
        self, 
        prev_images: List[torch.Tensor], 
        curr_images: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Detect changes between two sets of images."""
        change_masks = []
        for prev_img, curr_img in zip(prev_images, curr_images):
            # Images are [C, H, W], detector expects [H, W, C]
            prev_hwc = prev_img.permute(1, 2, 0)
            curr_hwc = curr_img.permute(1, 2, 0)
            mask = self.change_detector.predict_change_mask(prev_hwc, curr_hwc)
            change_masks.append(mask)
        return change_masks

    def _train_step(self, iteration: int, camera, gt_image) -> Dict[str, float]:
        """Execute a single training iteration."""
        metrics = {}
        
        if not RENDERER_AVAILABLE:
            return {"loss": 0.0}
        
        # Compute tile_mask from active_gaussian_mask for additional optimization
        tile_mask = None
        if self.active_gaussians_mask is not None and TILE_MASK_AVAILABLE:
            # We need means2D and radii from a preliminary render to compute tile_mask
            # For efficiency, we skip tile_mask on first few iterations or compute it periodically
            # Here we pass None and let the CUDA kernel handle it
            pass
        
        # Render with Local Optimization mask if available (CL-Splats)
        render_result = render(
            camera, 
            self.gaussians, 
            self.bg_color,
            active_gaussian_mask=self.active_gaussians_mask,
            tile_mask=tile_mask,
        )
        rendered_image = render_result["render"]
        viewspace_points = render_result["viewspace_points"]
        visibility_filter = render_result["visibility_filter"]
        radii = render_result["radii"]

        if rendered_image.shape != gt_image.shape:
            raise RuntimeError(
                "Render/GT image size mismatch: "
                f"timestep={self.timestep}, camera={getattr(camera, 'image_name', 'unknown')}, "
                f"rendered_shape={tuple(rendered_image.shape)}, gt_shape={tuple(gt_image.shape)}. "
                "Please check dataset source paths and image resolutions in logs."
            )
        
        # Compute loss
        train_cfg = self.cfg.get("train", {})
        lambda_dssim = train_cfg.get("lambda_dssim", 0.2)
        loss = combined_loss(rendered_image, gt_image, lambda_dssim)
        metrics["loss"] = loss.item()
        
        # Backward
        loss.backward()
        
        with torch.no_grad():
            # CL-Splats: Zero gradients for inactive Gaussians before any updates
            # This ensures frozen Gaussians receive absolutely no parameter updates
            if self.active_gaussians_mask is not None and self.timestep > 0:
                if self.active_gaussians_mask.shape[0] == self.gaussians.get_xyz.shape[0]:
                    self.gaussians.zero_gradients_for_inactive(self.active_gaussians_mask)
            
            # Update learning rate
            self.gaussians.update_learning_rate(iteration)
            
            # Track if Gaussian count changed (for mask invalidation)
            num_gaussians_before = self.gaussians.get_xyz.shape[0]
            
            # Densification
            if iteration < self.training_args.densify_until_iter:
                # Track max radii for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                self.gaussians.add_densification_stats(
                    viewspace_points, visibility_filter,
                    active_mask=self.active_gaussians_mask if self.timestep > 0 else None
                )
                
                if iteration > self.training_args.densify_from_iter and \
                   iteration % self.training_args.densification_interval == 0:
                    # Keep size-based pruning smooth: no hard switch at opacity_reset_interval.
                    size_threshold = None
                    
                    # CL-Splats: Pass active_mask to densify_and_prune
                    # Only active Gaussians can be pruned by opacity
                    active_mask_for_prune = None
                    if self.timestep > 0:
                        # Recompute mask if it was invalidated by previous densification
                        if self.active_gaussians_mask is None and self.change_masks is not None:
                            self._compute_active_gaussians_mask()
                        
                        if self.active_gaussians_mask is not None and \
                           self.active_gaussians_mask.shape[0] == self.gaussians.get_xyz.shape[0]:
                            active_mask_for_prune = self.active_gaussians_mask
                    
                    num_gaussians_before = self.gaussians.get_xyz.shape[0]
                    # CL-Splats: Capture updated mask from densify_and_prune
                    # This ensures new points inherit active status correctly
                    updated_mask = self.gaussians.densify_and_prune(
                        self.training_args.densify_grad_threshold,
                        0.005,  # min_opacity
                        self.scene_extent,
                        size_threshold,
                        radii,
                        active_mask=active_mask_for_prune,
                    )
                    
                    # Use the updated mask instead of recomputing from scratch
                    if updated_mask is not None:
                        self.active_gaussians_mask = updated_mask
                    elif self.gaussians.get_xyz.shape[0] != num_gaussians_before:
                        # Fallback: invalidate mask if count changed but no mask returned
                        self.active_gaussians_mask = None
                
                # Opacity reset intentionally removed to avoid scheduled quality drops
                # around opacity_reset_interval (e.g. iter 3000 on t0).
            
            # CL-Splats Sphere Pruning: Check every 15 iterations if active Gaussians
            # have moved outside their bounding spheres (per paper)
            sphere_prune_interval = train_cfg.get("sphere_prune_interval", 15)
            if self.timestep > 0 and iteration % sphere_prune_interval == 0:
                # Only apply if mask is valid
                if self.active_gaussians_mask is not None and \
                   self.active_gaussians_mask.shape[0] == self.gaussians.get_xyz.shape[0]:
                    self._apply_sphere_pruning()
            
            # Check if Gaussian count changed - invalidate mask if so
            num_gaussians_after = self.gaussians.get_xyz.shape[0]
            if num_gaussians_before != num_gaussians_after and self.active_gaussians_mask is not None:
                # Mask is now invalid, set to None until recomputed
                self.active_gaussians_mask = None
            
            # CL-Splats: Freeze optimizer momentum for inactive Gaussians
            # This prevents any accumulated momentum from affecting frozen Gaussians
            if self.active_gaussians_mask is not None and self.timestep > 0:
                if self.active_gaussians_mask.shape[0] == self.gaussians.get_xyz.shape[0]:
                    self.gaussians.freeze_inactive_gaussians(self.active_gaussians_mask)
            
            # CL-Splats: Set visible mask for SparseGaussianAdam before optimizer step
            # Only set mask if it's valid (matches current Gaussian count)
            if self.active_gaussians_mask is not None and SPARSE_ADAM_AVAILABLE:
                if self.active_gaussians_mask.shape[0] == self.gaussians.get_xyz.shape[0]:
                    if hasattr(self.gaussians.optimizer, 'set_visible_mask'):
                        self.gaussians.optimizer.set_visible_mask(self.active_gaussians_mask)
            
            # Optimizer step
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            
            # Reset visible mask after step
            if SPARSE_ADAM_AVAILABLE and hasattr(self.gaussians.optimizer, 'set_visible_mask'):
                self.gaussians.optimizer.set_visible_mask(None)
        
        return metrics

    def _apply_sphere_pruning(self) -> None:
        """
        Apply sphere-based pruning to active Gaussians.
        
        Per CL-Splats paper: every 15 iterations, check if active Gaussians
        have moved outside their bounding spheres. Only prune those that have
        escaped - inactive Gaussians are never touched.
        """
        # Skip if no active mask or mask is invalid
        if self.active_gaussians_mask is None:
            return
        
        # Validate mask size matches current Gaussian count
        num_gaussians = self.gaussians.get_xyz.shape[0]
        if self.active_gaussians_mask.shape[0] != num_gaussians:
            # Mask is stale, skip this round
            return
        
        if not hasattr(self, '_spheres_fitted') or not self._spheres_fitted:
            # Fit spheres to active Gaussians on first call
            active_xyz = self.gaussians.get_xyz[self.active_gaussians_mask].detach()
            if active_xyz.shape[0] > 0:
                self.pruner.fit(active_xyz)
                self._spheres_fitted = True
                msg = f"Fitted {self.pruner.get_num_spheres()} spheres to {active_xyz.shape[0]} active Gaussians"
                self._emit_progress_message(msg)
            return
        
        # Check which active Gaussians have moved outside spheres
        xyz = self.gaussians.get_xyz.detach()
        
        # Only check active Gaussians
        active_indices = torch.where(self.active_gaussians_mask)[0]
        if len(active_indices) == 0:
            return
        
        active_xyz = xyz[active_indices]
        should_prune_active = self.pruner.should_prune(active_xyz)
        
        if should_prune_active.any():
            # Create full prune mask (only active Gaussians that escaped spheres)
            prune_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=xyz.device)
            prune_mask[active_indices[should_prune_active]] = True
            
            num_to_prune = prune_mask.sum().item()
            msg = f"Sphere pruning: removing {num_to_prune} Gaussians that escaped bounding spheres"
            self._emit_progress_message(msg)
            
            self.gaussians.prune_points(prune_mask)
            
            # Invalidate active mask after pruning (will be recomputed later)
            self.active_gaussians_mask = None

    def train(self) -> Dict[str, Any]:
        """Execute training for the current timestep."""
        logger.info(f"Starting training for timestep {self.timestep}")
        
        if self.gaussians.get_xyz.shape[0] == 0:
            logger.error("No Gaussians initialized! Call initialize_from_point_cloud first.")
            return {"avg_loss": 0.0, "timestep": self.timestep}
        
        # Setup training if not done
        # Training loop config
        train_cfg = self.cfg.get("train", {})
        base_iterations = train_cfg.get("iterations", 30000)
        incremental_iterations = train_cfg.get("incremental_iterations", 15000)
        log_interval = train_cfg.get("log_interval", 100)
        
        # Determine iterations based on timestep
        if self.timestep > 0:
            iterations = incremental_iterations
            logger.info(f"Incremental training: using {iterations} iterations (timestep > 0)")
        else:
            iterations = base_iterations
            logger.info(f"Full training: using {iterations} iterations (timestep 0)")
            
        # Always set up (reset) training/optimizer for new timestep to clear momentum
        # This prevents momentum leak from previous timesteps affecting inactive Gaussians
        self.setup_training(iterations=iterations)
        
        epoch_losses = []
        recent_metrics: List[Dict[str, float]] = []
        
        # Get camera-image pairs
        if not self.current_cameras or not self.current_images:
            logger.error("No cameras or images loaded!")
            return {"avg_loss": 0.0, "timestep": self.timestep}
        
        num_views = min(len(self.current_cameras), len(self.current_images))
        
        # Log Local Optimization status
        if self.active_gaussians_mask is not None:
            num_active = self.active_gaussians_mask.sum().item()
            logger.info(f"Local Optimization enabled: {num_active} active Gaussians")
        else:
            logger.info("Local Optimization disabled (first timestep or no changes)")
        
        # Interval for recomputing active mask after densification
        mask_recompute_interval = train_cfg.get("mask_recompute_interval", 500)
        
        # Use tqdm progress bar (update every log_interval iterations)
        pbar = tqdm(range(iterations), desc=f"[T{self.timestep}] Training", 
                    dynamic_ncols=True, leave=True, miniters=log_interval)
        self._training_pbar = pbar  # Store reference for tqdm.write in other methods
        
        for iteration in pbar:
            # Random view selection
            idx = random.randint(0, num_views - 1)
            camera = self.current_cameras[idx]
            gt_image = self.current_images[idx]
            
            metrics = self._train_step(iteration, camera, gt_image)
            epoch_losses.append(metrics.get("loss", 0.0))
            numeric_metrics: Dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_metrics[str(key)] = float(value)
            recent_metrics.append(numeric_metrics)
            if len(recent_metrics) > max(1, int(log_interval)):
                recent_metrics.pop(0)
            
            # NOTE: We no longer recompute active_gaussians_mask periodically.
            # The mask is computed once at the start and then synchronized
            # through densify_and_prune. Recomputing would cause points that
            # moved during optimization to be incorrectly re-classified.
            
            # Update progress bar postfix every log_interval iterations
            if iteration % log_interval == 0:
                avg_loss = sum(epoch_losses[-log_interval:]) / min(log_interval, len(epoch_losses))
                num_gaussians = self.gaussians.get_xyz.shape[0]

                def _avg_recent(metric_key: str) -> Optional[float]:
                    vals = [m[metric_key] for m in recent_metrics if metric_key in m]
                    if len(vals) == 0:
                        return None
                    return float(sum(vals) / len(vals))

                avg_loss_img = _avg_recent("loss_task")
                avg_loss_attr = _avg_recent("loss_global_attr")
                
                # Build postfix dict for progress bar
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "gs": num_gaussians
                }
                has_explicit_loss_breakdown = (
                    avg_loss_img is not None
                    or avg_loss_attr is not None
                )
                if has_explicit_loss_breakdown:
                    postfix["img"] = f"{float(avg_loss_img or 0.0):.4f}"
                    postfix["attr"] = f"{float(avg_loss_attr or 0.0):.4f}"
                elif avg_loss_img is not None:
                    postfix["img"] = f"{avg_loss_img:.4f}"
                if self.active_gaussians_mask is not None:
                    postfix["active"] = self.active_gaussians_mask.sum().item()
                
                pbar.set_postfix(postfix)
                
                # Log iteration info to file only (not console)
                active_info = ""
                if self.active_gaussians_mask is not None:
                    active_info = f", active={self.active_gaussians_mask.sum().item()}"
                split_terms = ""
                if has_explicit_loss_breakdown:
                    split_terms += f", loss_img={float(avg_loss_img or 0.0):.6f}"
                    split_terms += f", loss_attr={float(avg_loss_attr or 0.0):.6f}"
                elif avg_loss_img is not None:
                    split_terms += f", loss_img={avg_loss_img:.6f}"
                self._log_to_file_only(
                    f"[T{self.timestep}] Iter {iteration}/{iterations}: "
                    f"loss={avg_loss:.6f}, gaussians={num_gaussians}{active_info}{split_terms}"
                )
                
                # Log to wandb
                if wandb.run is not None:
                    log_dict = {
                        "timestep": self.timestep,
                        "iteration": iteration,
                        "loss": avg_loss,
                        "num_gaussians": num_gaussians,
                    }
                    if avg_loss_img is not None:
                        log_dict["loss_img"] = avg_loss_img
                    if avg_loss_attr is not None:
                        log_dict["loss_attr"] = avg_loss_attr
                    if self.active_gaussians_mask is not None:
                        log_dict["num_active_gaussians"] = self.active_gaussians_mask.sum().item()
                    wandb.log(log_dict)
            
            # Visualization - save comparison images periodically (including t0)
            vis_interval = self.cfg.get("train", {}).get("vis_interval", 1000)
            if vis_interval > 0 and iteration % vis_interval == 0 and iteration > 0:
                self._save_visualization(iteration)
            
            # SH degree increase
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()
        
        # Record stats
        self._training_pbar = None  # Clear reference after training loop
        
        stats = {
            "timestep": self.timestep,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "avg_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
            "num_gaussians": self.gaussians.get_xyz.shape[0],
            "local_optimization_enabled": self.active_gaussians_mask is not None,
        }
        
        if self.active_gaussians_mask is not None:
            stats["num_active_gaussians"] = self.active_gaussians_mask.sum().item()
        
        self.history["losses"].append(stats["avg_loss"])
        self.history["num_gaussians"].append(stats["num_gaussians"])
        
        # GIF generation is opt-in; keep visualization PNG frames by default.
        train_cfg = self.cfg.get("train", {})
        if train_cfg.get("vis_generate_gif", False) and self.timestep > 0:
            self.generate_training_gif()
        
        logger.info(f"Completed training for timestep {self.timestep}")
        return stats

    def log_history(self) -> None:
        """Log training history to wandb."""
        if wandb.run is not None:
            wandb.log({
                "history/losses": self.history["losses"],
                "history/num_gaussians": self.history["num_gaussians"],
            })

    def _postprocess_visualization_render(
        self,
        camera,
        render_result: Dict[str, torch.Tensor],
        rendered: torch.Tensor,
        gt_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hook for subclasses to post-process visualization renders."""
        del camera, render_result, gt_image
        return rendered

    def _get_visualization_render_kwargs(self, camera) -> Dict[str, Any]:
        """Hook for subclasses to customize visualization-time render arguments."""
        del camera
        return {}

    def _default_visualization_view_indices(self, num_views: int) -> List[int]:
        """Return the standard first/middle/last visualization indices."""
        if num_views <= 0:
            return []
        return [0, num_views // 2, num_views - 1]

    def _load_t0_visualization_view_indices_from_disk(self) -> Optional[List[int]]:
        """Load the exact t0 visualization view indices already written to disk."""
        output_dir = self.cfg.get("output_dir", "outputs")
        t0_vis_dir = os.path.join(output_dir, "visualizations", "t0")
        if not os.path.isdir(t0_vis_dir):
            return None

        pattern = re.compile(r"^iter(\d+)_view(\d+)_compare\.png$")
        views_by_iter: Dict[int, set] = {}
        for filename in os.listdir(t0_vis_dir):
            match = pattern.match(filename)
            if match is None:
                continue
            iteration = int(match.group(1))
            view_idx = int(match.group(2))
            views_by_iter.setdefault(iteration, set()).add(view_idx)

        if not views_by_iter:
            return None

        latest_iteration = max(views_by_iter)
        view_indices = sorted(views_by_iter[latest_iteration])
        if view_indices:
            logger.info(
                "Using t0 visualization view indices from disk "
                f"(iter {latest_iteration}): {view_indices}"
            )
            return view_indices
        return None

    def _save_visualization(self, iteration: int) -> None:
        """
        Save visualization snapshots for current timestep (including t0).
        
        Saves for first/middle/last views:
        - Comparison image: Rendered | Change Mask | GT (side-by-side)
        - Active-only image: Render using current active Gaussian mask
        
        Saves for special view (if configured):
        - Single rendered image (for forgetting check)
        """
        if not RENDERER_AVAILABLE or self.current_cameras is None:
            return
        
        output_dir = self.cfg.get("output_dir", "outputs")
        vis_dir = os.path.join(output_dir, "visualizations", f"t{self.timestep}")
        os.makedirs(vis_dir, exist_ok=True)
        
        render_cameras = self.current_cameras
        render_images = self.current_images
        using_t0_views = self.timestep > 0 and self.t0_cameras is not None
        if using_t0_views:
            render_cameras = self.t0_cameras
            render_images = None
            logger.info("Using t0 camera views for t>0 visualization/test renders")
        render_change_masks = None if using_t0_views else self.change_masks

        num_views = len(render_cameras)
        if using_t0_views:
            disk_view_indices = self._load_t0_visualization_view_indices_from_disk()
            if disk_view_indices is not None:
                self.t0_visualization_view_indices = disk_view_indices

            if self.t0_visualization_view_indices is not None:
                view_indices = self.t0_visualization_view_indices
            else:
                view_indices = self._default_visualization_view_indices(num_views)
        else:
            view_indices = self._default_visualization_view_indices(num_views)
        valid_view_indices = [idx for idx in view_indices if idx < num_views]
        skipped_view_indices = [idx for idx in view_indices if idx >= num_views]
        if skipped_view_indices:
            logger.warning(
                "Skipping fixed t0 visualization view indices that are not available "
                f"in loaded render cameras: {skipped_view_indices} (num_views={num_views})"
            )
        view_indices = valid_view_indices
        
        # Get special view index from config
        train_cfg = self.cfg.get("train", {})
        special_view_idx = train_cfg.get("vis_special_view_idx", -1)
        
        bg_color = torch.tensor(
            [1, 1, 1] if self.cfg.get("white_background", False) else [0, 0, 0],
            dtype=torch.float32, device=self.device
        )
        
        from clsplats.rendering import render
        
        # Save standard views (first, middle, last) as comparison images
        for idx in view_indices:
            if idx >= num_views:
                continue
            
            camera = render_cameras[idx]
            gt_image = render_images[idx] if render_images else None
            
            with torch.no_grad():
                render_kwargs = self._get_visualization_render_kwargs(camera)
                result = render(
                    camera,
                    self.gaussians,
                    bg_color,
                    **render_kwargs,
                )
                rendered = result["render"]
                rendered = self._postprocess_visualization_render(
                    camera=camera,
                    render_result=result,
                    rendered=rendered,
                    gt_image=gt_image,
                )
            
            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
            rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
            h, w = rendered_np.shape[:2]
            
            # Prepare change mask overlay
            mask_overlay_np = None
            if render_change_masks is not None and idx < len(render_change_masks):
                mask = render_change_masks[idx]
                mask_np = mask.cpu().numpy().astype(np.float32)
                
                if mask_np.shape != (h, w):
                    mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
                    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                
                # Create red overlay for changed regions
                mask_overlay_np = rendered_np.copy().astype(np.float32)
                mask_overlay_np[mask_np > 0.5, 0] = np.minimum(mask_overlay_np[mask_np > 0.5, 0] + 100, 255)
                mask_overlay_np = mask_overlay_np.astype(np.uint8)
            
            # Prepare GT
            gt_np = None
            if gt_image is not None:
                gt_np = gt_image.permute(1, 2, 0).cpu().numpy()
                gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)
            
            # Create comparison: Rendered | Mask Overlay | GT
            gap = 5
            num_panels = 1
            if mask_overlay_np is not None:
                num_panels += 1
            if gt_np is not None:
                num_panels += 1
            
            total_width = w * num_panels + gap * (num_panels - 1)
            comparison = np.zeros((h, total_width, 3), dtype=np.uint8)
            
            x_offset = 0
            comparison[:, x_offset:x_offset+w] = rendered_np
            x_offset += w + gap
            
            if mask_overlay_np is not None:
                comparison[:, x_offset:x_offset+w] = mask_overlay_np
                x_offset += w + gap
            
            if gt_np is not None:
                comparison[:, x_offset:x_offset+w] = gt_np
            
            comp_path = os.path.join(vis_dir, f"iter{iteration:05d}_view{idx}_compare.png")
            PILImage.fromarray(comparison).save(comp_path)
        
        # Save special view from the fixed visualization camera set
        if special_view_idx >= 0 and special_view_idx < num_views:
            camera = render_cameras[special_view_idx]
            with torch.no_grad():
                render_kwargs = self._get_visualization_render_kwargs(camera)
                result = render(
                    camera,
                    self.gaussians,
                    bg_color,
                    **render_kwargs,
                )
                rendered = result["render"]
                rendered = self._postprocess_visualization_render(
                    camera=camera,
                    render_result=result,
                    rendered=rendered,
                    gt_image=None,
                )
            
            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
            rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
            
            special_path = os.path.join(vis_dir, f"iter{iteration:05d}_special_view{special_view_idx}.png")
            PILImage.fromarray(rendered_np).save(special_path)
        
        # Save t0 view for forgetting check (render current model from t0 camera)
        t0_image_path = train_cfg.get("vis_t0_image_path", None)
        if t0_image_path and self.t0_cameras is not None:
            # Find camera by image name
            t0_image_name = os.path.basename(t0_image_path)
            t0_camera = None
            for cam in self.t0_cameras:
                if hasattr(cam, 'image_name') and cam.image_name == t0_image_name:
                    t0_camera = cam
                    break
            
            if t0_camera is not None:
                with torch.no_grad():
                    render_kwargs = self._get_visualization_render_kwargs(t0_camera)
                    result = render(
                        t0_camera,
                        self.gaussians,
                        bg_color,
                        **render_kwargs,
                    )
                    rendered = result["render"]
                    rendered = self._postprocess_visualization_render(
                        camera=t0_camera,
                        render_result=result,
                        rendered=rendered,
                        gt_image=None,
                    )
                
                rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
                rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
                
                t0_path = os.path.join(vis_dir, f"iter{iteration:05d}_t0_{t0_image_name}")
                PILImage.fromarray(rendered_np).save(t0_path)
        
        # Save one active-only Gaussian visualization using the fixed visualization view
        if self.active_gaussians_mask is not None:
            num_gaussians = self.gaussians.get_xyz.shape[0]
            if self.active_gaussians_mask.shape[0] == num_gaussians:
                active_view_idx = num_views // 2 if num_views > 0 else 0
                active_camera = render_cameras[active_view_idx]
                with torch.no_grad():
                    active_result = render(
                        active_camera,
                        self.gaussians,
                        bg_color,
                        active_gaussian_mask=self.active_gaussians_mask,
                    )
                    active_rendered = active_result["render"]
                
                active_np = active_rendered.permute(1, 2, 0).cpu().numpy()
                active_np = np.clip(active_np * 255, 0, 255).astype(np.uint8)
                active_path = os.path.join(
                    vis_dir, f"iter{iteration:05d}_active_gaussians_view{active_view_idx}.png"
                )
                PILImage.fromarray(active_np).save(active_path)
            else:
                logger.warning(
                    "Skip active Gaussian visualization: mask size mismatch "
                    f"({self.active_gaussians_mask.shape[0]} vs {num_gaussians})"
                )
        
        logger.info(f"Saved visualizations to {vis_dir}")
    
    def generate_training_gif(self) -> None:
        """
        Keep saved visualization PNG frames.

        GIF generation used to remove the per-iteration PNG snapshots after
        writing training_*.gif files. This project keeps those PNGs instead.
        """
        logger.info("Skipping training GIF generation; keeping visualization PNG frames.")
        return

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state to checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        checkpoint = {
            "timestep": self.timestep,
            "history": self.history,
            "config": omegaconf.OmegaConf.to_container(self.cfg),
        }
        
        # Save Gaussians separately as PLY
        ply_path = path.replace(".pt", ".ply")
        self.gaussians.save_ply(ply_path)
        checkpoint["ply_path"] = ply_path
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load trainer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.timestep = checkpoint["timestep"]
        self.history = checkpoint["history"]
        
        # Load Gaussians from PLY
        if "ply_path" in checkpoint and os.path.exists(checkpoint["ply_path"]):
            self.gaussians.load_ply(checkpoint["ply_path"])
        
        logger.info(f"Loaded checkpoint from {path}")

