"""
Pure sequential 3DGS baseline trainer.

This trainer keeps the timestep-by-timestep training protocol, but disables
all CL-Splats-specific mechanisms:
  - no change detection
  - no 2D->3D lifting
  - no active mask / local optimization
  - no sphere pruning
  - no mask-aware densification / pruning
  - no freezing of inactive Gaussians
"""

from __future__ import annotations

from typing import Dict

from loguru import logger
import torch

from clsplats.models.cl_splats import CLSplatsTrainer, RENDERER_AVAILABLE
from clsplats.rendering import render
from clsplats.utils.loss_utils import combined_loss


class Pure3DGSTrainer(CLSplatsTrainer):
    """Naive sequential 3DGS baseline on multi-timestep data."""

    def _load_timestep_data(self, dataset, timestep: int) -> None:
        """Load data only; disable all CL-specific preprocessing."""
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

        logger.info(
            f"Pure 3DGS mode: loaded {len(self.current_cameras)} cameras "
            f"for timestep {timestep} (all Gaussians trainable)."
        )

        # Hard-disable CL-specific states to prevent accidental carry-over.
        self.change_masks = None
        self.current_depths = None
        self.active_gaussians_mask = None

        # Keep t0 cameras for optional visualization compatibility.
        if timestep == 0:
            self.t0_cameras = self.current_cameras

    def _train_step(self, iteration: int, camera, gt_image) -> Dict[str, float]:
        """Execute one vanilla 3DGS training step on full Gaussian set."""
        if not RENDERER_AVAILABLE:
            return {"loss": 0.0}

        render_result = render(
            camera,
            self.gaussians,
            self.bg_color,
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

        train_cfg = self.cfg.get("train", {})
        lambda_dssim = train_cfg.get("lambda_dssim", 0.2)
        loss = combined_loss(rendered_image, gt_image, lambda_dssim)
        metrics = {"loss": float(loss.item())}

        loss.backward()

        with torch.no_grad():
            self.gaussians.update_learning_rate(iteration)

            if iteration < self.training_args.densify_until_iter:
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_points,
                    visibility_filter,
                )

                if (
                    iteration > self.training_args.densify_from_iter
                    and iteration % self.training_args.densification_interval == 0
                ):
                    # Disable large-Gaussian pruning entirely.
                    size_threshold = None
                    self.gaussians.densify_and_prune(
                        self.training_args.densify_grad_threshold,
                        0.005,
                        self.scene_extent,
                        size_threshold,
                        radii,
                        active_mask=None,
                    )

                # Pure 3DGS behavior: reset opacity globally at every interval.
                if iteration % self.training_args.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()

            if hasattr(self.gaussians.optimizer, "set_visible_mask"):
                self.gaussians.optimizer.set_visible_mask(None)

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            if hasattr(self.gaussians.optimizer, "set_visible_mask"):
                self.gaussians.optimizer.set_visible_mask(None)

        return metrics
