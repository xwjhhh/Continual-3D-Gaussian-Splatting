"""
Base sampler for sampling new Gaussian points in changed regions.

This module implements the sampling strategies described in the CL-Splats paper
for handling new objects that appear in the scene.
"""

import abc
from typing import List, Tuple, Optional

import omegaconf
import torch
import numpy as np

from clsplats.utils.base_lifter import BaseLifter


class BaseSampler(abc.ABC):
    """Base class for sampling new Gaussian points."""

    def __init__(self, cfg: omegaconf.DictConfig, lifter: Optional[BaseLifter] = None):
        """
        Initialize the sampler.

        Args:
            cfg: Configuration dictionary containing sampler parameters.
            lifter: Optional lifter for 2D→3D lifting. If None, a default
                    MajorityVoteLifter will be created.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup lifter
        if lifter is not None:
            self.lifter = lifter
        else:
            # Create default lifter
            from clsplats.utils.majority_vote_lifter import MajorityVoteLifter
            lifter_cfg = omegaconf.OmegaConf.create({
                "vote_threshold": cfg.get("majority_vote_threshold", 2)
            })
            self.lifter = MajorityVoteLifter(lifter_cfg)

    @abc.abstractmethod
    def sample_new_points(
        self,
        existing_gaussians: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample new Gaussian points in the changed regions.

        Args:
            existing_gaussians: Existing Gaussian positions [N, 3]
            change_masks_2d: List of 2D change masks for each view [H, W]
            cameras: List of camera objects with projection information
            num_samples: Number of points to sample

        Returns:
            Tuple of (sampled_points [M, 3], valid_mask [M]) where valid_mask
            indicates which points passed the majority voting criterion.
        """
        pass

    def lift_to_3d(
        self,
        points_3d: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
    ) -> torch.Tensor:
        """
        Use the lifter to determine which 3D points are in changed regions.

        Args:
            points_3d: 3D points to check [N, 3]
            change_masks_2d: List of 2D change masks [H, W]
            cameras: List of camera objects

        Returns:
            Boolean mask [N] indicating which points are in changed regions
        """
        return self.lifter.lift(points_3d, change_masks_2d, cameras)

    def _compute_scene_bounds(
        self, existing_gaussians: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the bounding box of the existing scene.

        Args:
            existing_gaussians: Existing Gaussian positions [N, 3]

        Returns:
            Tuple of (min_bounds [3], max_bounds [3])
        """
        if existing_gaussians.shape[0] == 0:
            # Default bounds if no existing points
            return (
                torch.tensor([-1.0, -1.0, -1.0], device=self.device),
                torch.tensor([1.0, 1.0, 1.0], device=self.device),
            )
        
        min_bounds = existing_gaussians.min(dim=0)[0]
        max_bounds = existing_gaussians.max(dim=0)[0]
        
        # Add some padding
        padding = (max_bounds - min_bounds) * 0.1
        min_bounds -= padding
        max_bounds += padding
        
        return min_bounds, max_bounds
