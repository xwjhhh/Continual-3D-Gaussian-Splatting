"""
Majority Vote Lifter for 2D鈫?D lifting.

This module implements the 2D to 3D lifting strategy from the CL-Splats paper
using majority voting across multiple views.
"""

from typing import List, Optional, Tuple
import torch
import omegaconf

from clsplats.utils.base_lifter import BaseLifter


class MajorityVoteLifter(BaseLifter):
    """
    2D鈫?D lifting using majority voting across multiple views.
    
    A 3D point is considered to be in a changed region if it projects
    into the change mask in at least `threshold` views.
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Initialize the majority vote lifter.
        
        Args:
            cfg: Configuration with parameters:
                - vote_threshold: Minimum views for a point to be valid (default: 2)
        """
        super().__init__(cfg)
        self.threshold = cfg.get("vote_threshold", 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def lift(
        self,
        points_3d: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
        depth_maps: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Lift 2D change masks to 3D by majority voting.
        
        For each 3D point, project it to each view and check if it falls
        within the change mask. A point is considered "changed" if it
        appears in the change mask in at least `threshold` views.
        
        Args:
            points_3d: 3D points to check [N, 3]
            change_masks_2d: List of 2D change masks [H, W] for each view
            cameras: List of camera objects with projection information
            depth_maps: Optional list of depth maps (invdepths [1, H, W]) from renderer

            
        Returns:
            Boolean tensor [N] indicating which points are in changed regions
        """
        if points_3d.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        
        num_points = points_3d.shape[0]
        vote_counts = torch.zeros(num_points, dtype=torch.int32, device=self.device)

        for i, (mask_2d, camera) in enumerate(zip(change_masks_2d, cameras)):
            # Project 3D points to 2D
            projected_2d, point_depths = self._project_points(points_3d, camera)
            
            h, w = mask_2d.shape
            
            # Check valid projections
            valid_x = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < w)
            valid_y = (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < h)
            valid = valid_x & valid_y
            
            # Vectorized lookup for valid points
            valid_idx = valid.nonzero(as_tuple=True)[0]
            if valid_idx.shape[0] > 0:
                x = projected_2d[valid_idx, 0].long()
                y = projected_2d[valid_idx, 1].long()
                
                # Ensure mask is on same device
                mask_device = mask_2d.to(self.device)
                in_mask = mask_device[y, x]
                
                # Occlusion Check (if depth maps available)
                if depth_maps is not None:
                    # depth_maps[i] is [1, H, W] containing invdepths (1/z)
                    inv_depth_map = depth_maps[i].to(self.device).squeeze(0)  # [H, W]
                    
                    # Surface depth at projection
                    surface_inv_depth = inv_depth_map[y, x]
                    surface_depth = 1.0 / (surface_inv_depth + 1e-7)
                    
                    # Point depth
                    pt_depth = point_depths[valid_idx].squeeze()
                    
                    # Check if point is significantly behind surface
                    # If pt_depth > surface_depth + margin, it is occluded
                    # margin can be small, e.g. 0.1
                    margin = 0.2
                    is_occluded = pt_depth > (surface_depth + margin)
                    
                    # Only vote if NOT occluded
                    should_vote = in_mask & (~is_occluded)
                    vote_counts[valid_idx] += should_vote.int()
                else:
                    vote_counts[valid_idx] += in_mask.int()

        return vote_counts >= self.threshold

    def _project_points(self, points_3d: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D image coordinates.
        
        Uses the same projection as 3DGS renderer (full_proj_transform).
        
        Args:
            points_3d: 3D points [N, 3]
            camera: Camera object with projection matrices
            
        Returns:
            Tuple containing:
            - 2D pixel coordinates [N, 2]
            - Point depths (z-values) [N, 1]
        """
        points_3d = points_3d.to(self.device)
        
        # Convert to homogeneous coordinates
        ones = torch.ones(points_3d.shape[0], 1, device=self.device)
        points_homo = torch.cat([points_3d, ones], dim=1)  # [N, 4]
        
        # Get full projection matrix (world_view @ projection)
        full_proj = camera.full_proj_transform.to(self.device)  # [4, 4]
        
        # Apply full projection: p_clip = P * p_world
        points_clip = points_homo @ full_proj  # [N, 4]
        
        # Perspective division - 3DGS puts depth in w component
        w = points_clip[:, 3:4]
        valid_depth = (w > 0.01).squeeze()
        
        # NDC coordinates
        points_ndc = points_clip[:, :2] / (w + 1e-7)  # [N, 2]
        
        # Convert from NDC [-1, 1] to pixel coordinates
        points_2d = torch.zeros_like(points_ndc)
        points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * camera.image_width
        points_2d[:, 1] = (points_ndc[:, 1] + 1.0) * 0.5 * camera.image_height
        
        # Set invalid points (behind camera) to out-of-bounds
        points_2d[~valid_depth] = -1
        
        return points_2d, w

    def set_threshold(self, threshold: int):
        """Update the voting threshold."""
        self.threshold = threshold

