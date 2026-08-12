"""
Gaussian Mixture Model based sampler for new points.

Implements Algorithm 2 and Algorithm 3 from the CL-Splats paper for sampling
new Gaussian points in changed regions.
"""

from typing import List, Tuple, Optional

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import omegaconf

from ircgs.sampling.base_sampler import BaseSampler
from ircgs.utils.base_lifter import BaseLifter


class GaussianSampler(BaseSampler):
    """
    Sampler that uses Gaussian Mixture Models to sample new points.
    
    This implements the sampling strategies from the CL-Splats paper:
    - Full-scene sampling when no nearby points exist
    - Region-based sampling around existing changed points
    """

    def __init__(self, cfg: omegaconf.DictConfig, lifter: Optional[BaseLifter] = None):
        """
        Initialize the Gaussian sampler.

        Args:
            cfg: Configuration with parameters:
                - num_clusters: Number of K-Means clusters (default: 10)
                - samples_per_round: Number of samples per iteration (default: 100)
                - max_rounds: Maximum sampling rounds (default: 10)
                - min_points_for_region_sampling: Minimum points to use region sampling (default: 50)
                - majority_vote_threshold: Minimum views for a point to be valid (default: 2)
            lifter: Optional lifter for 2D→3D lifting
        """
        super().__init__(cfg, lifter)
        self.num_clusters = cfg.get("num_clusters", 10)
        self.samples_per_round = cfg.get("samples_per_round", 100)
        self.max_rounds = cfg.get("max_rounds", 10)
        self.min_points_for_region = cfg.get("min_points_for_region_sampling", 50)

    def sample_new_points(
        self,
        existing_gaussians: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample new Gaussian points using the appropriate strategy.

        Implements Algorithm 1 from the paper, which chooses between
        full-scene sampling and region-based sampling.

        Args:
            existing_gaussians: Existing Gaussian positions [N, 3]
            change_masks_2d: List of 2D change masks for each view
            cameras: List of camera objects
            num_samples: Target number of valid samples to generate

        Returns:
            Tuple of (sampled_points [M, 3], valid_mask [M])
        """
        # First, identify which existing Gaussians are in the changed region
        changed_gaussians_mask = self.lift_to_3d(
            existing_gaussians, change_masks_2d, cameras
        )
        changed_gaussians = existing_gaussians[changed_gaussians_mask]

        # Decide sampling strategy based on number of changed points
        if changed_gaussians.shape[0] < self.min_points_for_region:
            # Algorithm 2: Full-scene sampling
            return self._sample_full_scene(
                existing_gaussians, change_masks_2d, cameras, num_samples
            )
        else:
            # Algorithm 3: Region-based sampling
            return self._sample_region(
                changed_gaussians, change_masks_2d, cameras, num_samples
            )

    def _sample_full_scene(
        self,
        existing_gaussians: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points uniformly across the entire scene.

        Implements Algorithm 2 from the paper.

        Args:
            existing_gaussians: Existing Gaussian positions [N, 3]
            change_masks_2d: List of 2D change masks
            cameras: List of camera objects
            num_samples: Target number of valid samples

        Returns:
            Tuple of (sampled_points, valid_mask)
        """
        min_bounds, max_bounds = self._compute_scene_bounds(existing_gaussians)
        
        all_samples = []
        all_valid = []
        
        for round_idx in range(self.max_rounds):
            # Sample uniformly in the scene bounding box
            samples = torch.rand(
                self.samples_per_round, 3, device=self.device
            ) * (max_bounds - min_bounds) + min_bounds
            
            # Apply majority voting to filter valid samples
            valid_mask = self.lift_to_3d(samples, change_masks_2d, cameras)
            
            all_samples.append(samples)
            all_valid.append(valid_mask)
            
            # Check if we have enough valid samples
            total_valid = sum(mask.sum().item() for mask in all_valid)
            if total_valid >= num_samples:
                break
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        all_valid = torch.cat(all_valid, dim=0)
        
        return all_samples, all_valid

    def _sample_region(
        self,
        changed_gaussians: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points around existing changed Gaussians using GMM.

        Implements Algorithm 3 from the paper.

        Args:
            changed_gaussians: Gaussians in the changed region [N, 3]
            change_masks_2d: List of 2D change masks
            cameras: List of camera objects
            num_samples: Target number of valid samples

        Returns:
            Tuple of (sampled_points, valid_mask)
        """
        # Fit K-Means to cluster the changed points
        num_clusters = min(self.num_clusters, changed_gaussians.shape[0])
        
        # Convert to numpy for sklearn
        points_np = changed_gaussians.cpu().numpy()
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(points_np)
        
        # Initialize Gaussian Mixture Model from K-Means clusters
        gmm = self._init_gmm_from_kmeans(points_np, kmeans)
        
        all_samples = []
        all_valid = []
        
        for round_idx in range(self.max_rounds):
            # Sample from the GMM
            samples_np, _ = gmm.sample(self.samples_per_round)
            samples = torch.from_numpy(samples_np).float().to(self.device)
            
            # Apply majority voting via lifter
            valid_mask = self.lift_to_3d(samples, change_masks_2d, cameras)
            
            all_samples.append(samples)
            all_valid.append(valid_mask)
            
            # Check if we have enough valid samples
            total_valid = sum(mask.sum().item() for mask in all_valid)
            if total_valid >= num_samples:
                break
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        all_valid = torch.cat(all_valid, dim=0)
        
        return all_samples, all_valid

    def _init_gmm_from_kmeans(
        self, points: np.ndarray, kmeans: KMeans
    ) -> GaussianMixture:
        """
        Initialize a Gaussian Mixture Model from K-Means clustering results.

        As described in the supplementary material, we use K-Means cluster centers
        as means and compute covariances based on the furthest point in each cluster.

        Args:
            points: Point cloud [N, 3]
            kmeans: Fitted K-Means model

        Returns:
            Initialized GaussianMixture model
        """
        num_clusters = kmeans.n_clusters
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # Compute covariances for each cluster
        covariances = []
        for i in range(num_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) == 0:
                # Fallback for empty clusters
                covariances.append(np.eye(3) * 0.01)
                continue
            
            # Find the furthest point from the center
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            max_dist = distances.max()
            
            # Create diagonal covariance matrix
            # Use max_dist as the standard deviation
            cov = np.eye(3) * (max_dist ** 2)
            covariances.append(cov)
        
        covariances = np.array(covariances)
        
        # Initialize GMM with these parameters
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type='full',
            random_state=42
        )
        
        # Set the parameters directly
        gmm.means_ = centers
        
        # Add regularization to prevent singular matrices
        reg_covar = 1e-6
        regularized_covariances = []
        for cov in covariances:
            # Add small value to diagonal for numerical stability
            reg_cov = cov + np.eye(cov.shape[0]) * reg_covar
            regularized_covariances.append(reg_cov)
        
        gmm.covariances_ = np.array(regularized_covariances)
        gmm.weights_ = np.ones(num_clusters) / num_clusters
        
        # Compute precision matrices (inverse of covariance)
        try:
            gmm.precisions_cholesky_ = np.array([
                np.linalg.cholesky(np.linalg.inv(cov)) for cov in regularized_covariances
            ])
        except np.linalg.LinAlgError:
            # If still singular, use larger regularization
            reg_covar = 1e-4
            regularized_covariances = [cov + np.eye(cov.shape[0]) * reg_covar for cov in covariances]
            gmm.covariances_ = np.array(regularized_covariances)
            gmm.precisions_cholesky_ = np.array([
                np.linalg.cholesky(np.linalg.inv(cov)) for cov in regularized_covariances
            ])
        
        return gmm

    def get_valid_samples(
        self,
        all_samples: torch.Tensor,
        all_valid: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract valid samples from the sampled points.

        Args:
            all_samples: All sampled points [N, 3]
            all_valid: Boolean mask indicating valid points [N]
            num_samples: Optional number of samples to return (returns all if None)

        Returns:
            Valid sampled points [M, 3] where M <= N
        """
        valid_samples = all_samples[all_valid]
        
        if num_samples is not None and valid_samples.shape[0] > num_samples:
            # Randomly select num_samples points
            indices = torch.randperm(valid_samples.shape[0])[:num_samples]
            valid_samples = valid_samples[indices]
        
        return valid_samples
