"""
Sphere-based pruning for CL-Splats.

This module implements the sphere pruning strategy described in the CL-Splats paper.
It uses HDBSCAN clustering to identify change regions and fits bounding spheres
around each cluster to constrain optimization.
"""

from typing import List, Tuple, Optional
import torch
import numpy as np
from sklearn.cluster import HDBSCAN
import omegaconf

from clsplats.pruning.base_pruner import BasePruner


class Sphere:
    """
    Represents a 3D sphere for spatial pruning.
    
    Attributes:
        center: Sphere center coordinates (3D)
        radius: Sphere radius
    """
    
    def __init__(self, center: np.ndarray, radius: float):
        """
        Initialize a sphere.
        
        Args:
            center: Numpy array of shape (3,) for sphere center
            radius: Sphere radius
        """
        self.center = center
        self.radius = radius
    
    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are inside the sphere.
        
        Args:
            points: Numpy array of shape (N, 3)
            
        Returns:
            Boolean array of shape (N,) indicating containment
        """
        distances = np.linalg.norm(points - self.center, axis=1)
        return distances <= self.radius
    
    def __repr__(self) -> str:
        return f"Sphere(center={self.center}, radius={self.radius:.4f})"


class SpherePruner(BasePruner):
    """
    Sphere-based pruner for constraining Gaussian optimization.
    
    This pruner uses HDBSCAN clustering to identify distinct change regions,
    then fits a bounding sphere around each cluster. Gaussians that move
    outside all spheres during optimization are pruned.
    
    Configuration:
        min_cluster_size: Minimum number of points to form a cluster (default: 1000)
        quantile: Percentile for radius calculation (default: 0.98)
        radius_multiplier: Safety factor for sphere radius (default: 1.1)
    """
    
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Initialize the sphere pruner.
        
        Args:
            cfg: Configuration dictionary with pruning parameters
        """
        self.cfg = cfg
        self.min_cluster_size = cfg.get('min_cluster_size', 1000)
        self.quantile = cfg.get('quantile', 0.98)
        self.radius_multiplier = cfg.get('radius_multiplier', 1.1)
        
        self.spheres: List[Sphere] = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def fit(self, gaussians: torch.Tensor) -> None:
        """
        Fit bounding spheres to the given Gaussians.
        
        This method:
        1. Clusters the Gaussians using HDBSCAN
        2. Computes the center and radius for each cluster
        3. Creates bounding spheres with a safety margin
        
        Args:
            gaussians: Tensor of shape (N, 3) containing Gaussian centers
        """
        if gaussians.shape[0] == 0:
            self.spheres = []
            return
        
        # Convert to numpy for sklearn
        points_np = gaussians.cpu().numpy()
        
        # Adjust min_cluster_size if we have too few points
        effective_min_cluster_size = min(self.min_cluster_size, max(2, len(points_np) // 2))
        
        # Cluster the points
        clusterer = HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(points_np)
        
        # Fit a sphere for each cluster
        self.spheres = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get points in this cluster
            cluster_mask = cluster_labels == label
            cluster_points = points_np[cluster_mask]
            
            if len(cluster_points) < self.min_cluster_size:
                continue
            
            # Compute sphere parameters
            center = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - center, axis=1)
            radius = np.quantile(distances, self.quantile) * self.radius_multiplier
            
            sphere = Sphere(center, radius)
            self.spheres.append(sphere)
        
        if len(self.spheres) == 0:
            # Fallback: create a single sphere encompassing all points
            center = np.mean(points_np, axis=0)
            distances = np.linalg.norm(points_np - center, axis=1)
            radius = np.quantile(distances, self.quantile) * self.radius_multiplier
            self.spheres.append(Sphere(center, radius))
    
    def should_prune(self, gaussians: torch.Tensor) -> torch.Tensor:
        """
        Determine which Gaussians should be pruned.
        
        A Gaussian is pruned if its center is outside all bounding spheres.
        
        Args:
            gaussians: Tensor of shape (N, 3) containing Gaussian centers
            
        Returns:
            Boolean tensor of shape (N,) where True indicates pruning
        """
        if len(self.spheres) == 0:
            # No spheres fitted, don't prune anything
            return torch.zeros(gaussians.shape[0], dtype=torch.bool, device=gaussians.device)
        
        inside = self.is_inside(gaussians)
        return ~inside  # Prune if NOT inside
    
    def is_inside(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside any bounding sphere.
        
        Args:
            points: Tensor of shape (N, 3) containing point coordinates
            
        Returns:
            Boolean tensor of shape (N,) where True indicates the point
            is inside at least one sphere
        """
        if len(self.spheres) == 0:
            return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        
        points_np = points.cpu().numpy()
        inside_any = np.zeros(points_np.shape[0], dtype=bool)
        
        for sphere in self.spheres:
            inside_any |= sphere.contains(points_np)
        
        return torch.from_numpy(inside_any).to(points.device)
    
    def get_num_spheres(self) -> int:
        """
        Get the number of fitted spheres.
        
        Returns:
            Number of spheres
        """
        return len(self.spheres)
    
    def get_sphere_info(self) -> List[dict]:
        """
        Get information about all fitted spheres.
        
        Returns:
            List of dictionaries containing sphere parameters
        """
        return [
            {
                'center': sphere.center.tolist(),
                'radius': float(sphere.radius)
            }
            for sphere in self.spheres
        ]
    
    def visualize_spheres(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sphere data for visualization.
        
        Returns:
            Tuple of (centers, radii) as numpy arrays
        """
        if len(self.spheres) == 0:
            return np.array([]), np.array([])
        
        centers = np.array([s.center for s in self.spheres])
        radii = np.array([s.radius for s in self.spheres])
        return centers, radii
