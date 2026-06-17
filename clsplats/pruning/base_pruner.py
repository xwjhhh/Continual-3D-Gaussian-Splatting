"""
Base pruner class for CL-Splats.

This module defines the abstract interface for pruning strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch


class BasePruner(ABC):
    """
    Abstract base class for pruning strategies.
    
    Pruners are responsible for identifying and removing Gaussians that
    move outside the designated optimization regions during training.
    """
    
    @abstractmethod
    def fit(self, gaussians: torch.Tensor) -> None:
        """
        Fit the pruning boundaries based on the given Gaussians.
        
        Args:
            gaussians: Tensor of shape (N, 3) containing Gaussian centers
        """
        pass
    
    @abstractmethod
    def should_prune(self, gaussians: torch.Tensor) -> torch.Tensor:
        """
        Determine which Gaussians should be pruned.
        
        Args:
            gaussians: Tensor of shape (N, 3) containing Gaussian centers
            
        Returns:
            Boolean tensor of shape (N,) where True indicates the Gaussian
            should be pruned
        """
        pass
    
    @abstractmethod
    def is_inside(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the pruning boundaries.
        
        Args:
            points: Tensor of shape (N, 3) containing point coordinates
            
        Returns:
            Boolean tensor of shape (N,) where True indicates the point
            is inside the boundaries
        """
        pass
