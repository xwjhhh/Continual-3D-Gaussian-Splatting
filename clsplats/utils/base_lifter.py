"""
Base class for 2D→3D lifting modules.

Lifters are responsible for determining which 3D points correspond to
changed regions detected in 2D images.
"""

import abc
from typing import List

import torch
import omegaconf


class BaseLifter(abc.ABC):
    """
    Abstract base class for 2D to 3D lifting.
    
    Lifters take 2D change masks from multiple views and determine
    which 3D points are in the changed regions.
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Initialize the lifter.
        
        Args:
            cfg: Configuration dictionary containing lifter parameters.
        """
        self.cfg = cfg

    @abc.abstractmethod
    def lift(
        self,
        points_3d: torch.Tensor,
        change_masks_2d: List[torch.Tensor],
        cameras: List,
    ) -> torch.Tensor:
        """
        Lift 2D change information to 3D.
        
        Args:
            points_3d: 3D points to check [N, 3]
            change_masks_2d: List of 2D change masks for each view
            cameras: List of camera objects with projection information
            
        Returns:
            Boolean tensor [N] indicating which points are in changed regions
        """
        pass
