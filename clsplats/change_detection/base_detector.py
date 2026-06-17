import abc

import omegaconf
import torch

from clsplats.utils.custom_types import Image

class BaseDetector(abc.ABC):

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def predict_change_mask(self, rendered_image: Image, observation: Image) -> torch.Tensor:
        """Given a rendered image and an observation, predict a change mask.

        Args:
            rendered_image: The rendered image. [H, W, 3]
            observation: The observation. [H, W, 3]

        Returns:
            A change mask. [H, W]
        """
        pass