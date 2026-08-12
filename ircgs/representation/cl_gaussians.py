import abc
import typing

import omegaconf
import torch

import ircgs.representation.gaussian_model as gaussian_model


class CLGaussians(gaussian_model.GaussianModel):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self._static_gaussians = {}

    @abc.abstractmethod
    def prune_gaussians(
        self, pruning_fn: typing.Callable[[dict], torch.Tensor]
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def unify_gaussians(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def split_gaussians(self, active_mask: torch.Tensor) -> None:
        raise NotImplementedError
