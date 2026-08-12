"""
Sampling module for CL-Splats.

This module provides sampling strategies for generating new Gaussian points
in changed regions of the scene.
"""

from ircgs.sampling.base_sampler import BaseSampler
from ircgs.sampling.gaussian_sampler import GaussianSampler

__all__ = ["BaseSampler", "GaussianSampler"]
