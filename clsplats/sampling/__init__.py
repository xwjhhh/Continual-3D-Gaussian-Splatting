"""
Sampling module for CL-Splats.

This module provides sampling strategies for generating new Gaussian points
in changed regions of the scene.
"""

from clsplats.sampling.base_sampler import BaseSampler
from clsplats.sampling.gaussian_sampler import GaussianSampler

__all__ = ["BaseSampler", "GaussianSampler"]
