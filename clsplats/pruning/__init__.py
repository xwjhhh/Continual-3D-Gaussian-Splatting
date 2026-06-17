"""
Pruning module for CL-Splats.

This module implements sphere-based pruning to constrain Gaussian optimization
within detected change regions.
"""

from clsplats.pruning.sphere_pruner import SpherePruner

__all__ = ['SpherePruner']
