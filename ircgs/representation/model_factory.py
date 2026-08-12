"""
Model factory for CL-Splats.

This module provides a lightweight model registry so training can select
different model implementations through configuration, e.g. `model.name=...`.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import omegaconf

from ircgs.representation.gaussian_model import GaussianModel


ModelBuilder = Callable[[omegaconf.DictConfig], GaussianModel]

_MODEL_BUILDERS: Dict[str, ModelBuilder] = {}


def _cfg_get(cfg: omegaconf.DictConfig, key: str, default):
    if cfg is None:
        return default
    value = cfg.get(key, default)
    return default if value is None else value


def register_model(name: str):
    """Register a model builder under a name."""

    def _decorator(builder: ModelBuilder):
        _MODEL_BUILDERS[name] = builder
        return builder

    return _decorator


def get_available_models() -> List[str]:
    """Return all registered model names."""
    return sorted(_MODEL_BUILDERS.keys())


@register_model("gaussian")
def build_gaussian(cfg: omegaconf.DictConfig) -> GaussianModel:
    """
    Standard Gaussian model.

    Uses cfg.sh_degree and cfg.optimizer_type (if provided).
    """
    sh_degree = _cfg_get(cfg, "sh_degree", 3)
    optimizer_type = _cfg_get(cfg, "optimizer_type", "sparse_adam")
    return GaussianModel(sh_degree, optimizer_type=optimizer_type)


@register_model("gaussian_sparse")
def build_gaussian_sparse(cfg: omegaconf.DictConfig) -> GaussianModel:
    """
    Gaussian model with sparse_adam as default optimizer.
    """
    sh_degree = _cfg_get(cfg, "sh_degree", 3)
    optimizer_type = _cfg_get(cfg, "optimizer_type", "sparse_adam")
    return GaussianModel(sh_degree, optimizer_type=optimizer_type)


@register_model("gaussian_default")
def build_gaussian_default(cfg: omegaconf.DictConfig) -> GaussianModel:
    """
    Gaussian model with default Adam optimizer as default.
    """
    sh_degree = _cfg_get(cfg, "sh_degree", 3)
    optimizer_type = _cfg_get(cfg, "optimizer_type", "default")
    return GaussianModel(sh_degree, optimizer_type=optimizer_type)


def create_model(cfg: omegaconf.DictConfig) -> GaussianModel:
    """
    Create model from configuration.

    Required/optional fields:
      - name: model key in registry (default: "gaussian")
      - sh_degree: passed to selected model (if supported)
      - optimizer_type: passed to selected model (if supported)
    """
    model_name = _cfg_get(cfg, "name", "gaussian")
    builder = _MODEL_BUILDERS.get(model_name)
    if builder is None:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    return builder(cfg)
