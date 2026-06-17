"""
Training method factory for the open-source release.

Only stable public methods are registered here:
  - cl-splats
  - pure-3dgs
  - ours
  - 4dgs
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List

import omegaconf
from loguru import logger

from clsplats.models.cl_splats import CLSplatsTrainer
from clsplats.representation.model_factory import (
    get_available_models as get_available_representations,
)


TrainerBuilder = Callable[[omegaconf.DictConfig], CLSplatsTrainer]

_TRAINER_BUILDERS: Dict[str, TrainerBuilder] = {}


def _cfg_get(cfg: omegaconf.DictConfig, key: str, default):
    if cfg is None:
        return default
    value = cfg.get(key, default)
    return default if value is None else value


def register_model(name: str):
    """Register a trainer builder under a method name."""

    def _decorator(builder: TrainerBuilder):
        _TRAINER_BUILDERS[name] = builder
        return builder

    return _decorator


def get_available_models() -> List[str]:
    """Return all registered training method names."""
    return sorted(_TRAINER_BUILDERS.keys())


@register_model("cl-splats")
@register_model("clsplats")
def build_cl_splats(cfg: omegaconf.DictConfig) -> CLSplatsTrainer:
    return CLSplatsTrainer(cfg)


@register_model("pure-3dgs")
@register_model("pure_3dgs")
@register_model("3dgs")
def build_pure_3dgs(cfg: omegaconf.DictConfig) -> CLSplatsTrainer:
    from clsplats.models.pure_3dgs import Pure3DGSTrainer

    return Pure3DGSTrainer(cfg)


@register_model("ours")
@register_model("scaffold-gs")
@register_model("scaffold_gs")
@register_model("sfgs")
def build_ours(cfg: omegaconf.DictConfig) -> CLSplatsTrainer:
    from clsplats.models.ours import EveryTimeRebuildTrainer

    return EveryTimeRebuildTrainer(cfg)


@register_model("4dgs")
@register_model("4DGS")
@register_model("four-d-gs")
@register_model("four_d_gs")
def build_4dgs(cfg: omegaconf.DictConfig) -> CLSplatsTrainer:
    module = importlib.import_module("clsplats.models.4dgs")
    return module.FourDGSTrainer(cfg)


def create_trainer(cfg: omegaconf.DictConfig) -> CLSplatsTrainer:
    model_cfg = cfg.get("model", {})
    model_name = _cfg_get(model_cfg, "name", "cl-splats")
    builder = _TRAINER_BUILDERS.get(model_name)
    if builder is None and model_name in get_available_representations():
        logger.warning(
            "Deprecated config detected: "
            f"`model.name={model_name}` is a representation key. "
            f"Use `model.name=cl-splats` and `model.representation={model_name}`."
        )
        builder = _TRAINER_BUILDERS.get("cl-splats")

    if builder is None:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown training method '{model_name}'. Available methods: {available}"
        )
    return builder(cfg)
