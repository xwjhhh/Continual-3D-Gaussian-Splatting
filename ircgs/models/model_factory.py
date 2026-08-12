"""
Training method factory for the open-source release.

Only public release methods are registered here:
  - irc-gs
  - cl-splats
  - scaffold-gs
  - 3dgs
  - 4dgs
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List

import omegaconf
from loguru import logger

from ircgs.models.official_cl_splats import OfficialCLSplatsTrainer
from ircgs.representation.model_factory import (
    get_available_models as get_available_representations,
)


TrainerBuilder = Callable[[omegaconf.DictConfig], Any]

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
def build_cl_splats(cfg: omegaconf.DictConfig) -> OfficialCLSplatsTrainer:
    """Use the public sibling implementation for the canonical method name."""
    return OfficialCLSplatsTrainer(cfg)

@register_model("3dgs")
def build_pure_3dgs(cfg: omegaconf.DictConfig) -> Any:
    from ircgs.models.pure_3dgs import Pure3DGSTrainer

    return Pure3DGSTrainer(cfg)


@register_model("irc-gs")
def build_irc_gs(cfg: omegaconf.DictConfig) -> Any:
    from ircgs.models.irc_gs import EveryTimeRebuildTrainer

    with omegaconf.open_dict(cfg):
        cfg.model.rebuild_every_timestep = False
    return EveryTimeRebuildTrainer(cfg)


@register_model("scaffold-gs")
def build_scaffold_gs(cfg: omegaconf.DictConfig) -> Any:
    from ircgs.models.irc_gs import EveryTimeRebuildTrainer

    with omegaconf.open_dict(cfg):
        cfg.model.rebuild_every_timestep = True
    return EveryTimeRebuildTrainer(cfg)


@register_model("4dgs")
def build_4dgs(cfg: omegaconf.DictConfig) -> Any:
    module = importlib.import_module("ircgs.models.4dgs")
    return module.FourDGSTrainer(cfg)


def create_trainer(cfg: omegaconf.DictConfig) -> Any:
    model_cfg = cfg.get("model", {})
    model_name = _cfg_get(model_cfg, "name", "irc-gs")
    builder = _TRAINER_BUILDERS.get(model_name)
    if builder is None and model_name in get_available_representations():
        logger.warning(
            "Deprecated config detected: "
            f"`model.name={model_name}` is a representation key. "
            f"Use `model.name=irc-gs` and `model.representation={model_name}`."
        )
        builder = _TRAINER_BUILDERS.get("irc-gs")

    if builder is None:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown training method '{model_name}'. Available methods: {available}"
        )
    return builder(cfg)
