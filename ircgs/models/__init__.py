"""Public training methods for the open-source release."""

import importlib

from ircgs.models.cl_splats import CLSplatsTrainer, TrainingArgs
from ircgs.models.irc_gs import IRCGSTrainer
from ircgs.models.pure_3dgs import Pure3DGSTrainer
from ircgs.models.model_factory import create_trainer, get_available_models

FourDGSTrainer = importlib.import_module("ircgs.models.4dgs").FourDGSTrainer

__all__ = [
    "CLSplatsTrainer",
    "IRCGSTrainer",
    "Pure3DGSTrainer",
    "FourDGSTrainer",
    "TrainingArgs",
    "create_trainer",
    "get_available_models",
]
