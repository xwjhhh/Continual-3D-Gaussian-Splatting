"""Public training methods for the open-source release."""

import importlib

from clsplats.models.cl_splats import CLSplatsTrainer, TrainingArgs
from clsplats.models.irc_gs import IRCGSTrainer
from clsplats.models.pure_3dgs import Pure3DGSTrainer
from clsplats.models.model_factory import create_trainer, get_available_models

FourDGSTrainer = importlib.import_module("clsplats.models.4dgs").FourDGSTrainer

__all__ = [
    "CLSplatsTrainer",
    "IRCGSTrainer",
    "Pure3DGSTrainer",
    "FourDGSTrainer",
    "TrainingArgs",
    "create_trainer",
    "get_available_models",
]
