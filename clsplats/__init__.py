"""IRC-GS and supported Gaussian-splatting baselines."""

__version__ = "0.1.0"
__all__ = ["IRCGSTrainer", "CLSplatsTrainer"]


def __getattr__(name):
    if name == "IRCGSTrainer":
        from clsplats.models.irc_gs import IRCGSTrainer

        return IRCGSTrainer
    if name == "CLSplatsTrainer":
        from clsplats.models.cl_splats import CLSplatsTrainer

        return CLSplatsTrainer
    raise AttributeError(name)
