"""
This module contains the main classes and functions for the Super Duper Resolution model.
"""

__all__ = [
    "Trainer",
    "SuperDuperResolution",
    "MSELoss",
    "optim"
]

from torch import optim
from torch.nn import MSELoss

from sdr_model.trainer import Trainer
from sdr_model.model import SuperDuperResolution
