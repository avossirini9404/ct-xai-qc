"""Seeding and determinism helpers.

Every experiment is driven by a single integer seed that is recorded in the
configuration file and written into the output tables, so that any number in the
README can be traced back to the run that produced it.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and, if it is installed, PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # the perturbation and metric modules do not need torch
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_deterministic(warn_only: bool = True) -> None:
    """Ask PyTorch for deterministic kernels where they exist."""
    try:
        import torch
    except ImportError:
        return
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
