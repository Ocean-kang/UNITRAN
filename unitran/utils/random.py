"""Reproducibility helpers."""

from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int, cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
