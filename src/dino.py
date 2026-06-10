"""DINOv2 feature extraction public facade."""

import warnings

warnings.warn(
    "src.dino is a compatibility facade. Import from src.features.dinov2 instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.features.dinov2 import *  # noqa: F401,F403
