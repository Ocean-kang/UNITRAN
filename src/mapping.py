"""UNITRAN / vec2vec-style mapping public facade."""

import warnings

warnings.warn(
    "src.mapping is a compatibility facade. Import from src.training.unitran_mapping instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.training.unitran_mapping import *  # noqa: F401,F403
