"""Distribution metric public facade."""

import warnings

warnings.warn(
    "src.metrics is a compatibility facade. Import from src.evaluation.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.evaluation.metrics import *  # noqa: F401,F403
