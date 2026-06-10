"""
Deprecated compatibility shim.

The UNITRAN mapping implementation belongs to the training package and has
been moved to:

    src.training.unitran_mapping

This module is kept only so older imports keep working. New code should import
from ``src.training.unitran_mapping`` or the public facade ``src.mapping``.
"""

import warnings

warnings.warn(
    "src.visualization.unitran_mapping has moved to src.training.unitran_mapping.",
    DeprecationWarning,
    stacklevel=2,
)

from src.training.unitran_mapping import *  # noqa: F401,F403
