"""
Deprecated compatibility shim.

The UNITRAN mapping implementation belongs to the training package and has
been moved to:

    src.training.unitran_mapping

This module is kept only so older imports keep working. New code should import
from ``src.training.unitran_mapping`` or the public facade ``src.mapping``.
"""

from src.training.unitran_mapping import *  # noqa: F401,F403
