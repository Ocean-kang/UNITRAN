"""Evaluation public facade.

Heavy experiment flows live in experiments/run_density_eval.py so users do not need to
choose between multiple evaluation entrypoints.
"""

import warnings

warnings.warn(
    "src.evaluate is a legacy facade for distribution metrics. Use experiments/run_density_eval.py for current evaluation.",
    DeprecationWarning,
    stacklevel=2,
)

from src.evaluation.distribution import *  # noqa: F401,F403
