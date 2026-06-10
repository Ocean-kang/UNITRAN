"""Text atom construction public facade."""

import warnings

warnings.warn(
    "src.text_atoms is a compatibility facade. Import from src.features.text_atoms instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.features.text_atoms import *  # noqa: F401,F403
