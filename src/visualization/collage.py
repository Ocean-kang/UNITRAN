"""Small image-collage helper."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image


def save_collage(images: Sequence[Image.Image], path: str | Path, cols: int = 5, background=(255, 255, 255)) -> None:
    if not images:
        raise ValueError('No images to save')
    cols = max(1, int(cols))
    rows = math.ceil(len(images) / cols)
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new('RGB', (cols * w, rows * h), background)
    for i, im in enumerate(images):
        canvas.paste(im.convert('RGB'), ((i % cols) * w, (i // cols) * h))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)
