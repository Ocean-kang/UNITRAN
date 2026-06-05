"""Heatmap utilities shared by visualization modules."""

from __future__ import annotations

import numpy as np
from PIL import Image


def simple_colormap(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(x - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - x), 0, 1)
    return np.stack([r, g, b], axis=-1)


def make_attention_overlay(image: Image.Image, attention: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = (simple_colormap(attention) * 255.0).astype('uint8')
    heat_im = Image.fromarray(heat, mode='RGB')
    resample = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
    heat_im = heat_im.resize(image.size, resample=resample)
    return Image.blend(image.convert('RGB'), heat_im, alpha=float(alpha))
