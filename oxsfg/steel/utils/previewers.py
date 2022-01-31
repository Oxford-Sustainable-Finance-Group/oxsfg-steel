"""Previewers save RGB images for quick inspection"""
from typing import List

import numpy as np
from PIL import Image


def s2_previewer(data: np.ndarray, savepath: str, bands: List[str], **kwargs) -> bool:

    rgb_indices = tuple(np.argmax(np.array(bands) == b) for b in ["B4", "B3", "B2"])

    im = Image.fromarray(
        ((data[:, :, rgb_indices] / 4000).clip(0, 1) * 255).astype(np.uint8)
    )
    im.save(savepath)

    return True


def basemap_previewer(data: np.ndarray, savepath: str, **kwargs) -> bool:

    im = Image.fromarray(data.astype(np.uint8))
    im.save(savepath)

    return True
