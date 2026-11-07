from __future__ import annotations

import cv2
import numpy as np

from ..quality.assess import text_and_background_masks


def conservative_sharpen(image: np.ndarray, amount: float = 0.28, radius: float = 1.0) -> np.ndarray:
    """Recover edge contrast without synthesising new strokes or strong halos."""
    source = image.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(source, (0, 0), max(0.3, radius))
    detail = np.clip(source - blurred, -0.065, 0.065)
    text, _ = text_and_background_masks(image)
    mask = cv2.GaussianBlur(text.astype(np.float32), (0, 0), 0.8)[:, :, None]
    # Background stays mostly untouched so paper grain and noise are not amplified.
    weight = (0.20 + 0.80 * mask) * amount
    return np.clip(np.rint((source + detail * weight) * 255), 0, 255).astype(np.uint8)
