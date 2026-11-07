from __future__ import annotations

import cv2
import numpy as np

from ..quality.assess import text_and_background_masks


def denoise_document(image: np.ndarray, noise_sigma: float, threshold: float = 0.018,
                     strength: float = 5.0) -> np.ndarray:
    if noise_sigma < threshold:
        return image.copy()
    text, _ = text_and_background_masks(image)
    if noise_sigma > 0.035 and max(image.shape[:2]) <= 2400:
        filtered = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        filtered = cv2.bilateralFilter(image, 7, 35, 5)
    protection = cv2.dilate(text.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32) * 0.72
    protection = cv2.GaussianBlur(protection, (0, 0), 0.6)[:, :, None]
    return np.clip(filtered * (1 - protection) + image * protection, 0, 255).astype(np.uint8)
