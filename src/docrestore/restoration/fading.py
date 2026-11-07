from __future__ import annotations

import cv2
import numpy as np


def restore_fading(image: np.ndarray, strength: float = 0.28, clahe_clip: float = 1.2) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0]
    low, high = np.percentile(luminance, [4, 96])
    contrast = float(high - low) / 255.0
    if contrast > 0.72 or strength <= 0:
        return image.copy()
    clahe = cv2.createCLAHE(clipLimit=max(1.0, clahe_clip), tileGridSize=(8, 8))
    local = clahe.apply(luminance)
    # A percentile anchor prevents CLAHE from making aged paper pure white.
    stretched = np.clip((luminance.astype(np.float32) - low) * 210.0 / max(25.0, high-low) + 20, 0, 255)
    enhanced = 0.65 * local + 0.35 * stretched
    adaptive_strength = strength * np.clip((0.72 - contrast) / 0.35, 0.25, 1.0)
    ycrcb[:, :, 0] = np.clip(luminance * (1-adaptive_strength) + enhanced * adaptive_strength, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
