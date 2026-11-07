from __future__ import annotations

import cv2
import numpy as np


def detect_glare(image: np.ndarray, threshold: float = 0.94) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    mask = ((gray >= threshold) & (saturation < 0.18)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    minimum = max(8, round(image.shape[0] * image.shape[1] * 0.00001))
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            cleaned[labels == index] = 255
    return cleaned


def repair_glare(image: np.ndarray, threshold: float = 0.94,
                 max_inpaint_ratio: float = 0.025) -> tuple[np.ndarray, np.ndarray, bool]:
    mask = detect_glare(image, threshold)
    ratio = float((mask > 0).mean())
    if ratio == 0:
        return image.copy(), mask, False
    if ratio > max_inpaint_ratio:
        # Large blown areas contain no reliable pixels to restore; avoid fake text.
        return image.copy(), mask, False
    expanded = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    repaired = cv2.inpaint(image, expanded, 4, cv2.INPAINT_TELEA)
    # Keep an edge-preserving blend rather than flattening the whole highlight.
    alpha = cv2.GaussianBlur((expanded > 0).astype(np.float32), (0, 0), 1.0)[:, :, None]
    result = repaired.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8), mask, True
