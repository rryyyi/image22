from __future__ import annotations

import cv2
import numpy as np


def repair_scratches(image: np.ndarray, strength: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Conservative long-thin defect repair; disabled by default in the pipeline."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = []
    for kernel_size in ((25, 1), (1, 25)):
        line = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, line)
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, line)
        masks.append(((bright > 28) | (dark > 32)).astype(np.uint8))
    candidate = (np.maximum(*masks) * 255).astype(np.uint8)
    # Protect long continuous table rulings, which are not scratches.
    horizontal = cv2.morphologyEx((gray < 100).astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)))
    vertical = cv2.morphologyEx((gray < 100).astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80)))
    candidate[(horizontal.astype(bool) | vertical.astype(bool))] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    selected = np.zeros_like(candidate)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        aspect = max(w, h) / max(1, min(w, h))
        if area >= 12 and aspect >= 4 and max(w, h) <= max(image.shape[:2]) * 0.45:
            selected[labels == index] = 255
    if not selected.any() or strength <= 0:
        return image.copy(), selected
    repaired = cv2.inpaint(image, selected, 3, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur((selected > 0).astype(np.float32), (0, 0), 0.8)[:, :, None] * strength
    return np.clip(repaired * alpha + image * (1-alpha), 0, 255).astype(np.uint8), selected
