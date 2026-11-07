from __future__ import annotations

import cv2
import numpy as np

from ..types import PageGeometry


def ordered_quad(quad: np.ndarray | list) -> np.ndarray:
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(differences)],
                     points[np.argmax(sums)], points[np.argmax(differences)]], np.float32)


def rectify_page(image: np.ndarray, geometry: PageGeometry, enabled: bool = True,
                 white_background: bool = True) -> np.ndarray:
    quad = ordered_quad(geometry.quad)
    height, width = image.shape[:2]
    if not enabled or geometry.fallback:
        if white_background and not geometry.fallback:
            return image.copy()
        x, y, w, h = geometry.bbox
        return image[max(0, y):min(height, y+h), max(0, x):min(width, x+w)].copy()
    tl, tr, br, bl = quad
    target_width = int(round(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))) + 1
    target_height = int(round(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))) + 1
    if target_width < 32 or target_height < 32:
        geometry.fallback = True
        geometry.confidence = 0.0
        return image.copy()
    # Avoid enormous upsampling from a noisy detector.
    scale = min(1.0, max(width, height) * 1.5 / max(target_width, target_height))
    target_width = max(32, round(target_width * scale))
    target_height = max(32, round(target_height * scale))
    destination = np.array([[0, 0], [target_width-1, 0],
                            [target_width-1, target_height-1], [0, target_height-1]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    border = (255, 255, 255) if white_background else (0, 0, 0)
    return cv2.warpPerspective(image, matrix, (target_width, target_height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
