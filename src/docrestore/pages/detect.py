from __future__ import annotations

import cv2
import numpy as np

from ..io import resize_long_edge
from ..types import PageGeometry


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                     points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)


def _page_candidates(image: np.ndarray, min_area_ratio: float) -> list[PageGeometry]:
    small, scale = resize_long_edge(image, 1024)
    height, width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    border = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3),
                             lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    color_distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    bright = gray > np.percentile(gray, 55)
    separated = color_distance > max(12.0, float(np.percentile(color_distance, 65)))
    mask = ((bright | separated) * 255).astype(np.uint8)
    kernel_size = max(3, round(min(height, width) * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[PageGeometry] = []
    image_area = height * width
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < image_area * min_area_ratio:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(polygon) >= 4:
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect) if len(polygon) > 6 else _quad_from_polygon(polygon, hull)
        else:
            quad = cv2.boxPoints(cv2.minAreaRect(hull))
        quad = _order_quad(quad) / scale
        rectangularity = min(1.0, area / max(1.0, cv2.contourArea(cv2.convexHull(quad.astype(np.int32)))))
        coverage = min(1.0, area / image_area / 0.25)
        confidence = float(np.clip(0.55 * rectangularity + 0.45 * coverage, 0, 1))
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        results.append(PageGeometry(quad.tolist(), [x, y, w, h], confidence,
                                    [image.shape[1], image.shape[0]]))
        if len(results) >= 8:
            break
    # Remove detections almost contained in a higher-confidence page.
    filtered: list[PageGeometry] = []
    for candidate in results:
        x, y, w, h = candidate.bbox
        duplicate = False
        for other in filtered:
            ox, oy, ow, oh = other.bbox
            intersection = max(0, min(x+w, ox+ow)-max(x, ox)) * max(0, min(y+h, oy+oh)-max(y, oy))
            if intersection / max(1, w*h) > 0.82:
                duplicate = True
                break
        if not duplicate:
            filtered.append(candidate)
    return sorted(filtered, key=lambda p: (p.bbox[1], p.bbox[0]))


def _quad_from_polygon(polygon: np.ndarray, hull: np.ndarray) -> np.ndarray:
    if len(polygon) == 4:
        return polygon.reshape(4, 2)
    # Farthest-point sampling retains the page's extreme corners better than a bbox.
    points = hull.reshape(-1, 2).astype(np.float32)
    selected = [int(np.argmin(points[:, 0] + points[:, 1]))]
    while len(selected) < 4:
        distances = np.min(np.stack([np.linalg.norm(points - points[i], axis=1) for i in selected]), axis=0)
        selected.append(int(np.argmax(distances)))
    return points[selected]


def detect_pages(image: np.ndarray, enabled: bool = True, max_pages: int = 8,
                 min_area_ratio: float = 0.035) -> list[PageGeometry]:
    height, width = image.shape[:2]
    fallback = PageGeometry([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                            [0, 0, width, height], 0.0, [width, height], True)
    if not enabled:
        fallback.confidence = 1.0
        return [fallback]
    candidates = _page_candidates(image, min_area_ratio)
    if not candidates:
        return [fallback]
    # A small, low-confidence contour is more likely clutter than a real page.
    candidates = [p for p in candidates if p.confidence >= 0.20]
    return (candidates or [fallback])[:max_pages]
