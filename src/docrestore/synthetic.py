from __future__ import annotations

import cv2
import numpy as np


def apply_shadow(image: np.ndarray, seed: int = 3407, colored: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= max(1, width-1); y /= max(1, height-1)
    cx, cy = rng.uniform(0.25, 0.75, 2)
    blob = np.exp(-(((x-cx)/0.34)**2 + ((y-cy)/0.42)**2) * 2.0)
    gradient = np.clip(0.55*x + 0.45*y, 0, 1)
    field = np.clip(1.0 - 0.42*blob - 0.16*gradient, 0.42, 1.0).astype(np.float32)
    source = image.astype(np.float32) / 255.0
    if colored:
        tint = np.stack([field*0.96, field*0.985, field], axis=2)
    else:
        tint = field[:, :, None]
    degraded = np.clip(source * tint, 0, 1)
    return np.rint(degraded*255).astype(np.uint8), field, (field < 0.88)


def apply_glare(image: np.ndarray, center: tuple[float, float] = (0.72, 0.28), radius: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= width; y /= height
    distance = ((x-center[0])**2 + (y-center[1])**2)
    mask = np.clip(1 - distance / max(1e-5, radius**2), 0, 1) ** 2
    result = image.astype(np.float32)/255.0
    result = result*(1-0.88*mask[:, :, None]) + 0.88*mask[:, :, None]
    return np.rint(np.clip(result, 0, 1)*255).astype(np.uint8), mask > 0.2


def apply_noise_blur_fade(image: np.ndarray, seed: int = 3407) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blurred = cv2.GaussianBlur(image, (0, 0), 0.75).astype(np.float32)
    noise = rng.normal(0, 4.0, image.shape).astype(np.float32)
    faded = blurred*0.82 + 255*0.13 + noise
    return np.clip(faded, 0, 255).astype(np.uint8)


def apply_scratches(image: np.ndarray, seed: int = 3407) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    for _ in range(3):
        x = int(rng.integers(image.shape[1]//8, 7*image.shape[1]//8))
        y = int(rng.integers(image.shape[0]//8, 7*image.shape[0]//8))
        end = (min(image.shape[1]-1, x+int(rng.integers(-30, 31))), min(image.shape[0]-1, y+int(rng.integers(60, 180))))
        cv2.line(mask, (x, y), end, 255, 1)
    cv2.line(result, (0, 0), (0, 0), 0)
    result[mask > 0] = 245
    return result, mask > 0


def multipage_canvas(first: np.ndarray, second: np.ndarray, gap: int = 60) -> np.ndarray:
    height = max(first.shape[0], second.shape[0]) + 100
    width = first.shape[1] + second.shape[1] + gap + 100
    canvas = np.full((height, width, 3), 65, np.uint8)
    canvas[50:50+first.shape[0], 40:40+first.shape[1]] = first
    x2 = 40 + first.shape[1] + gap
    canvas[50:50+second.shape[0], x2:x2+second.shape[1]] = second
    return canvas
