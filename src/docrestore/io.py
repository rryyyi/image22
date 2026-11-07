from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def imwrite(path: str | Path, image: np.ndarray, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    params: list[int] = []
    if suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(kwargs.get("jpeg_quality", 95))]
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buffer.tofile(str(path))
    return path


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    result = image.astype(np.float32)
    return np.clip(result, 0.0, 1.0) if result.max(initial=0) <= 1.0 else result / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def resize_long_edge(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(size) / max(height, width))
    if scale == 1.0:
        return image.copy(), scale
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale
