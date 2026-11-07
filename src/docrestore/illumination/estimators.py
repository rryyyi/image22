from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ..io import resize_long_edge, to_float


@dataclass(slots=True)
class IlluminationEstimate:
    field: np.ndarray
    work_field: np.ndarray
    method: str
    elapsed_seconds: float
    parameters: dict[str, Any]
    paper_mask: np.ndarray | None = None
    figure_mask: np.ndarray | None = None


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Fast edge-aware smoothing implemented with OpenCV box filters only."""
    guide = guide.astype(np.float32)
    source = source.astype(np.float32)
    kernel = (2 * max(1, radius) + 1,) * 2
    mean_guide = cv2.boxFilter(guide, -1, kernel, normalize=True)
    mean_source = cv2.boxFilter(source, -1, kernel, normalize=True)
    corr_guide = cv2.boxFilter(guide * guide, -1, kernel, normalize=True)
    corr_product = cv2.boxFilter(guide * source, -1, kernel, normalize=True)
    variance = corr_guide - mean_guide * mean_guide
    covariance = corr_product - mean_guide * mean_source
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    return cv2.boxFilter(a, -1, kernel, normalize=True) * guide + cv2.boxFilter(b, -1, kernel, normalize=True)


def _morphological(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    size = max(5, int(kernel_size) | 1)
    size = min(size, max(5, (min(gray.shape) // 3) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(background, (0, 0), max(1.0, size / 8))


def _water_filling(gray: np.ndarray, iterations: int) -> np.ndarray:
    """Vectorized water-filling interpretation, designed for modest CPU memory.

    The original C++ update is mapped to array operations. A finite exponential
    schedule spreads pouring over the configured iteration budget.
    """
    height = max(24, round(gray.shape[0] * 0.2))
    width = max(24, round(gray.shape[1] * 0.2))
    terrain = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    water = np.zeros_like(terrain)
    eta = 0.2
    iterations = max(20, int(iterations))
    for t in range(iterations):
        surface = water + terrain
        padded = np.pad(surface, 1, mode="edge")
        neighbours = (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
        diffusion = sum(np.minimum(neighbour - surface, 0.0) for neighbour in neighbours)
        pouring = np.exp(-8.0 * t / iterations) * (float(surface.max()) - surface)
        water = np.maximum(0.0, water + eta * diffusion + pouring)
    surface = water + terrain
    # A short increasing-fill pass removes small remaining text depressions.
    terrain = surface.copy()
    water.fill(0)
    for _ in range(40):
        surface = water + terrain
        padded = np.pad(surface, 1, mode="edge")
        neighbours = (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
        diffusion = sum(neighbour - surface for neighbour in neighbours)
        water = np.maximum(0.0, water + eta * diffusion)
    return cv2.resize(surface, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)



def _large_components(mask: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
    """Keep connected dark/coloured blocks while preserving small text strokes."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    selected = np.zeros_like(binary)
    limit = max(12, int(binary.size * float(min_area_ratio)))
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= limit:
            selected[labels == index] = 255
    return selected


def estimate_paper_mask(small_bgr: np.ndarray, small_gray: np.ndarray, coarse_background: np.ndarray,
                        saturation_threshold: float = 0.22, darkness_threshold: float = 0.08,
                        min_area_ratio: float = 0.002, dilate_size: int = 5) -> np.ndarray:
    """Separate illustration blocks from paper while keeping text and table lines.

    Shadows remain paper because they lower both the observation and the coarse
    background together. Figures are dark or coloured relative to their local
    paper background and often form large connected blocks.
    """
    hsv = cv2.cvtColor(np.clip(small_bgr, 0, 1), cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    dark_ink = np.clip(coarse_background - small_gray, 0, 1) > float(darkness_threshold)
    colored = saturation > float(saturation_threshold)
    blocks = _large_components((dark_ink | colored).astype(np.uint8), min_area_ratio)
    size = max(3, int(dilate_size) | 1)
    if blocks.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        blocks = cv2.dilate(blocks, kernel)
    paper = np.where(blocks > 0, 0, 255).astype(np.uint8)
    # Close tiny mask holes so the illumination source is spatially coherent.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)
    return paper.astype(np.uint8)


def _paper_guided(small_gray: np.ndarray, paper_mask: np.ndarray,
                  parameters: dict[str, Any]) -> np.ndarray:
    coarse = _morphological(small_gray, int(parameters.get("morph_kernel", 41))).astype(np.float32)
    figures = (paper_mask == 0).astype(np.uint8)
    if figures.any():
        # Inpaint illustration blocks with surrounding paper background only.
        filled = cv2.inpaint(np.clip(coarse * 255, 0, 255).astype(np.uint8), figures, 7, cv2.INPAINT_TELEA)
        coarse = filled.astype(np.float32) / 255.0
    radius = max(2, int(parameters.get("paper_guided_radius", 8)))
    eps = max(0.001, float(parameters.get("guided_eps", 0.015)))
    # Replace only illustration blocks with inpainted paper background. Keeping
    # the original morphological field elsewhere preserves shadow boundaries.
    hybrid = coarse.copy()
    if figures.any():
        filled = cv2.inpaint(np.clip(coarse * 255, 0, 255).astype(np.uint8), figures, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(figures.astype(np.float32), (0, 0), 1.2)
        hybrid = coarse * (1 - alpha) + filled * alpha
    return np.maximum(0.01, guided_filter(hybrid, hybrid, radius, eps))



def _estimate_work(gray: np.ndarray, method: str, parameters: dict[str, Any]) -> np.ndarray:
    morph = _morphological(gray, int(parameters.get("morph_kernel", 41)))
    if method == "morphological":
        return morph
    if method == "guided":
        return np.maximum(0.01, guided_filter(gray, morph, int(parameters.get("guided_radius", 24)),
                                              float(parameters.get("guided_eps", 0.015))))
    if method == "retinex":
        log_image = np.log(np.maximum(morph, 0.015))
        scales = parameters.get("retinex_scales", [15, 80, 250])
        illumination = sum(cv2.GaussianBlur(log_image, (0, 0), float(sigma)) for sigma in scales) / len(scales)
        return np.exp(illumination).astype(np.float32)
    if method == "contrast_guided":
        mean = cv2.boxFilter(gray, -1, (31, 31))
        square = cv2.boxFilter(gray * gray, -1, (31, 31))
        contrast = np.sqrt(np.maximum(0.0, square - mean * mean))
        low_contrast = 1.0 - np.clip(contrast / 0.09, 0, 1)
        guided = guided_filter(gray, morph, int(parameters.get("guided_radius", 24)), 0.01)
        smooth = cv2.GaussianBlur(morph, (0, 0), 12)
        return np.clip(guided * (1 - 0.35 * low_contrast) + smooth * (0.35 * low_contrast), 0.01, None)
    if method == "paper_guided":
        raise RuntimeError("paper_guided requires colour input and is handled by estimate_illumination")
    if method == "water_filling":
        return np.maximum(0.01, _water_filling(gray, int(parameters.get("water_iterations", 500))))
    raise ValueError(f"Unknown estimator: {method}")


def estimate_illumination(image: np.ndarray, method: str = "guided", work_size: int = 1200,
                          **parameters: Any) -> IlluminationEstimate:
    started = perf_counter()
    bgr = to_float(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper_mask = None
    figure_mask = None
    if method == "paper_guided":
        small_bgr, _ = resize_long_edge(bgr, work_size)
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        coarse = _morphological(small_gray, int(parameters.get("morph_kernel", 41)))
        figure_mask = 255 - estimate_paper_mask(
            small_bgr, small_gray, coarse,
            saturation_threshold=float(parameters.get("paper_saturation", 0.22)),
            darkness_threshold=float(parameters.get("paper_darkness", 0.08)),
            min_area_ratio=float(parameters.get("paper_min_area_ratio", 0.002)),
            dilate_size=int(parameters.get("paper_dilate", 5)),
        )
        paper_mask = 255 - figure_mask
        work = _paper_guided(small_gray, paper_mask, parameters).astype(np.float32)
    else:
        small, _ = resize_long_edge(gray, work_size)
        work = _estimate_work(small, method, parameters).astype(np.float32)
    field = cv2.resize(work, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    field = np.clip(field, 0.01, 1.0).astype(np.float32)
    if paper_mask is not None:
        paper_mask = cv2.resize(paper_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        figure_mask = cv2.resize(figure_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return IlluminationEstimate(field, work, method, perf_counter() - started, dict(parameters),
                                paper_mask, figure_mask)


def estimate_color_field(image: np.ndarray, work_size: int = 900, kernel_size: int = 41) -> np.ndarray:
    """Estimate paper chroma without applying a global white balance to stamps."""
    bgr = to_float(image)
    small, _ = resize_long_edge(bgr, work_size)
    channels = [_morphological(small[:, :, index], kernel_size) for index in range(3)]
    field = np.stack(channels, axis=2)
    return np.clip(cv2.resize(field, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC), 0.01, 1.0)

