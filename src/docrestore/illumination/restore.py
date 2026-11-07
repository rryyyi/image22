from __future__ import annotations

import cv2
import numpy as np

from ..io import to_float, to_uint8
from .estimators import estimate_color_field, estimate_illumination


def _text_edge_mask(gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    mask = np.clip(blackhat / 0.075, 0, 1) * np.clip(gradient / 0.12, 0, 1)
    return cv2.GaussianBlur(mask, (0, 0), 0.65)


def restore_illumination(image: np.ndarray, config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bgr = to_float(image)
    estimate = estimate_illumination(
        image, method=str(config.get("estimator", "guided")), work_size=int(config.get("work_size", 1200)),
        morph_kernel=int(config.get("morph_kernel", 41)), guided_radius=int(config.get("guided_radius", 24)),
        guided_eps=float(config.get("guided_eps", 0.015)), retinex_scales=config.get("retinex_scales", [15, 80, 250]),
        water_iterations=int(config.get("water_iterations", 500)),
    )
    field = estimate.field
    target = max(0.08, float(np.percentile(field, float(config.get("target_percentile", 92)))))
    maximum = float(config.get("max_gain", 3.2))
    gain = np.clip(target / np.maximum(field, 0.025), 1.0 / maximum, maximum)
    strength = float(config.get("strength", 0.92))
    gain = 1.0 + (gain - 1.0) * strength
    gain = cv2.GaussianBlur(gain, (0, 0), 1.2)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0] * gain, 0, 1)
    luminance_result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Correct coloured illumination mainly on low-saturation paper, preserving inks.
    color_strength = float(config.get("color_strength", 0.30))
    if color_strength > 0:
        color_field = estimate_color_field(image, int(config.get("work_size", 1200)), int(config.get("morph_kernel", 41)))
        color_target = np.percentile(color_field.reshape(-1, 3), float(config.get("target_percentile", 92)), axis=0)
        color_gain = np.clip(color_target / np.maximum(color_field, 0.025), 1.0 / maximum, maximum)
        color_gain = 1.0 + (color_gain - 1.0) * strength
        color_result = np.clip(bgr * color_gain, 0, 1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        paper = np.clip((0.42 - hsv[:, :, 1]) / 0.25, 0, 1) * np.clip((ycrcb[:, :, 0] - 0.18) / 0.25, 0, 1)
        alpha = (paper * color_strength)[:, :, None]
        result = luminance_result * (1 - alpha) + color_result * alpha
    else:
        result = luminance_result

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = _text_edge_mask(gray)
    protection = float(config.get("edge_protection", 0.72)) * edges[:, :, None]
    result = result * (1 - protection) + bgr * protection

    residual_strength = float(config.get("residual_strength", 0.18))
    if residual_strength > 0:
        result_gray = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_BGR2GRAY)
        low = cv2.GaussianBlur(result_gray, (0, 0), 35)
        reference = np.percentile(low, 92)
        residual = np.clip(reference / np.maximum(low, 0.08), 0.85, 1.25)
        residual = 1 + (residual - 1) * residual_strength
        result = np.clip(result * residual[:, :, None], 0, 1)

    intermediates = {"illumination": field, "gain": gain.astype(np.float32),
                     "text_edges": edges.astype(np.float32)}
    if estimate.paper_mask is not None:
        intermediates["paper_mask"] = estimate.paper_mask
        intermediates["figure_mask"] = estimate.figure_mask
    return to_uint8(result), intermediates
