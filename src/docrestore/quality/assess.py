from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from ..io import to_float
from ..types import Quality


def text_and_background_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bgr = to_float(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    text = (blackhat > 0.025) | (gradient > 0.16)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    background = (gray > np.percentile(gray, 55)) & (gradient < 0.035) & (hsv[:, :, 1] < 0.45)
    background = cv2.morphologyEx(background.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    if background.sum() < 100:
        background = gray > np.median(gray)
    return text, background


def assess_quality(image: np.ndarray) -> Quality:
    bgr = to_float(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    text, background = text_and_background_masks(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    low = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    low = cv2.GaussianBlur(low, (0, 0), 15)
    mean = cv2.boxFilter(low, -1, (41, 41))
    mean2 = cv2.boxFilter(low * low, -1, (41, 41))
    local_variance = np.maximum(0, mean2 - mean * mean)
    illumination_variation = float(np.sqrt(local_variance.mean()) / max(0.05, low.mean()))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    edge_energy = float(gradient[text].mean()) if text.any() else float(gradient.mean())
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    high_pass = gray - cv2.medianBlur(gray, 3)
    values = high_pass[background]
    noise_sigma = float(np.median(np.abs(values)) / 0.6745) if values.size else 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    glare_ratio = float(((gray > 0.94) & (hsv[:, :, 1] < 0.18)).mean())
    p05, p95 = np.percentile(gray, [5, 95])
    background_std = float(gray[background].std()) if background.any() else 0.0
    contrast = float(p95 - p05)
    black_ratio = float((gray < 0.02).mean())
    white_ratio = float((gray > 0.995).mean())
    sharp_score = np.clip(laplacian_variance / 0.012, 0, 1)
    contrast_score = np.clip(contrast / 0.55, 0, 1)
    uniform_score = 1 - np.clip(illumination_variation / 0.12, 0, 1)
    noise_score = 1 - np.clip(noise_sigma / 0.055, 0, 1)
    glare_score = 1 - np.clip(glare_ratio / 0.12, 0, 1)
    score = float(100 * (0.30 * sharp_score + 0.25 * contrast_score + 0.20 * uniform_score
                         + 0.15 * noise_score + 0.10 * glare_score))
    label = "readable" if score >= 70 else "review" if score >= 45 else "poor"
    return Quality(background_std, illumination_variation, edge_energy, laplacian_variance,
                   noise_sigma, glare_ratio, black_ratio, white_ratio, contrast, score, label)


def derive_shadow_mask(shadowed: np.ndarray, clean: np.ndarray, threshold: float = 0.07) -> np.ndarray:
    a = cv2.cvtColor(to_float(shadowed), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(to_float(clean), cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    a_low = cv2.GaussianBlur(cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel), (0, 0), 12)
    b_low = cv2.GaussianBlur(cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel), (0, 0), 12)
    mask = (b_low - a_low) > threshold
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.dilate(mask, np.ones((9, 9), np.uint8)).astype(bool)


def reference_metrics(prediction: np.ndarray, target: np.ndarray,
                      regions: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, float]]:
    pred = to_float(prediction)
    gt = to_float(target)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    regions = regions or {"full": np.ones(pred.shape[:2], bool)}
    results: dict[str, dict[str, float]] = {}
    for name, mask in regions.items():
        mask = mask.astype(bool)
        count = int(mask.sum())
        if count == 0:
            results[name] = {"psnr": float("nan"), "ssim": float("nan"), "rmse": float("nan"), "pixels": 0}
            continue
        error = pred - gt
        mse = float(np.mean(error[mask] ** 2))
        psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
        ssim, score_map = structural_similarity(gt, pred, channel_axis=2, data_range=1.0, full=True)
        # A map average respects irregular masks better than pasting black pixels.
        masked_ssim = float(score_map[mask].mean()) if score_map.shape == mask.shape else float(ssim)
        results[name] = {"psnr": psnr, "ssim": masked_ssim, "rmse": float(np.sqrt(mse)), "pixels": count}
    return results
