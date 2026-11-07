import cv2
import numpy as np
import pytest

from docrestore.illumination.estimators import estimate_illumination, guided_filter


@pytest.mark.parametrize("method", ["morphological", "guided", "retinex", "contrast_guided", "water_filling"])
def test_estimators_are_finite_and_full_size(method):
    image = np.full((90, 120, 3), 210, np.uint8)
    image[30:60, 30:60] = 40
    estimate = estimate_illumination(image, method, work_size=100, water_iterations=20)
    assert estimate.field.shape == image.shape[:2]
    assert np.isfinite(estimate.field).all()
    assert 0.01 <= estimate.field.min() <= estimate.field.max() <= 1.0


def test_guided_filter_constant_source():
    guide = np.zeros((40, 50), np.float32)
    guide[:, 25:] = 1
    result = guided_filter(guide, np.full_like(guide, 0.7), 4, 0.01)
    assert np.allclose(result, 0.7, atol=1e-5)
