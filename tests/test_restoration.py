import numpy as np

from docrestore.restoration.glare import detect_glare, repair_glare
from docrestore.synthetic import apply_glare, multipage_canvas


def test_small_glare_detection_and_repair():
    image = np.full((160, 180, 3), 160, np.uint8)
    degraded, _ = apply_glare(image, radius=0.06)
    mask = detect_glare(degraded, threshold=0.90)
    assert mask.any()
    repaired, _, attempted = repair_glare(degraded, threshold=0.90, max_inpaint_ratio=0.2)
    assert attempted
    assert repaired.shape == image.shape


def test_large_glare_is_not_hallucinated():
    image = np.full((100, 100, 3), 255, np.uint8)
    repaired, _, attempted = repair_glare(image, max_inpaint_ratio=0.01)
    assert not attempted
    assert np.array_equal(repaired, image)
