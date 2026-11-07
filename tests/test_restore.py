import cv2
import numpy as np

from docrestore.illumination.restore import restore_illumination
from docrestore.synthetic import apply_shadow


def config(**changes):
    base = dict(estimator="guided", work_size=160, morph_kernel=21, guided_radius=8,
                guided_eps=0.01, target_percentile=92, max_gain=3.2, strength=0.92,
                residual_strength=0.1, color_strength=0.2, edge_protection=0.7)
    base.update(changes)
    return base


def test_restore_keeps_flat_image_stable():
    image = np.full((100, 140, 3), 220, np.uint8)
    result, data = restore_illumination(image, config())
    assert result.shape == image.shape
    assert np.isfinite(data["gain"]).all()
    assert abs(float(result.mean()) - 220) < 8


def test_restore_reduces_synthetic_shadow():
    clean = np.full((150, 180, 3), 225, np.uint8)
    cv2.putText(clean, "TEXT", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    shadowed, field, mask = apply_shadow(clean, colored=False)
    result, _ = restore_illumination(shadowed, config())
    before = cv2.cvtColor(shadowed, cv2.COLOR_BGR2GRAY).astype(float)
    after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(float)
    background = mask & (cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) > 200)
    assert after[background].mean() > before[background].mean()
