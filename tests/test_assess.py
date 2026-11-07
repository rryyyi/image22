import numpy as np

from docrestore.quality.assess import assess_quality, derive_shadow_mask, reference_metrics


def test_reference_metrics_identical():
    image = np.full((60, 80, 3), 180, np.uint8)
    image[20:30, 20:30] = 10
    mask = np.zeros(image.shape[:2], bool); mask[:, :40] = True
    result = reference_metrics(image, image, {"full": np.ones(mask.shape, bool), "left": mask, "right": ~mask})
    assert result["full"]["ssim"] == 1.0
    assert result["left"]["rmse"] == 0
    assert result["right"]["pixels"] == 60*40


def test_quality_and_shadow_mask():
    clean = np.full((80, 100, 3), 230, np.uint8)
    shadow = clean.copy(); shadow[:, :50] = 120
    mask = derive_shadow_mask(shadow, clean, threshold=0.1)
    assert mask[:, 10].mean() > 0.8
    assert mask[:, 90].mean() < 0.1
    quality = assess_quality(clean)
    assert quality.readability_label in {"readable", "review", "poor"}
