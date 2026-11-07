from docrestore.io import imread, imwrite
import numpy as np


def test_unicode_roundtrip(tmp_path):
    image = np.zeros((12, 15, 3), np.uint8)
    image[:, :, 1] = 120
    path = imwrite(tmp_path / "中文 文档.png", image)
    assert np.array_equal(imread(path), image)
