import numpy as np

from docrestore import DocumentPipeline, PipelineConfig
from docrestore.geometry.rectify import ordered_quad, rectify_page
from docrestore.pages.detect import detect_pages
from docrestore.synthetic import multipage_canvas
from docrestore.types import PageGeometry


def page(width=120, height=160):
    image = np.full((height, width, 3), 240, np.uint8)
    image[30:32, 20:100] = 20
    return image


def test_ordered_quad_and_rectify():
    geometry = PageGeometry([[0, 0], [119, 0], [119, 159], [0, 159]], [0, 0, 120, 160], 1, [120, 160])
    result = rectify_page(page(), geometry)
    assert result.shape[:2] == (160, 120)
    assert ordered_quad(geometry.quad).shape == (4, 2)


def test_multipage_detection_and_pipeline():
    canvas = multipage_canvas(page(), page())
    geometries = detect_pages(canvas)
    assert len(geometries) == 2
    config = PipelineConfig.from_mapping({"pages": {"enabled": True}, "illumination": {"work_size": 200, "morph_kernel": 21}})
    result = DocumentPipeline(config).process(canvas)
    assert len(result.pages) == 2
    assert result.combined.ndim == 3


def test_disabled_detection_does_not_warn():
    config = PipelineConfig.from_mapping({"pages": {"enabled": False}, "illumination": {"work_size": 200, "morph_kernel": 21}})
    result = DocumentPipeline(config).process(page())
    assert not any(item["code"] == "page_fallback" for item in result.pages[0].warnings)
