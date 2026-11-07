from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import yaml

from .config import PipelineConfig, load_config
from .geometry.rectify import rectify_page
from .illumination.restore import restore_illumination
from .io import imread, imwrite, write_json
from .pages.detect import detect_pages
from .quality.assess import assess_quality
from .quality.failure import detect_failures
from .restoration.deblur import conservative_sharpen
from .restoration.denoise import denoise_document
from .restoration.fading import restore_fading
from .restoration.glare import repair_glare
from .restoration.scratches import repair_scratches
from .types import PageResult, PipelineResult


def _contact_sheet(pages: list[np.ndarray], max_width: int = 1800) -> np.ndarray:
    if len(pages) == 1:
        return pages[0].copy()
    thumb_width = min(900, max_width // 2)
    thumbs = []
    for index, page in enumerate(pages):
        scale = thumb_width / page.shape[1]
        thumb = cv2.resize(page, (thumb_width, round(page.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb.shape[0] + 46, thumb_width, 3), 255, np.uint8)
        canvas[46:] = thumb
        cv2.putText(canvas, f"Page {index+1}", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        thumbs.append(canvas)
    rows = []
    for index in range(0, len(thumbs), 2):
        pair = thumbs[index:index+2]
        height = max(item.shape[0] for item in pair)
        normalized = [np.full((height, thumb_width, 3), 245, np.uint8) for _ in pair]
        for j, item in enumerate(pair): normalized[j][:item.shape[0]] = item
        rows.append(np.concatenate(normalized, axis=1) if len(normalized) == 2 else normalized[0])
    width = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, width-row.shape[1], cv2.BORDER_CONSTANT, value=(245,245,245)) for row in rows]
    return np.concatenate(rows, axis=0)


class DocumentPipeline:
    def __init__(self, config: PipelineConfig | dict[str, Any] | None = None):
        if config is None:
            self.config = load_config()
        elif isinstance(config, PipelineConfig):
            self.config = config
        else:
            self.config = PipelineConfig.from_mapping(config)

    def process(self, image: np.ndarray) -> PipelineResult:
        started = perf_counter()
        np.random.seed(int(self.config.data["seed"]))
        cv2.setRNGSeed(int(self.config.data["seed"]))
        pages_config = self.config.pages
        geometries = detect_pages(image, bool(pages_config["enabled"]), int(pages_config["max_pages"]),
                                  float(pages_config["min_area_ratio"]))
        results: list[PageResult] = []
        restored_pages: list[np.ndarray] = []
        for index, geometry in enumerate(geometries):
            page = rectify_page(image, geometry, bool(pages_config["perspective_enabled"]),
                                bool(pages_config["white_background"]))
            before = assess_quality(page)
            work = page.copy()
            intermediates: dict[str, np.ndarray] = {}
            restoration = self.config.restoration
            if restoration["glare"]:
                work, glare_mask, _ = repair_glare(work, float(restoration["glare_threshold"]),
                                                   float(restoration["max_inpaint_ratio"]))
                intermediates["glare_mask"] = glare_mask
            if self.config.illumination["enabled"]:
                work, illumination_data = restore_illumination(work, self.config.illumination)
                intermediates.update(illumination_data)
            if restoration["denoise"]:
                work = denoise_document(work, before.noise_sigma, float(restoration["noise_threshold"]),
                                        float(restoration["denoise_strength"]))
            if restoration["fading"]:
                work = restore_fading(work, float(restoration["fade_strength"]), float(restoration["clahe_clip"]))
            if restoration["scratches"]:
                work, scratch_mask = repair_scratches(work, float(restoration["scratch_strength"]))
                intermediates["scratch_mask"] = scratch_mask
            if restoration["deblur"]:
                work = conservative_sharpen(work, float(restoration["sharpen_amount"]),
                                            float(restoration["sharpen_radius"]))
            after = assess_quality(work)
            warnings = detect_failures(before, after, geometry, float(self.config.quality["warn_edge_retention"]),
                                       float(self.config.quality["warn_glare_ratio"]))
            results.append(PageResult(index, work, geometry, before, after, warnings, intermediates))
            restored_pages.append(work)
        return PipelineResult(results, _contact_sheet(restored_pages), perf_counter()-started)

    def process_path(self, input_path: str | Path, output_dir: str | Path,
                     save_intermediates: bool | None = None) -> PipelineResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image = imread(input_path)
        result = self.process(image)
        save_intermediates = self.config.data["output"]["save_intermediates"] if save_intermediates is None else save_intermediates
        for page in result.pages:
            stem = f"page_{page.index+1:02d}"
            imwrite(output_dir / f"{stem}.png", page.image)
            if save_intermediates:
                for name, value in page.intermediates.items():
                    if value.ndim == 2 and value.dtype != np.uint8:
                        low, high = float(value.min()), float(value.max())
                        value = np.clip((value-low) / max(1e-6, high-low) * 255, 0, 255).astype(np.uint8)
                    imwrite(output_dir / "intermediates" / f"{stem}_{name}.png", value)
        imwrite(output_dir / "combined.png", result.combined)
        manifest = result.manifest()
        manifest["input"] = str(input_path.resolve())
        manifest["config"] = self.config.to_dict()
        write_json(output_dir / "manifest.json", manifest)
        with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config.to_dict(), handle, allow_unicode=True, sort_keys=False)
        return result
