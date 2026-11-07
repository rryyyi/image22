from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 3407,
    "pages": {
        "enabled": True, "max_pages": 8, "detect_size": 1024,
        "min_area_ratio": 0.035, "white_background": True,
        "perspective_enabled": True, "min_confidence": 0.20,
    },
    "illumination": {
        "enabled": True, "estimator": "morphological", "work_size": 1200,
        "morph_kernel": 41, "guided_radius": 24, "guided_eps": 0.015,
        "paper_saturation": 0.22, "paper_darkness": 0.08,
        "paper_min_area_ratio": 0.002, "paper_dilate": 5, "paper_guided_radius": 8,
        "retinex_scales": [15, 80, 250], "water_iterations": 500,
        "target_percentile": 92, "max_gain": 3.2, "strength": 0.92,
        "residual_strength": 0.18, "color_strength": 0.30,
        "edge_protection": 0.0,
    },
    "restoration": {
        "denoise": False, "noise_threshold": 0.018, "denoise_strength": 5.0,
        "deblur": False, "sharpen_amount": 0.0, "sharpen_radius": 1.0,
        "glare": False, "glare_threshold": 0.94, "max_inpaint_ratio": 0.025,
        "scratches": False, "scratch_strength": 0.5,
        "fading": False, "fade_strength": 0.28, "clahe_clip": 1.2,
    },
    "quality": {"warn_edge_retention": 0.72, "warn_glare_ratio": 0.08},
    "output": {"save_intermediates": False, "jpeg_quality": 95},
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass(slots=True)
class PipelineConfig:
    """Validated, serializable configuration with conservative defaults."""

    data: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_CONFIG))

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None = None) -> "PipelineConfig":
        merged = _deep_update(deepcopy(DEFAULT_CONFIG), values or {})
        if merged["illumination"]["estimator"] not in {"guided", "morphological", "retinex", "contrast_guided", "paper_guided", "water_filling"}:
            raise ValueError("Unknown illumination estimator")
        if merged["illumination"]["max_gain"] < 1:
            raise ValueError("max_gain must be >= 1")
        if merged["pages"]["max_pages"] < 1:
            raise ValueError("max_pages must be positive")
        return cls(merged)

    @property
    def pages(self) -> dict[str, Any]: return self.data["pages"]
    @property
    def illumination(self) -> dict[str, Any]: return self.data["illumination"]
    @property
    def restoration(self) -> dict[str, Any]: return self.data["restoration"]
    @property
    def quality(self) -> dict[str, Any]: return self.data["quality"]

    def to_dict(self) -> dict[str, Any]: return deepcopy(self.data)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
    return PipelineConfig.from_mapping(values)
