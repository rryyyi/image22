from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Quality:
    background_std: float = 0.0
    illumination_variation: float = 0.0
    edge_energy: float = 0.0
    laplacian_variance: float = 0.0
    noise_sigma: float = 0.0
    glare_ratio: float = 0.0
    clipped_black_ratio: float = 0.0
    clipped_white_ratio: float = 0.0
    contrast: float = 0.0
    readability_score: float = 0.0
    readability_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(slots=True)
class PageGeometry:
    quad: list[list[float]]
    bbox: list[int]
    confidence: float
    source_size: list[int]
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(slots=True)
class PageResult:
    index: int
    image: np.ndarray
    geometry: PageGeometry
    before: Quality
    after: Quality
    warnings: list[dict[str, str]] = field(default_factory=list)
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    pages: list[PageResult]
    combined: np.ndarray
    elapsed_seconds: float

    def manifest(self) -> dict[str, Any]:
        return {
            "page_count": len(self.pages), "elapsed_seconds": self.elapsed_seconds,
            "pages": [{
                "index": p.index, "geometry": p.geometry.to_dict(),
                "before": p.before.to_dict(), "after": p.after.to_dict(),
                "warnings": p.warnings,
            } for p in self.pages],
        }
