from __future__ import annotations

from ..types import PageGeometry, Quality


def detect_failures(before: Quality, after: Quality, geometry: PageGeometry,
                    warn_edge_retention: float = 0.72, warn_glare_ratio: float = 0.08) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    def add(code: str, message: str, severity: str = "warning") -> None:
        warnings.append({"code": code, "message": message, "severity": severity})
    if geometry.fallback and geometry.confidence == 0.0:
        add("page_fallback", "No reliable page boundary was found; the full frame was processed.")
    retention = after.edge_energy / max(1e-6, before.edge_energy)
    if retention < warn_edge_retention:
        add("edge_loss", f"Text-edge energy fell to {retention:.1%} of the input; inspect small text.")
    if after.glare_ratio > warn_glare_ratio:
        add("glare_residual", f"Possible glare remains on {after.glare_ratio:.1%} of pixels.")
    if after.clipped_white_ratio > 0.12:
        add("highlight_clipping", "Highlight clipping is high; illumination recovery may be too strong.")
    if after.laplacian_variance < 0.0003:
        add("severe_blur", "The result is still very blurry and needs manual review.", "error")
    if after.readability_label == "poor":
        add("readability_poor", "The heuristic readability score remains low; do not archive without review.")
    return warnings
