from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from docrestore.io import imread, imwrite
from docrestore.synthetic import apply_glare, apply_noise_blur_fade, apply_scratches, apply_shadow, multipage_canvas

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "data/synthetic"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for path in (ROOT / "third_party/Water-Filling/GroundTruth").iterdir():
        match = re.search(r"rect_(\d+)", path.name)
        if match and int(match.group(1)) <= 5: paths.append(path)
    paths = sorted(paths)
    clean_images = []
    for index, path in enumerate(paths, 1):
        clean = imread(path)
        clean_images.append(clean)
        shadowed, field, mask = apply_shadow(clean, seed=3407+index)
        glare, glare_mask = apply_glare(shadowed)
        degraded = apply_noise_blur_fade(glare, seed=5407+index)
        degraded, scratches = apply_scratches(degraded, seed=6407+index)
        stem = f"synthetic_{index:02d}"
        imwrite(output / f"{stem}_clean.png", clean)
        imwrite(output / f"{stem}_degraded.png", degraded)
        imwrite(output / f"{stem}_illumination.png", np.rint(field*255).astype(np.uint8))
        imwrite(output / f"{stem}_shadow_mask.png", mask.astype(np.uint8)*255)
        imwrite(output / f"{stem}_glare_mask.png", glare_mask.astype(np.uint8)*255)
        imwrite(output / f"{stem}_scratch_mask.png", scratches.astype(np.uint8)*255)
    if len(clean_images) >= 2:
        imwrite(output / "multipage.png", multipage_canvas(clean_images[0], clean_images[1]))
    metadata = {"seed": 3407, "samples": len(paths), "source": "Water-Filling GroundTruth",
                "degradations": ["colored_gaussian_shadow", "linear_illumination", "glare", "gaussian_noise",
                                 "gaussian_blur", "fading", "thin_scratches"],
                "note": "Synthetic data supplements failure analysis; it does not replace real benchmarks."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
