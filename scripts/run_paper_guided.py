from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from docrestore import DocumentPipeline, PipelineConfig
from docrestore.io import imread, imwrite
from docrestore.quality.assess import assess_quality, derive_shadow_mask, reference_metrics
from run_benchmark import ROOT, jung_pairs

VARIANTS = {
    "morphological": {"illumination": {"estimator": "morphological"}},
    "paper_guided": {"illumination": {"estimator": "paper_guided"}},
}
FAILURES = {"jung_06", "jung_09", "jung_10", "jung_05", "jung_18"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="results/benchmark_paper_guided")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    pipelines = {name: DocumentPipeline(PipelineConfig.from_mapping({"pages": {"enabled": False}, **config}))
                 for name, config in VARIANTS.items()}
    rows = []
    for sample, input_path, target_path in jung_pairs()[:args.limit or None]:
        source, target = imread(input_path), imread(target_path)
        shadow = derive_shadow_mask(source, target)
        regions = {"full": np.ones(shadow.shape, bool), "shadow": shadow, "non_shadow": ~shadow}
        for variant, pipeline in pipelines.items():
            result = pipeline.process(source)
            page = result.pages[0]
            prediction = page.image
            if args.save_all if hasattr(args, "save_all") else True:
                imwrite(output / "outputs" / variant / f"{sample}.png", prediction)
            if sample in FAILURES:
                imwrite(output / "masks" / variant / f"{sample}_paper.png", page.intermediates.get("paper_mask", np.zeros(source.shape[:2], np.uint8)))
                imwrite(output / "masks" / variant / f"{sample}_figure.png", page.intermediates.get("figure_mask", np.zeros(source.shape[:2], np.uint8)))
                imwrite(output / "masks" / variant / f"{sample}_illumination.png",
                        np.rint(page.intermediates["illumination"] * 255).astype(np.uint8))
            metrics = reference_metrics(prediction, target, regions)
            quality = assess_quality(prediction)
            for region, values in metrics.items():
                rows.append({"sample": sample, "variant": variant, "region": region, **values,
                             "runtime_seconds": result.elapsed_seconds,
                             "background_std": quality.background_std,
                             "illumination_variation": quality.illumination_variation,
                             "edge_energy": quality.edge_energy,
                             "laplacian_variance": quality.laplacian_variance,
                             "noise_sigma": quality.noise_sigma,
                             "readability_score": quality.readability_score})
        print("processed", sample, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_image.csv", index=False)
    numeric = ["psnr", "ssim", "rmse", "runtime_seconds", "background_std", "illumination_variation",
               "edge_energy", "laplacian_variance", "noise_sigma", "readability_score"]
    summary = frame.groupby(["variant", "region"], as_index=False)[numeric].mean(numeric_only=True)
    summary.to_csv(ROOT / "results/tables/paper_guided_summary.csv", index=False)
    frame.to_csv(ROOT / "results/tables/paper_guided_per_image.csv", index=False)
    full = frame[frame.region == "full"].pivot(index="sample", columns="variant", values="psnr")
    full["delta"] = full.paper_guided - full.morphological
    full.sort_values("delta").to_csv(ROOT / "results/tables/paper_guided_delta.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
