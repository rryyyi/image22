from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from docrestore import DocumentPipeline, PipelineConfig
from docrestore.io import imread, imwrite
from docrestore.quality.assess import assess_quality, derive_shadow_mask, reference_metrics

ROOT = Path(__file__).resolve().parents[1]


def jung_pairs() -> list[tuple[str, Path, Path]]:
    root = ROOT / "third_party/Water-Filling"
    def indexed(folder: Path) -> dict[int, Path]:
        result = {}
        for path in folder.iterdir():
            match = re.search(r"rect_(\d+)", path.name)
            if match: result[int(match.group(1))] = path
        return result
    originals, targets = indexed(root / "Original"), indexed(root / "GroundTruth")
    return [(f"jung_{number:02d}", originals[number], targets[number]) for number in sorted(originals.keys() & targets.keys())]


def doc_image_tool_runner():
    path = ROOT / "third_party/Doc-Image-Tool/function_method/DocShadowRemoval.py"
    spec = importlib.util.spec_from_file_location("doc_image_tool_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.removeShadow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="input,doc_image_tool,ours,docshadow")
    parser.add_argument("--limit", type=int, default=0, help="0 uses all 20 paired Jung images")
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()
    methods = args.methods.split(",")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = jung_pairs()[:args.limit or None]
    config = PipelineConfig.from_mapping({"pages": {"enabled": False}, "output": {"save_intermediates": False}})
    ours = DocumentPipeline(config)
    runners = {"ours": lambda image: ours.process(image).pages[0].image, "doc_image_tool": doc_image_tool_runner()}
    if "docshadow" in methods:
        from docrestore.adapters.docshadow import DocShadowRunner
        runners["docshadow"] = DocShadowRunner(ROOT / "third_party/DocShadow-SD7K", ROOT / "data/models/docshadow/SD7K.pth")
    rows = []
    for sample, input_path, target_path in pairs:
        source, target = imread(input_path), imread(target_path)
        shadow = derive_shadow_mask(source, target)
        regions = {"full": np.ones(shadow.shape, bool), "shadow": shadow, "non_shadow": ~shadow}
        imwrite(output / "masks" / f"{sample}.png", shadow.astype(np.uint8) * 255)
        for method in methods:
            started = time.perf_counter()
            prediction = source.copy() if method == "input" else runners[method](source)
            elapsed = time.perf_counter() - started
            if args.save_all or sample in {"jung_01", "jung_06", "jung_14", "jung_20"}:
                imwrite(output / "outputs" / method / f"{sample}.png", prediction)
            metrics = reference_metrics(prediction, target, regions)
            quality = assess_quality(prediction)
            for region, values in metrics.items():
                rows.append({"sample": sample, "method": method, "region": region, **values,
                             "runtime_seconds": elapsed, "background_std": quality.background_std,
                             "illumination_variation": quality.illumination_variation,
                             "edge_energy": quality.edge_energy, "laplacian_variance": quality.laplacian_variance,
                             "noise_sigma": quality.noise_sigma, "readability_score": quality.readability_score})
            print(sample, method, f"PSNR={metrics['full']['psnr']:.3f}", f"shadow={metrics['shadow']['psnr']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_image.csv", index=False)
    numeric = ["psnr", "ssim", "rmse", "runtime_seconds", "background_std", "illumination_variation",
               "edge_energy", "laplacian_variance", "noise_sigma", "readability_score"]
    summary = frame.groupby(["method", "region"], as_index=False)[numeric].mean(numeric_only=True)
    summary.to_csv(ROOT / "results/tables/benchmark_summary.csv", index=False)
    frame.to_csv(ROOT / "results/tables/benchmark_per_image.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps({"methods": methods, "samples": len(pairs),
        "dataset": "Water-Filling Jung paired samples", "config": config.to_dict()}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
