from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docrestore.io import imread

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="jung_14")
    parser.add_argument("--methods", default="input,doc_image_tool,ours,docshadow")
    parser.add_argument("--benchmark", default="results/benchmark")
    args = parser.parse_args()
    root = Path(args.benchmark)
    target = next((ROOT / "third_party/Water-Filling/GroundTruth").glob(f"rect_{int(args.sample.split('_')[-1])}.*"))
    methods = args.methods.split(",")
    images = [(method.replace("_", " ").title(), imread(root / "outputs" / method / f"{args.sample}.png")) for method in methods]
    images.append(("Ground Truth", imread(target)))
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 7))
    for axis, (title, image) in zip(axes, images):
        axis.imshow(image[:, :, ::-1]); axis.set_title(title); axis.axis("off")
    fig.suptitle(f"Document restoration comparison: {args.sample}")
    fig.tight_layout()
    output = ROOT / "results/figures/samples" / f"{args.sample}_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    print(output)


if __name__ == "__main__": main()
