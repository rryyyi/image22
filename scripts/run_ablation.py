from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from docrestore import DocumentPipeline, PipelineConfig
from docrestore.io import imread, imwrite
from docrestore.quality.assess import assess_quality, derive_shadow_mask, reference_metrics
from run_benchmark import ROOT, jung_pairs

VARIANTS = {
    "full": {},
    "illumination_only": {"restoration": {"denoise": False, "deblur": False, "glare": False, "fading": False, "scratches": False}},
    "no_fading": {"restoration": {"fading": False}},
    "no_denoising": {"restoration": {"denoise": False}},
    "no_residual": {"illumination": {"residual_strength": 0.0}},
    "no_color_correction": {"illumination": {"color_strength": 0.0}},
}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=0);args=parser.parse_args()
    pairs=jung_pairs()[:args.limit or None]
    pipelines={name:DocumentPipeline(PipelineConfig.from_mapping({"pages":{"enabled":False},**changes})) for name,changes in VARIANTS.items()}
    rows=[]
    for sample,input_path,target_path in pairs:
        source,target=imread(input_path),imread(target_path);shadow=derive_shadow_mask(source,target)
        regions={"full":np.ones(shadow.shape,bool),"shadow":shadow,"non_shadow":~shadow}
        for name,pipeline in pipelines.items():
            prediction=pipeline.process(source).pages[0].image
            metrics=reference_metrics(prediction,target,regions);quality=assess_quality(prediction)
            if sample in {"jung_01","jung_14"}: imwrite(ROOT/"results/ablation_outputs"/name/f"{sample}.png",prediction)
            for region,values in metrics.items():
                rows.append({"sample":sample,"variant":name,"region":region,**values,"edge_energy":quality.edge_energy,
                             "background_std":quality.background_std,"illumination_variation":quality.illumination_variation})
        print("ablation",sample,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/"results/tables/ablation_per_image.csv",index=False)
    summary=frame.groupby(["variant","region"],as_index=False)[["psnr","ssim","rmse","edge_energy","background_std","illumination_variation"]].mean()
    summary.to_csv(ROOT/"results/tables/ablation_summary.csv",index=False)
    full=summary[summary.region=="full"].set_index("variant").loc[list(VARIANTS)]
    fig,ax=plt.subplots(figsize=(10,5));full["psnr"].plot(kind="bar",ax=ax,color="#4472C4");ax.set_ylabel("PSNR (dB)");ax.set_title("Ablation on 20 real paired documents");ax.tick_params(axis='x',rotation=30);fig.tight_layout()
    fig.savefig(ROOT/"results/figures/ablation_psnr.png",dpi=170)


if __name__=="__main__":main()
