from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from docrestore import DocumentPipeline,PipelineConfig
from docrestore.io import imread
from docrestore.quality.assess import assess_quality,derive_shadow_mask,reference_metrics
from run_benchmark import ROOT,jung_pairs

METHODS=["morphological","retinex","guided","contrast_guided","water_filling"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=0);args=parser.parse_args()
    restoration={"denoise":False,"deblur":False,"glare":False,"fading":False,"scratches":False}
    pipelines={m:DocumentPipeline(PipelineConfig.from_mapping({"pages":{"enabled":False},"illumination":{"estimator":m,"water_iterations":300},"restoration":restoration})) for m in METHODS}
    rows=[];fields={}
    for sample,input_path,target_path in jung_pairs()[:args.limit or None]:
        source,target=imread(input_path),imread(target_path);shadow=derive_shadow_mask(source,target);regions={"full":np.ones(shadow.shape,bool),"shadow":shadow,"non_shadow":~shadow}
        for method,pipeline in pipelines.items():
            start=time.perf_counter();result=pipeline.process(source);elapsed=time.perf_counter()-start;page=result.pages[0]
            metrics=reference_metrics(page.image,target,regions);q=assess_quality(page.image)
            if sample=="jung_14":fields[method]=page.intermediates["illumination"]
            for region,values in metrics.items():rows.append({"sample":sample,"method":method,"region":region,**values,"runtime_seconds":elapsed,"edge_energy":q.edge_energy,"background_std":q.background_std,"illumination_variation":q.illumination_variation})
        print("estimators",sample,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/"results/tables/estimator_per_image.csv",index=False)
    summary=frame.groupby(["method","region"],as_index=False)[["psnr","ssim","rmse","runtime_seconds","edge_energy","background_std","illumination_variation"]].mean();summary.to_csv(ROOT/"results/tables/estimator_summary.csv",index=False)
    if fields:
        fig,axes=plt.subplots(2,3,figsize=(12,9));axes=axes.ravel()
        for ax,(method,field) in zip(axes,fields.items()):im=ax.imshow(field,cmap="gray",vmin=0,vmax=1);ax.set_title(method);ax.axis("off")
        axes[-1].axis("off");fig.suptitle("Estimated illumination fields on jung_14");fig.tight_layout();fig.savefig(ROOT/"results/figures/illumination_comparison.png",dpi=170)


if __name__=="__main__":main()
