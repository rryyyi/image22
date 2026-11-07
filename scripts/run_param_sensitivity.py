from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docrestore import DocumentPipeline,PipelineConfig
from docrestore.io import imread
from docrestore.quality.assess import derive_shadow_mask,reference_metrics
from run_benchmark import ROOT,jung_pairs

SWEEPS={"guided_radius":[8,24,48],"morph_kernel":[21,41,81],"max_gain":[1.8,2.6,3.2]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=0);args=parser.parse_args();pairs=jung_pairs()[:args.limit or None]
    loaded=[(sample,imread(ip),imread(tp)) for sample,ip,tp in pairs];rows=[]
    for parameter,values in SWEEPS.items():
        for value in values:
            pipeline=DocumentPipeline(PipelineConfig.from_mapping({"pages":{"enabled":False},"illumination":{parameter:value}}))
            for sample,source,target in loaded:
                prediction=pipeline.process(source).pages[0].image;shadow=derive_shadow_mask(source,target)
                metrics=reference_metrics(prediction,target,{"full":np.ones(shadow.shape,bool),"shadow":shadow,"non_shadow":~shadow})
                for region,item in metrics.items():rows.append({"sample":sample,"parameter":parameter,"value":value,"region":region,**item})
            print("sensitivity",parameter,value,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/"results/tables/sensitivity_per_image.csv",index=False)
    summary=frame.groupby(["parameter","value","region"],as_index=False)[["psnr","ssim","rmse"]].mean();summary.to_csv(ROOT/"results/tables/sensitivity_summary.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(parameter,values) in zip(axes,SWEEPS.items()):
        data=summary[(summary.parameter==parameter)&(summary.region=="full")].set_index("value").loc[values]
        ax.plot(values,data.psnr,marker="o",label="PSNR");ax.set_title(parameter);ax.set_xlabel("value");ax.set_ylabel("PSNR (dB)");ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/"results/figures/parameter_sensitivity.png",dpi=170)


if __name__=="__main__":main()
