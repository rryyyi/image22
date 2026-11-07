from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from docrestore.io import imread
from run_benchmark import ROOT, jung_pairs


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--benchmark",default="results/benchmark_final");args=parser.parse_args()
    frame=pd.read_csv(ROOT/"results/tables/benchmark_per_image.csv")
    full=frame[frame.region=="full"].pivot(index="sample",columns="method",values="psnr")
    full["gain_over_tool"]=full.ours-full.doc_image_tool
    full["gain_over_nn"]=full.ours-full.docshadow
    shadow_ratio=frame[(frame.method=="input")&(frame.region=="shadow")].set_index("sample").pixels
    non_ratio=frame[(frame.method=="input")&(frame.region=="non_shadow")].set_index("sample").pixels
    full["shadow_ratio"]=shadow_ratio/(shadow_ratio+non_ratio)
    ranked=full.reset_index().sort_values("gain_over_nn")
    ranked.to_csv(ROOT/"results/tables/failure_ranking.csv",index=False)
    worst=ranked.head(3)["sample"].tolist();best=ranked.tail(1)["sample"].tolist()
    paths={sample:pair for sample,_,_ in jung_pairs() for pair in [next((p for s,p,_ in jung_pairs() if s==sample),None)]}
    targets={sample:target for sample,_,target in jung_pairs()}
    fig,axes=plt.subplots(len(worst),5,figsize=(17,4.2*len(worst)),squeeze=False)
    for row,sample in enumerate(worst):
        root=Path(args.benchmark)
        images=[("Input",imread(root/"outputs/input"/f"{sample}.png")),("Doc Image Tool",imread(root/"outputs/doc_image_tool"/f"{sample}.png")),
                ("Ours",imread(root/"outputs/ours"/f"{sample}.png")),("DocShadow",imread(root/"outputs/docshadow"/f"{sample}.png")),
                ("GT",imread(targets[sample]))]
        for col,(title,image) in enumerate(images):
            axes[row,col].imshow(image[:,:,::-1]);axes[row,col].set_title(title);axes[row,col].axis("off")
        axes[row,0].set_ylabel(sample)
    fig.suptitle("Worst cases for the local pipeline versus DocShadow")
    fig.tight_layout();fig.savefig(ROOT/"results/figures/failure_cases.png",dpi=170,bbox_inches="tight")
    print("worst",worst,"best",best)


if __name__=="__main__":main()
