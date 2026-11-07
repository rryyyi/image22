from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    summary=pd.read_csv(ROOT/"results/tables/benchmark_summary.csv")
    full=summary[summary.region=="full"].set_index("method").loc[["input","doc_image_tool","ours","docshadow"]]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    full[["psnr"]].plot(kind="bar",ax=axes[0],legend=False,color="#4472C4");axes[0].set_ylabel("PSNR (dB)");axes[0].set_title("Full-page PSNR")
    full[["ssim"]].plot(kind="bar",ax=axes[1],legend=False,color="#ED7D31");axes[1].set_ylim(.75,.88);axes[1].set_title("Full-page SSIM")
    for ax in axes:ax.tick_params(axis='x',rotation=0)
    fig.tight_layout();fig.savefig(ROOT/"results/figures/benchmark_summary.png",dpi=170)
    summary=pd.read_csv(ROOT/"results/tables/ablation_summary.csv")
    full=summary[summary.region=="full"].set_index("variant")
    fig,ax=plt.subplots(figsize=(9,4.6));full[["psnr"]].plot(kind="bar",ax=ax,legend=False,color="#70AD47");ax.set_ylabel("PSNR (dB)");ax.set_title("Final ablation");ax.tick_params(axis='x',rotation=30);fig.tight_layout()
    fig.savefig(ROOT/"results/figures/final_ablation.png",dpi=170)


if __name__=="__main__":main()
