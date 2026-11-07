from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def contract_comparison():
    source = ROOT / "third_party" / "Doc-Image-Tool" / "test" / "阴影.jpg"
    restored = ROOT / "results" / "examples" / "real_shadow" / "page_01.png"
    original_image = Image.open(source).convert("RGB")
    restored_image = Image.open(restored).convert("RGB")
    if restored_image.size != original_image.size:
        restored_image = restored_image.resize(original_image.size, Image.Resampling.LANCZOS)

    width, height = original_image.size
    crop_box = (int(width * 0.04), int(height * 0.01), int(width * 0.66), int(height * 0.43))
    panels = [
        (original_image, "原始输入：整体阴影与偏色"),
        (restored_image, "修复结果：光照更均匀"),
        (original_image.crop(crop_box), "原始输入局部：文字、手写与勾选项"),
        (restored_image.crop(crop_box), "修复结果局部：内容仍然保留"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.5), gridspec_kw={"height_ratios": [1.22, 1.0]})
    for ax, (image, title) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12.5, pad=10, color="#23364A")
        ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.955, bottom=0.035, wspace=0.06, hspace=0.12)
    fig.savefig(FIG / "contract_comparison_expanded.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ablation_delta():
    labels = [
        "关闭残余光照校正",
        "关闭彩色增益",
        "关闭去噪（默认未启用）",
        "关闭褪色恢复（默认未启用）",
        "仅保留光照修复",
    ]
    values = [-0.4557, -0.0286, 0.0, 0.0, 0.0]
    colors = ["#C75B4B" if value < 0 else "#AEB8C2" for value in values]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.8, 5.5))
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.axvline(0, color="#26384A", linewidth=1.0)
    ax.set_yticks(y, labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.52, 0.035)
    ax.set_xlabel("相对完整配置的全图 PSNR 变化 / dB", fontsize=10.5)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        if value < 0:
            ax.text(value - 0.012, bar.get_y() + bar.get_height()/2, f"{value:.3f} dB",
                    va="center", ha="right", fontsize=9.3, color="#7E2E22")
        else:
            ax.text(0.006, bar.get_y() + bar.get_height()/2, "0.000 dB",
                    va="center", ha="left", fontsize=9.0, color="#5C6670")
    ax.set_title("关闭不同模块后，结果相对完整配置下降了多少", fontsize=14.5, pad=13)
    ax.text(0.0, 1.01, "负值越大，说明该部分对当前数据越重要；0 表示该模块在默认配置中本来就没有启用",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color="#59636D")
    fig.tight_layout()
    fig.savefig(FIG / "final_ablation_delta.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    contract_comparison()
    ablation_delta()
    print(FIG / "contract_comparison_expanded.png")
    print(FIG / "final_ablation_delta.png")



