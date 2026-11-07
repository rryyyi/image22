from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def rounded_box(ax, x, y, w, h, title, body, color, edge, title_size=12.5, body_size=10.2):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor=edge, linewidth=1.15
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color="#1B2735")
    ax.text(x + w / 2, y + h - 0.18, body, ha="center", va="top",
            fontsize=body_size, linespacing=1.55, color="#2B2B2B")


def iteration_journey():
    stages = [
        ("第一版\n估计器 + 全增强", 17.049, 0.795),
        ("关闭边缘保护\n与默认锐化", 18.350, 0.827),
        ("只保留光照修复\nguided", 18.875, 0.842),
        ("改用形态学估计\n最终默认", 18.989, 0.843),
    ]
    x = list(range(1, len(stages) + 1))
    psnr = [s[1] for s in stages]
    ssim = [s[2] for s in stages]

    fig, ax1 = plt.subplots(figsize=(12.2, 5.8))
    ax2 = ax1.twinx()
    ax1.plot(x, psnr, marker="o", markersize=8, linewidth=2.6, color="#2F6FB0", label="PSNR")
    ax2.plot(x, ssim, marker="s", markersize=7, linewidth=2.2, color="#D27A2C", label="SSIM")
    for xi, yi in zip(x, psnr):
        if xi == 1:
            ax1.annotate(f"{yi:.3f} dB / SSIM {ssim[xi-1]:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 40),
                         ha="center", fontsize=9.5, color="#1F4E79")
        else:
            ax1.annotate(f"{yi:.3f} dB", (xi, yi), textcoords="offset points", xytext=(0, 13),
                         ha="center", fontsize=9.5, color="#1F4E79")
    for xi, yi in zip(x, ssim):
        if xi == 1:
            continue
        ax2.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, -18),
                     ha="center", fontsize=9.5, color="#9C4D0A")
    ax1.set_xticks(x, [s[0] for s in stages], fontsize=10.5)
    ax1.set_ylabel("全图 PSNR / dB", fontsize=11)
    ax2.set_ylabel("全图 SSIM", fontsize=11)
    ax1.set_ylim(16.8, 19.18)
    ax2.set_ylim(0.785, 0.850)
    ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax1.set_axisbelow(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=False)
    fig.suptitle("从“功能尽量多”到“只修复真正需要修复的部分”", fontsize=15.5, y=0.98)
    ax1.set_title("Water-Filling Jung 20 对真实文档上的阶段结果；纵轴范围经过压缩以突出差异",
                  fontsize=9.8, color="#555555", pad=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "iteration_journey.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def application_scenarios():
    cards = [
        ("手机随手拍", "阴影、透视、轻微噪声\n页面检测 + 光照修复\n按需打开反光和降噪", "#EAF2FB", "#5B8FC9"),
        ("学生作业", "笔迹浅、纸张透字、对比低\n温和光照校正\n保留铅笔痕与彩色批注", "#F3EEFA", "#8067B0"),
        ("合同与票据", "褶皱、反光、红章颜色\n受控彩色增益 + 边缘保护\n全程本地处理", "#FBEFEA", "#C57457"),
        ("档案与旧文件", "褪色、划痕、纸面老化\narchive 预设 + 失败警告\n只修复可见退化", "#F3F1E7", "#9A8A53"),
        ("扫描件", "页面已平整、主要是噪声\n关闭透视、轻量降噪\n避免二次锐化", "#EAF5EF", "#4F9673"),
        ("无障碍阅读", "低对比、色弱辨认困难\n增强明暗层次\n不改变文字与印章内容", "#EAF6F7", "#4C9299"),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "现实场景不是同一个“增强”按钮", ha="center", va="top",
            fontsize=16, fontweight="bold", color="#1B2735")
    ax.text(0.5, 0.915, "我先判断页面结构和主要退化，再决定哪些模块应该打开",
            ha="center", va="top", fontsize=10.8, color="#555555")

    x_positions = [0.045, 0.355, 0.665]
    y_positions = [0.56, 0.12]
    for index, (title, body, color, edge) in enumerate(cards):
        x = x_positions[index % 3]
        y = y_positions[index // 3]
        rounded_box(ax, x, y, 0.29, 0.30, title, body, color, edge,
                    title_size=12.7, body_size=10.2)

    principle = FancyBboxPatch((0.21, 0.445), 0.58, 0.075, boxstyle="round,pad=0.01,rounding_size=0.02",
                               facecolor="#1E3A5F", edgecolor="#1E3A5F", linewidth=1)
    ax.add_patch(principle)
    ax.text(0.50, 0.4825, "共同原则：先识别退化，再控制修复强度；不确定时警告，而不是生成看似完整的内容",
            ha="center", va="center", fontsize=10.8, color="white")
    fig.tight_layout()
    fig.savefig(OUT / "application_scenarios.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    iteration_journey()
    application_scenarios()
    print(OUT / "iteration_journey.png")
    print(OUT / "application_scenarios.png")



