from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, color="#dbe7ff", edge="#4472C4", fontsize=11):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            linespacing=1.35, color="#1F1F1F")


def arrow(ax, xy, xytext, color="#4472C4"):
    ax.add_patch(FancyArrowPatch(xy, xytext, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.2, connectionstyle="arc3,rad=0.02"))


def pipeline():
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .02, .39, .15, .22, "现实输入\n手机拍照\n扫描仪\n摄像头", color="#F2F2F2", edge="#8497B0")
    box(ax, .21, .60, .18, .22, "页面理解\n纸面候选\n多页拆分\n背景分离")
    box(ax, .21, .19, .18, .22, "几何修复\n四边形排序\n透视映射\n异常回退", color="#E8F1FB")
    box(ax, .44, .60, .18, .22, "光照场估计\n形态学\n引导滤波\nRetinex等")
    box(ax, .44, .19, .18, .22, "反射率恢复\n亮度增益\n低饱和彩色增益\n边界控制", color="#E8F1FB")
    box(ax, .67, .60, .15, .22, "可选增强\n反光\n噪声/模糊\n褪色/划痕", color="#FFF2CC", edge="#BF9000")
    box(ax, .67, .19, .15, .22, "质量评估\n背景/边缘\n噪声/反光\n可读性", color="#E2F0D9", edge="#548235")
    box(ax, .86, .39, .12, .22, "输出\n增强图像\n质量报告\n失败警告", color="#FBE5D6", edge="#C55A11")
    arrow(ax, (.17, .50), (.21, .50))
    arrow(ax, (.39, .68), (.44, .70)); arrow(ax, (.39, .34), (.44, .30))
    arrow(ax, (.62, .68), (.67, .70)); arrow(ax, (.62, .34), (.67, .30))
    arrow(ax, (.74, .60), (.80, .41)); arrow(ax, (.82, .19), (.88, .39))
    fig.suptitle("文档修复流水线：先理解页面，再估计光照，最后可控增强", fontsize=15, y=.97)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_diagram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analysis():
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .02, .39, .16, .24, "数据源\n真实配对文档\n合成退化\n真实中文样例", color="#F2F2F2", edge="#8497B0")
    box(ax, .22, .62, .17, .22, "配对数据\n阴影输入\n无阴影GT\n尺寸校验", color="#E8F1FB")
    box(ax, .22, .17, .17, .22, "阴影掩膜\n低频背景差异\n形态学清理\n区域比例记录", color="#FFF2CC", edge="#BF9000")
    box(ax, .43, .39, .19, .24, "统一实验\n同一方法集\n同一指标\n同一图像尺寸")
    box(ax, .66, .62, .15, .22, "参考指标\nPSNR\nSSIM\nRMSE", color="#E2F0D9", edge="#548235")
    box(ax, .66, .17, .15, .22, "无参考指标\n光照残留\n背景均匀\n边缘能量", color="#E2F0D9", edge="#548235")
    box(ax, .84, .39, .14, .24, "结论\n全图/区域\n失败案例\n主观复核", color="#FBE5D6", edge="#C55A11")
    arrow(ax, (.18, .50), (.22, .50))
    arrow(ax, (.30, .62), (.43, .56)); arrow(ax, (.30, .39), (.43, .44))
    arrow(ax, (.62, .58), (.66, .68)); arrow(ax, (.62, .42), (.66, .32))
    arrow(ax, (.81, .62), (.86, .56)); arrow(ax, (.81, .36), (.86, .45))
    fig.suptitle("评估流程：先固定数据与掩膜，再分层比较真实效果", fontsize=15, y=.97)
    fig.tight_layout()
    fig.savefig(OUT / "analysis_workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline(); analysis(); print(OUT / "pipeline_diagram.png"); print(OUT / "analysis_workflow.png")
