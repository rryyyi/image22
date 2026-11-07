from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Token

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, title, detail="", color="#EAF2FB", edge="#5B8FC9"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                           facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h*0.63, title, ha="center", va="center", fontsize=11.7,
            fontweight="bold", color="#1B2B3C")
    if detail:
        ax.text(x + w/2, y + h*0.28, detail, ha="center", va="center", fontsize=9.2,
                color="#44515F", linespacing=1.35)


def arrow(ax, start, end, color="#4D78A6"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.35, connectionstyle="arc3,rad=0.02"))


def technical_model():
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.98, "文档光照修复的技术建模", ha="center", va="top",
            fontsize=16.5, fontweight="bold", color="#182A3D")
    ax.text(0.5, 0.925, "把观测图看成反射率与照明场的乘积，再分别估计、限幅和混合",
            ha="center", va="top", fontsize=10.2, color="#5E6872")

    formula_box = FancyBboxPatch((0.16, 0.785), 0.68, 0.095, boxstyle="round,pad=0.012,rounding_size=0.018",
                                 facecolor="#F8F1E5", edgecolor="#C19A5B", linewidth=1.1)
    ax.add_patch(formula_box)
    ax.text(0.5, 0.847, "I(x,y) = R(x,y) × L(x,y)", ha="center", va="center",
            fontsize=15.5, family="DejaVu Sans", color="#4C3A21")
    ax.text(0.5, 0.805, "I：观测图    R：纸面与油墨反射率    L：低频照明场",
            ha="center", va="center", fontsize=9.2, color="#5A4A32")

    boxes_top = [
        (0.025, "输入 I", "手机照片或扫描图", "#EEF3F8", "#7288A0"),
        (0.275, "页面与几何", "页面检测、透视校正", "#EAF2FB", "#5B8FC9"),
        (0.525, "纸面掩膜", "分离大面积深色插图", "#F3EEFA", "#8067B0"),
        (0.775, "估计 L", "形态学闭运算 + 平滑", "#EAF5EF", "#4F9673"),
    ]
    boxes_bottom = [
        (0.025, "计算增益 g", "目标分位值与增益限幅", "#FBEFEA", "#C57457"),
        (0.275, "亮度与颜色", "YCrCb 亮度 + 低饱和彩色增益", "#FFF4DD", "#B58A38"),
        (0.525, "残余补偿", "低频残差有限幅度修正", "#EAF6F7", "#4C9299"),
        (0.775, "输出 R", "增强图 + 质量警告", "#F3F1E7", "#9A8A53"),
    ]
    for x, title, detail, color, edge in boxes_top:
        add_box(ax, x, 0.53, 0.19, 0.17, title, detail, color, edge)
    for x, title, detail, color, edge in boxes_bottom:
        add_box(ax, x, 0.18, 0.19, 0.17, title, detail, color, edge)
    for x in (0.215, 0.465, 0.715):
        arrow(ax, (x, 0.615), (x + 0.06, 0.615))
    arrow(ax, (0.87, 0.53), (0.12, 0.35), color="#9A6B52")
    for x in (0.215, 0.465, 0.715):
        arrow(ax, (x, 0.265), (x + 0.06, 0.265))

    ax.text(0.5, 0.095, "g = clip(L_target / max(L, 0.025), 1/3.2, 3.2)    g' = 1 + (g − 1) × 0.92",
            ha="center", va="center", fontsize=10.5, family="DejaVu Sans", color="#2F4052")
    ax.text(0.5, 0.045, "默认参数：morph_kernel=41，target_percentile=92，strength=0.92，color_strength=0.30",
            ha="center", va="center", fontsize=9.4, color="#65717E")
    fig.tight_layout()
    fig.savefig(FIG / "technical_modeling.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def intermediate_process():
    base = ROOT / "results" / "examples" / "real_shadow"
    source = ROOT / "third_party" / "Doc-Image-Tool" / "test" / "阴影.jpg"
    restored = base / "page_01.png"
    illumination = base / "intermediates" / "page_01_illumination.png"
    gain = base / "intermediates" / "page_01_gain.png"

    original = np.asarray(Image.open(source).convert("RGB"))
    output = np.asarray(Image.open(restored).convert("RGB"))
    illum = np.asarray(Image.open(illumination).convert("L"), dtype=np.float32) / 255.0
    gain_map = np.asarray(Image.open(gain).convert("L"), dtype=np.float32) / 255.0
    if output.shape[:2] != original.shape[:2]:
        output = np.asarray(Image.fromarray(output).resize((original.shape[1], original.shape[0]), Image.Resampling.LANCZOS))
    if illum.shape != original.shape[:2]:
        illum = np.asarray(Image.fromarray((illum * 255).astype(np.uint8)).resize((original.shape[1], original.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    if gain_map.shape != original.shape[:2]:
        gain_map = np.asarray(Image.fromarray((gain_map * 255).astype(np.uint8)).resize((original.shape[1], original.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.0))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("原始输入：阴影、褶皱与偏色", fontsize=12.5, pad=9)
    axes[0, 1].imshow(illum)
    im1 = axes[0, 1].imshow(illum, cmap="inferno")
    axes[0, 1].set_title("估计照明场 L：亮区表示需要较少增益", fontsize=12.5, pad=9)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.02, label="相对照明强度")

    im2 = axes[1, 0].imshow(gain_map, cmap="viridis")
    axes[1, 0].set_title("增益图：暗区获得较大增益", fontsize=12.5, pad=9)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.02, label="相对增益")
    axes[1, 1].imshow(output)
    axes[1, 1].set_title("修复结果：背景更均匀，内容保留", fontsize=12.5, pad=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("真实中文合同样例的处理过程可视化", fontsize=16.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "intermediate_process.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


TOKEN_COLORS = {
    Keyword: "#C586C0",
    Keyword.Constant: "#569CD6",
    Keyword.Namespace: "#C586C0",
    Name.Function: "#DCDCAA",
    Name.Class: "#4EC9B0",
    Name.Builtin: "#4EC9B0",
    Name.Namespace: "#4EC9B0",
    Name.Decorator: "#DCDCAA",
    String: "#CE9178",
    Number: "#B5CEA8",
    Comment: "#6A9955",
    Operator: "#D4D4D4",
    Punctuation: "#D4D4D4",
}


def token_color(token_type):
    while token_type is not Token:
        if token_type in TOKEN_COLORS:
            return TOKEN_COLORS[token_type]
        token_type = token_type.parent
    return "#D4D4D4"


def render_code(path: Path, start: int, end: int, title: str, output: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = [(number, lines[number - 1]) for number in range(start, end + 1)]
    code_font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 27)
    num_font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 23)
    header_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    sub_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)

    char_width = code_font.getlength("M")
    max_chars = max(len(line.expandtabs(4)) for _, line in selected)
    gutter = 92
    left_pad = 26
    right_pad = 35
    top_header = 82
    line_height = 39
    bottom_pad = 28
    width = int(gutter + left_pad + max_chars * char_width + right_pad)
    height = int(top_header + bottom_pad + len(selected) * line_height)

    image = Image.new("RGB", (width, height), "#1E1E1E")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, top_header), fill="#252526")
    draw.rectangle((0, 0, width, 3), fill="#007ACC")
    draw.text((24, 15), title, font=header_font, fill="#F0F0F0")
    draw.text((24, 50), f"src/docrestore/{path.parent.name}/{path.name}" if path.parent.name != "docrestore" else f"src/docrestore/{path.name}",
              font=sub_font, fill="#A9B7C6")
    draw.text((width - 260, 15), f"第 {start}-{end} 行", font=sub_font, fill="#A9B7C6")

    y = top_header + 18
    for line_number, line in selected:
        draw.text((18, y + 2), str(line_number), font=num_font, fill="#858585")
        x = gutter + left_pad
        for token_type, token_text in lex(line, PythonLexer()):
            if token_text == "\n":
                continue
            draw.text((x, y), token_text, font=code_font, fill=token_color(token_type))
            x += code_font.getlength(token_text)
        y += line_height
    image.save(output, optimize=True)


def code_figures():
    pipeline = ROOT / "src" / "docrestore" / "pipeline.py"
    restore = ROOT / "src" / "docrestore" / "illumination" / "restore.py"
    render_code(pipeline, 64, 82, "DocumentPipeline.process：主流水线调度", FIG / "core_code_pipeline.png")
    render_code(restore, 28, 51, "restore_illumination：增益、亮度与彩色保真", FIG / "core_code_illumination.png")


if __name__ == "__main__":
    technical_model()
    intermediate_process()
    code_figures()
    for name in ("technical_modeling.png", "intermediate_process.png", "core_code_pipeline.png", "core_code_illumination.png"):
        print(FIG / name)

