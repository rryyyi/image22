from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

FONT_CN = "C:/Windows/Fonts/msyh.ttc"
FONT_CN_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


def make_benchmark_chart():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = ["原始输入", "Doc-Image-Tool", "DocShadow", "自主方法"]
    psnr = [12.932, 15.918, 17.300, 18.989]
    ssim = [0.792, 0.810, 0.834, 0.843]
    colors = ["#A9B4C2", "#A9B4C2", "#A9B4C2", "#2F6FB0"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#FBFCFE")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C9D2DC")
        ax.spines["bottom"].set_color("#C9D2DC")
        ax.tick_params(colors="#33404F", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, color="#DCE4EC", alpha=0.9)
        ax.set_axisbelow(True)

    bars = axes[0].bar(labels, psnr, color=colors, width=0.58, edgecolor="white", linewidth=0.8)
    axes[0].set_title("全图 PSNR（越高越好）", fontsize=15, color="#1D3B5F", pad=12)
    axes[0].set_ylabel("PSNR / dB", fontsize=11.5, color="#44515F")
    axes[0].set_ylim(0, 21.5)
    for bar, value in zip(bars, psnr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=11, color="#23384E")

    bars = axes[1].bar(labels, ssim, color=["#E3A25B" if i < 3 else "#C9693B" for i in range(4)],
                       width=0.58, edgecolor="white", linewidth=0.8)
    axes[1].set_title("全图 SSIM（越高越好）", fontsize=15, color="#1D3B5F", pad=12)
    axes[1].set_ylabel("SSIM", fontsize=11.5, color="#44515F")
    axes[1].set_ylim(0.75, 0.86)
    for bar, value in zip(bars, ssim):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=11, color="#23384E")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=10.8)
    fig.text(0.5, 0.005, "数据：Water-Filling Jung 20 对真实配对文档；同一输入与同一阴影掩膜",
             ha="center", fontsize=9.5, color="#77828F")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "benchmark_summary.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _token_color(ttype) -> str:
    if ttype in Comment:
        return "#6A8A9A"
    if ttype in Keyword:
        return "#C792EA"
    if ttype in Name.Function or ttype in Name.Class:
        return "#82AAFF"
    if ttype in String:
        return "#C3E88D"
    if ttype in Number:
        return "#F78C6C"
    if ttype in Operator:
        return "#89DDFF"
    return "#D6DEEB"


def _code_lines(code: str):
    lines: list[list[tuple[str, str]]] = [[]]
    for ttype, value in lex(code, PythonLexer()):
        color = _token_color(ttype)
        parts = value.split("\n")
        for index, part in enumerate(parts):
            if part:
                lines[-1].append((part, color))
            if index < len(parts) - 1:
                lines.append([])
    if lines and not lines[-1]:
        lines.pop()
    return lines


def make_code_panel(code: str, title: str, subtitle: str, output: str, font_size: int = 34):
    font = ImageFont.truetype(FONT_MONO, font_size)
    title_font = ImageFont.truetype(FONT_CN_BOLD, 30)
    subtitle_font = ImageFont.truetype(FONT_CN, 19)
    small_font = ImageFont.truetype(FONT_MONO, 18)

    lines = _code_lines(code)
    line_height = int(font_size * 1.34)
    code_width = max(font.getlength("".join(text for text, _ in line)) for line in lines)
    width = max(1500, int(150 + code_width + 55))
    header_height = 84
    height = header_height + line_height * len(lines) + 38

    image = Image.new("RGB", (width, height), "#0F1720")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, header_height), fill="#16212E")
    draw.rectangle((0, header_height - 2, width, header_height), fill="#2C4056")
    draw.text((34, 16), title, font=title_font, fill="#E7EEF7")
    draw.text((34, 53), subtitle, font=subtitle_font, fill="#7F9AB3")
    right = f"{len(lines)} 行核心代码"
    draw.text((width - 34 - draw.textlength(right, font=subtitle_font), 55), right,
              font=subtitle_font, fill="#7F9AB3")

    y = header_height + 16
    for index, line in enumerate(lines, start=1):
        number = f"{index:02d}"
        draw.text((28, y + 3), number, font=small_font, fill="#516579")
        x = 94
        for text, color in line:
            draw.text((x, y), text, font=font, fill=color)
            x += font.getlength(text)
        y += line_height
    image.save(ASSETS / output, optimize=True)


PIPELINE_CODE = '''def process(self, image: np.ndarray) -> PipelineResult:
    geometries = detect_pages(image, ...)
    results: list[PageResult] = []
    for index, geometry in enumerate(geometries):
        page = rectify_page(image, geometry, perspective_enabled,
                            white_background)
        before = assess_quality(page)
        work = page.copy()
        if restoration["glare"]:
            work, glare_mask, _ = repair_glare(work, glare_threshold,
                                               max_inpaint_ratio)
        if self.config.illumination["enabled"]:
            work, illumination_data = restore_illumination(work, ...)
        if restoration["denoise"]:
            work = denoise_document(work, before.noise_sigma, ...)
        if restoration["fading"]:
            work = restore_fading(work, fade_strength, clahe_clip)
        if restoration["scratches"]:
            work, scratch_mask = repair_scratches(work, scratch_strength)
        if restoration["deblur"]:
            work = conservative_sharpen(work, sharpen_amount, radius)
        after = assess_quality(work)
        warnings = detect_failures(before, after, geometry, ...)'''

ILLUMINATION_CODE = '''field = estimate.field
target = max(0.08, float(np.percentile(field, 92)))
maximum = float(config.get("max_gain", 3.2))
gain = np.clip(target / np.maximum(field, 0.025),
               1.0 / maximum, maximum)
strength = float(config.get("strength", 0.92))
gain = 1.0 + (gain - 1.0) * strength
gain = cv2.GaussianBlur(gain, (0, 0), 1.2)
ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0] * gain, 0, 1)
luminance_result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
color_field = estimate_color_field(image, 1200, 41)
color_target = np.percentile(color_field.reshape(-1, 3), 92, axis=0)
color_gain = np.clip(color_target / np.maximum(color_field, 0.025),
                     1.0 / maximum, maximum)
paper = np.clip((0.42 - hsv[:, :, 1]) / 0.25, 0, 1) * \\
        np.clip((ycrcb[:, :, 0] - 0.18) / 0.25, 0, 1)
alpha = (paper * 0.30)[:, :, None]
result = luminance_result * (1 - alpha) + color_result * alpha'''


def main():
    make_benchmark_chart()
    make_code_panel(PIPELINE_CODE,
                    "DocumentPipeline.process：页面检测与分模块调度",
                    "src/docrestore/pipeline.py · 主流水线节选",
                    "core_code_pipeline.png")
    make_code_panel(ILLUMINATION_CODE,
                    "restore_illumination：增益、亮度与彩色保真",
                    "src/docrestore/illumination/restore.py · 核心实现节选",
                    "core_code_illumination.png")
    print("assets updated")


if __name__ == "__main__":
    main()
