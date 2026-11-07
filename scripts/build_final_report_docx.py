from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_expanded_report_docx import (  # reuse tested low-level helpers
    add_field,
    add_inline_runs,
    prevent_row_split,
    repeat_table_header,
    set_cell_margins,
    set_cell_shading,
    set_run_font,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "final_report.md"
OUTPUT = ROOT / "docs" / "文档图像修复实验报告.docx"

BODY_FONT = "宋体"
HEADING_FONT = "微软雅黑"
TEXT_COLOR = RGBColor(0x22, 0x22, 0x22)
HEADING_COLOR = RGBColor(0x1D, 0x3B, 0x5F)
SUBHEADING_COLOR = RGBColor(0x2C, 0x58, 0x83)
MUTED_COLOR = RGBColor(0x66, 0x66, 0x66)
ACCENT_COLOR = RGBColor(0x2F, 0x6F, 0xB0)


def style_font(style, name: str, size: float, bold: bool | None = None, color=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rfonts.set(qn(attr), name)
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


def configure_document(document: Document):
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.95)
    section.bottom_margin = Cm(1.85)
    section.left_margin = Cm(2.05)
    section.right_margin = Cm(2.05)
    section.header_distance = Cm(0.85)
    section.footer_distance = Cm(0.8)
    section.different_first_page_header_footer = True

    normal = document.styles["Normal"]
    style_font(normal, BODY_FONT, 10.5, color=TEXT_COLOR)
    normal.paragraph_format.line_spacing = 1.38
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    title_style = document.styles["Title"]
    style_font(title_style, HEADING_FONT, 24, bold=True, color=RGBColor(0x12, 0x12, 0x12))
    title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_style.paragraph_format.space_after = Pt(6)
    title_pPr = title_style.element.get_or_add_pPr()
    for border in title_pPr.findall(qn("w:pBdr")):
        title_pPr.remove(border)

    for name, size, color, before, after in (
        ("Heading 1", 15.5, HEADING_COLOR, 13, 5),
        ("Heading 2", 12.2, SUBHEADING_COLOR, 10, 4),
        ("Heading 3", 11.0, SUBHEADING_COLOR, 8, 3),
    ):
        style = document.styles[name]
        style_font(style, HEADING_FONT, size, bold=True, color=color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        style.paragraph_format.line_spacing = 1.15

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(header.add_run("文档图像修复实验报告"), name=BODY_FONT, size=8.4, color=RGBColor(0x7A, 0x7A, 0x7A))

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(footer.add_run("第 "), name=BODY_FONT, size=8.6, color=MUTED_COLOR)
    add_field(footer, "PAGE")
    set_run_font(footer.add_run(" 页 / 共 "), name=BODY_FONT, size=8.6, color=MUTED_COLOR)
    add_field(footer, "NUMPAGES")
    set_run_font(footer.add_run(" 页"), name=BODY_FONT, size=8.6, color=MUTED_COLOR)

    props = document.core_properties
    props.title = "文档图像修复实验报告"
    props.subject = "文档图像修复、光照估计与质量评估"
    props.author = ""
    props.keywords = "文档图像修复, 光照估计, 形态学背景, 透视校正, 质量评估"
    props.comments = "数字图像处理实验报告"


def add_cover(document: Document):
    for _ in range(2):
        document.add_paragraph().paragraph_format.space_after = Pt(0)

    kicker = document.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(24)
    set_run_font(kicker.add_run("数字图像处理实验"), name=HEADING_FONT, size=12, bold=True, color=ACCENT_COLOR)

    title = document.add_paragraph("文档图像修复实验报告", style="Title")
    title.paragraph_format.space_before = Pt(0)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(18)
    set_run_font(subtitle.add_run("从一张拍歪的合同说起：页面理解、光照恢复与质量评估"),
                 name=HEADING_FONT, size=13, color=RGBColor(0x45, 0x50, 0x5C))

    rule = document.add_paragraph()
    rule.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rule.paragraph_format.space_after = Pt(30)
    pPr = rule._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "B7C6D8")
    borders.append(bottom)
    pPr.append(borders)

    for _ in range(2):
        p = document.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    for key, value in (("课程", "数字图像处理"), ("姓名", "________________"),
                       ("学号", "________________"), ("日期", "2026 年 11 月")):
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(9)
        set_run_font(p.add_run(f"{key}："), name=BODY_FONT, size=11, color=RGBColor(0x55, 0x55, 0x55))
        set_run_font(p.add_run(value), name=BODY_FONT, size=11, color=RGBColor(0x33, 0x33, 0x33))

    for _ in range(4):
        p = document.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(note.add_run("实验数据：Water-Filling Jung 20 对真实文档、合成退化样例与真实中文合同"),
                 name=BODY_FONT, size=9.2, color=RGBColor(0x77, 0x77, 0x77))

    p = document.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)


def add_body_paragraph(document: Document, text: str):
    p = document.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.38
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.widow_control = True
    add_inline_runs(p, text, base_size=10.5, base_color=TEXT_COLOR)
    return p


def add_keywords(document: Document, text: str):
    p = document.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_inline_runs(p, text, base_size=10.0, base_color=RGBColor(0x3C, 0x3C, 0x3C))


def add_figure(document: Document, caption: str, image_path: Path):
    if not caption:
        # LaTeX equation image: fixed display width, no caption paragraph.
        eq_w_px, eq_h_px = Image.open(image_path).size
        eq_width = Inches(5.55) if eq_w_px / eq_h_px > 14 else Inches(5.0)
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_together = True
        p.add_run().add_picture(str(image_path), width=eq_width)
        return

    width_px, height_px = Image.open(image_path).size
    max_width = Inches(6.15)
    max_height = Cm(7.4) if image_path.name.startswith("core_code") else Cm(8.8)
    width = max_width
    height = int(width * height_px / width_px)
    if height > max_height:
        height = max_height
        width = int(height * width_px / height_px)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    inline_shape = p.add_run().add_picture(str(image_path), width=width, height=height)
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", caption)
    doc_pr.set("title", caption)

    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.space_after = Pt(7)
    cap.paragraph_format.keep_together = True
    set_run_font(cap.add_run(caption), name=BODY_FONT, size=8.8, color=RGBColor(0x5E, 0x5E, 0x5E))


def parse_table(lines: list[str], start: int):
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        row = [item.strip() for item in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", item) for item in row):
            rows.append(row)
        index += 1
    return rows, index


def add_table(document: Document, rows: list[list[str]], table_number: int):
    if not rows:
        return
    table_widths = {1: [1.05, 0.94, 0.90, 0.92, 1.02, 0.94, 0.90],
                    2: [1.10, 1.36, 1.36, 1.36, 1.34]}
    widths = table_widths.get(table_number)
    caption_text = ("表 1 四种方法在 Jung 20 对真实文档上的主要指标"
                    if table_number == 1 else
                    "表 2 三个困难样例的 PSNR 对比")
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    set_run_font(cap.add_run(caption_text), name=HEADING_FONT, size=9.0, bold=True, color=RGBColor(0x3D, 0x4D, 0x60))

    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)

    for row_idx, row in enumerate(rows):
        prevent_row_split(table.rows[row_idx])
        if row_idx == 0:
            repeat_table_header(table.rows[row_idx])
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            if widths and col_idx < len(widths):
                cell.width = Inches(widths[col_idx])
            set_cell_margins(cell, top=55, bottom=55, left=70, right=70)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 or col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(value.replace("**", ""))
            set_run_font(run, name=BODY_FONT, size=8.2 if len(rows[0]) > 5 else 8.8,
                         bold=bool(row_idx == 0 or "**" in value),
                         color=RGBColor(0xFF, 0xFF, 0xFF) if row_idx == 0 else RGBColor(0x1F, 0x1F, 0x1F))
            if row_idx == 0:
                set_cell_shading(cell, "1D3B5F")
            elif row_idx % 2 == 0:
                set_cell_shading(cell, "F3F6F9")

    note = document.add_paragraph()
    note.paragraph_format.space_before = Pt(2)
    note.paragraph_format.space_after = Pt(8)
    note.paragraph_format.first_line_indent = Cm(0)
    note.alignment = WD_ALIGN_PARAGRAPH.LEFT
    note_text = ("注：PSNR 与 SSIM 越高越好，RMSE 越低越好；运行时间为本机当前实现结果。"
                 if table_number == 1 else
                 "注：差值为自主方法减去 DocShadow，负值表示自主方法在该样例上仍落后。")
    set_run_font(note.add_run(note_text), name=BODY_FONT, size=8.0, color=RGBColor(0x66, 0x66, 0x66))


def build():
    lines = SOURCE.read_text(encoding="utf-8-sig").splitlines()
    document = Document()
    configure_document(document)
    add_cover(document)

    image_pattern = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
    table_number = 0
    skipped_title = False
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        if line.startswith("# ") and not skipped_title:
            skipped_title = True
            index += 1
            continue

        if line.startswith("## "):
            heading = line[3:].strip()
            document.add_paragraph(heading, style="Heading 1")
            index += 1
            continue

        if line.startswith("### "):
            document.add_paragraph(line[4:].strip(), style="Heading 2")
            index += 1
            continue

        image_match = image_pattern.match(line)
        if image_match:
            caption, relative = image_match.groups()
            add_figure(document, caption, (SOURCE.parent / relative).resolve())
            index += 1
            continue

        if line.startswith("|"):
            rows, index = parse_table(lines, index)
            table_number += 1
            add_table(document, rows, table_number)
            continue

        if line.startswith("**关键词：**"):
            add_keywords(document, line)
            index += 1
            continue

        add_body_paragraph(document, line)
        index += 1

    document.save(OUTPUT)
    char_count = sum(len(p.text) for p in document.paragraphs)
    print(f"output={OUTPUT}")
    print(f"size={OUTPUT.stat().st_size/1024/1024:.2f} MB")
    print(f"paragraph_chars={char_count} images={len(document.inline_shapes)} tables={len(document.tables)}")


if __name__ == "__main__":
    build()
