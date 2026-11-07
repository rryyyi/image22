from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "experiment_report_expanded.md"
OUTPUT = ROOT / "docs" / "experiment_report_expanded.docx"

BODY_FONT = "Microsoft YaHei"
MONO_FONT = "Consolas"
TEXT_COLOR = RGBColor(0x24, 0x24, 0x24)
HEADING_COLOR = RGBColor(0x1D, 0x3B, 0x5F)
SUBHEADING_COLOR = RGBColor(0x2C, 0x58, 0x83)
MUTED_COLOR = RGBColor(0x66, 0x66, 0x66)
ACCENT_COLOR = RGBColor(0x2F, 0x6F, 0xB0)


def set_run_font(run, name: str = BODY_FONT, size: float | None = None,
                 bold: bool | None = None, color: RGBColor | None = None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 70, bottom: int = 70, left: int = 90, right: int = 90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_field(paragraph, instruction: str):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(separate)
    run._r.append(placeholder)
    run._r.append(end)
    set_run_font(run, size=8.6, color=MUTED_COLOR)


def configure_document(document: Document):
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.05)
    section.bottom_margin = Cm(1.95)
    section.left_margin = Cm(2.15)
    section.right_margin = Cm(2.15)
    section.header_distance = Cm(0.9)
    section.footer_distance = Cm(0.9)
    section.different_first_page_header_footer = True

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = BODY_FONT
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
    normal.element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
    normal.font.size = Pt(10.8)
    normal.font.color.rgb = TEXT_COLOR
    normal.paragraph_format.line_spacing = 1.52
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    title = styles["Title"]
    title.font.name = BODY_FONT
    title.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    title.element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
    title.element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
    title.font.size = Pt(25)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0x12, 0x12, 0x12)
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(8)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for name, size, color, before, after in (
        ("Heading 1", 16, HEADING_COLOR, 15, 7),
        ("Heading 2", 12.5, SUBHEADING_COLOR, 11, 5),
        ("Heading 3", 11.2, SUBHEADING_COLOR, 9, 4),
    ):
        style = styles[name]
        style.font.name = BODY_FONT
        style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        style.element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
        style.element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("文档图像修复实验报告")
    set_run_font(run, size=8.4, color=RGBColor(0x7A, 0x7A, 0x7A))

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("第 ")
    set_run_font(run, size=8.6, color=MUTED_COLOR)
    add_field(footer, "PAGE")
    run = footer.add_run(" 页 / ")
    set_run_font(run, size=8.6, color=MUTED_COLOR)
    add_field(footer, "NUMPAGES")
    run = footer.add_run(" 页")
    set_run_font(run, size=8.6, color=MUTED_COLOR)

    props = document.core_properties
    props.title = "文档图像修复实验报告"
    props.subject = "文档图像修复、光照估计与质量评估"
    props.author = ""
    props.keywords = "文档图像修复, 光照估计, 形态学背景, 质量评估"


def add_inline_runs(paragraph, text: str, base_size: float = 10.8,
                    base_color: RGBColor = TEXT_COLOR):
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
    for part in pattern.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, size=base_size, bold=True, color=RGBColor(0x1A, 0x2F, 0x47))
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, name=MONO_FONT, size=base_size - 0.6, color=RGBColor(0x3D, 0x4F, 0x63))
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=base_size, color=base_color)


def add_cover(document: Document):
    for _ in range(3):
        p = document.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    kicker = document.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = kicker.add_run("实验报告")
    set_run_font(run, size=11.5, bold=True, color=ACCENT_COLOR)
    kicker.paragraph_format.space_after = Pt(18)

    title = document.add_paragraph("文档图像修复实验报告", style="Title")
    title.paragraph_format.space_before = Pt(0)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("页面理解、光照恢复与质量评估的实验记录")
    set_run_font(run, size=14, color=RGBColor(0x45, 0x50, 0x5C))
    subtitle.paragraph_format.space_after = Pt(26)

    for _ in range(4):
        p = document.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    info_lines = (
        "项目：本地文档图像修复与质量评估",
        "实验数据：Jung 20 对真实文档、合成退化样例与真实中文合同",
        "日期：2026 年 11 月",
    )
    for line in info_lines:
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(7)
        run = p.add_run(line)
        set_run_font(run, size=10.5, color=RGBColor(0x4C, 0x4C, 0x4C))

    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)


def add_body_paragraph(document: Document, text: str):
    p = document.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.line_spacing = 1.52
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.widow_control = True
    add_inline_runs(p, text)
    return p


def add_keywords(document: Document, text: str):
    p = document.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(11)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_inline_runs(p, text, base_size=10.3, base_color=RGBColor(0x3C, 0x3C, 0x3C))


def add_figure(document: Document, caption: str, image_path: Path):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    run = p.add_run()
    inline_shape = run.add_picture(str(image_path), width=Inches(6.34))
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", caption)
    doc_pr.set("title", caption)

    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.space_after = Pt(10)
    cap.paragraph_format.keep_together = True
    run = cap.add_run(caption)
    set_run_font(run, size=8.9, color=RGBColor(0x5E, 0x5E, 0x5E))


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
    width_by_table = {1: [1.15, 0.91, 0.85, 0.88, 1.02, 0.91, 0.86],
                      2: [1.05, 1.38, 1.38, 1.38, 1.35]}
    widths = width_by_table.get(table_number)
    caption_text = ("表 1 四种方法在 Jung 20 对真实文档上的主要指标"
                    if table_number == 1 else
                    "表 2 三个困难样例的 PSNR 对比")
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(5)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.keep_with_next = True
    run = cap.add_run(caption_text)
    set_run_font(run, size=9.1, bold=True, color=RGBColor(0x3D, 0x4D, 0x60))

    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_layout = OxmlElement("w:tblLayout")
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_pr.append(tbl_layout)

    for row_idx, row in enumerate(rows):
        prevent_row_split(table.rows[row_idx])
        if row_idx == 0:
            repeat_table_header(table.rows[row_idx])
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            if widths and col_idx < len(widths):
                cell.width = Inches(widths[col_idx])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 or col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            clean_value = value.replace("**", "")
            run = paragraph.add_run(clean_value)
            is_bold = bool(row_idx == 0 or "**" in value)
            set_run_font(run, size=8.1 if len(rows[0]) > 5 else 8.8,
                         bold=is_bold, color=RGBColor(0x1F, 0x1F, 0x1F) if row_idx else RGBColor(0xFF, 0xFF, 0xFF))
            if row_idx == 0:
                set_cell_shading(cell, "1D3B5F")
            elif row_idx % 2 == 0:
                set_cell_shading(cell, "F3F6F9")
    note = document.add_paragraph()
    note.paragraph_format.space_before = Pt(2)
    note.paragraph_format.space_after = Pt(9)
    note.paragraph_format.first_line_indent = Cm(0)
    note.alignment = WD_ALIGN_PARAGRAPH.LEFT
    note_text = ("注：PSNR 与 SSIM 越高越好，RMSE 越低越好；运行时间为本机当前实现结果。"
                 if table_number == 1 else
                 "注：差值为自主方法减去 DocShadow，负值表示自主方法在该样例上仍落后。")
    run = note.add_run(note_text)
    set_run_font(run, size=8.1, color=RGBColor(0x66, 0x66, 0x66))


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

        if line.startswith("### "):
            document.add_paragraph(line[4:].strip(), style="Heading 3")
            index += 1
            continue

        if line.startswith("## "):
            document.add_paragraph(line[3:].strip(), style="Heading 1")
            index += 1
            continue

        image_match = image_pattern.match(line)
        if image_match:
            caption, relative = image_match.groups()
            image_path = (SOURCE.parent / relative).resolve()
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            add_figure(document, caption, image_path)
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
    paragraphs = sum(len(p.text) for p in document.paragraphs)
    tables = sum(len(cell.text) for table in document.tables for row in table.rows for cell in row.cells)
    print(f"output={OUTPUT}")
    print(f"size={OUTPUT.stat().st_size}")
    print(f"paragraph_chars={paragraphs} table_chars={tables} images={len(document.inline_shapes)} tables={len(document.tables)}")


if __name__ == "__main__":
    build()


