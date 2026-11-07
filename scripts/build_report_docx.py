from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/experiment_report.md"
OUTPUT = ROOT / "docs/experiment_report.docx"


def set_font(run, name: str = "Microsoft YaHei", size: float | None = None, color=None, bold=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color
    if bold is not None:
        run.font.bold = bold


def style_document(document: Document):
    styles = document.styles
    for name, size, color in [("Normal", 10.5, RGBColor(0x20, 0x20, 0x20)),
                              ("Title", 22, RGBColor(0x10, 0x18, 0x28)),
                              ("Heading 1", 16, RGBColor(0x18, 0x3B, 0x6D)),
                              ("Heading 2", 13, RGBColor(0x21, 0x50, 0x8C))]:
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.element.rPr.rFonts.set(qn("w:ascii"), "Microsoft YaHei")
        style.element.rPr.rFonts.set(qn("w:hAnsi"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.color.rgb = color
    normal = styles["Normal"]
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.22
    section = document.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_font(run, size=8.5, color=RGBColor(0x66, 0x66, 0x66))
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)
    run = paragraph.add_run(" 页 / ")
    set_font(run, size=8.5, color=RGBColor(0x66, 0x66, 0x66))
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "NUMPAGES")
    paragraph._p.append(field)
    run = paragraph.add_run(" 页")
    set_font(run, size=8.5, color=RGBColor(0x66, 0x66, 0x66))


def add_rich_paragraph(document, text: str, style: str | None = None):
    paragraph = document.add_paragraph(style=style)
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2]); set_font(run, bold=True)
        else:
            run = paragraph.add_run(part); set_font(run)
    return paragraph


def add_image(document, path: Path, caption: str):
    document.add_picture(str(path), width=Inches(6.35))
    last = document.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    last.paragraph_format.space_after = Pt(3)
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = caption_paragraph.add_run(caption)
    set_font(run, size=8.8, color=RGBColor(0x55, 0x55, 0x55))
    caption_paragraph.paragraph_format.space_after = Pt(12)


def format_table(document, rows: list[list[str]]):
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_index == 0 or col_index else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(1)
            clean = value.replace("**", "")
            run = paragraph.add_run(clean)
            set_font(run, size=7.5 if len(rows[0]) > 7 else 9, bold=(row_index == 0))
            if row_index == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "D9EAF7")
                cell._tc.get_or_add_tcPr().append(shading)
    document.add_paragraph().paragraph_format.space_after = Pt(4)


def parse_table_row(line: str):
    return [item.strip() for item in line.strip().strip("|").split("|")]


def build():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    document = Document()
    style_document(document)
    footer = document.sections[0].footer
    footer.paragraphs[0].text = ""
    add_page_number(footer.paragraphs[0])
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1; continue
        if line.startswith("### "):
            document.add_heading(line[4:].strip(), level=2)
        elif line.startswith("## "):
            document.add_heading(line[3:].strip(), level=1)
        elif line.startswith("# "):
            document.add_heading(line[2:].strip(), level=0)
        elif line.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(f"Bad image line: {line}")
            path = (SOURCE.parent / match.group(2)).resolve()
            if not path.exists(): raise FileNotFoundError(path)
            add_image(document, path, match.group(1))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = parse_table_row(lines[index])
                if not all(re.fullmatch(r":?-{2,}:?", item) for item in row):
                    rows.append(row)
                index += 1
            format_table(document, rows)
            continue
        else:
            add_rich_paragraph(document, line.strip())
        index += 1
    document.save(OUTPUT)
    print(OUTPUT, OUTPUT.stat().st_size)


if __name__ == "__main__":
    build()
