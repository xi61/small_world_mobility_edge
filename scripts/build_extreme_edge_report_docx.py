#!/usr/bin/env python3
"""Build the updated extreme-edge forensic report as a polished Word document."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


NAVY = "0B2545"
BLUE = "1D5D8C"
ACCENT = "2A7F9E"
TEXT = "24313D"
MUTED = "607080"
PALE_BLUE = "E8EEF5"
PALE_GRAY = "F4F6F8"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=75, start=95, bottom=75, end=95) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_twips: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_twips))
    tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_table_width(table, width_twips: int) -> None:
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(width_twips))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")


def set_paragraph_border_bottom(paragraph, color=ACCENT, size=10, space=5) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def set_keep_with_next(paragraph, value=True) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    node = p_pr.find(qn("w:keepNext"))
    if value and node is None:
        node = OxmlElement("w:keepNext")
        p_pr.append(node)
    elif not value and node is not None:
        p_pr.remove(node)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("PAGE  ")
    run.font.name = "Calibri"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED)
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr_text)
    run._r.append(fld_char_2)


def new_numbering_instance(doc: Document) -> int:
    """Create a fresh List Number instance so a separated list restarts at 1."""
    numbering = doc.part.numbering_part.element
    abstract_id = None
    for abstract in numbering.findall(qn("w:abstractNum")):
        for p_style in abstract.iter(qn("w:pStyle")):
            if p_style.get(qn("w:val")) == "ListNumber":
                abstract_id = int(abstract.get(qn("w:abstractNumId")))
                break
        if abstract_id is not None:
            break
    if abstract_id is None:
        raise RuntimeError("ListNumber numbering definition not found")

    existing = [int(node.get(qn("w:numId"))) for node in numbering.findall(qn("w:num"))]
    num_id = max(existing, default=0) + 1
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    override = OxmlElement("w:lvlOverride")
    override.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:startOverride")
    start.set(qn("w:val"), "1")
    override.append(start)
    num.append(override)
    numbering.append(num)
    return num_id


def apply_numbering(paragraph, num_id: int) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.append(ilvl)
    num_pr.append(num)
    p_pr.append(num_pr)


def configure_section(section, landscape=False) -> None:
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11)
        section.page_height = Inches(8.5)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)
        section.top_margin = Inches(0.72)
        section.bottom_margin = Inches(0.72)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.top_margin = Inches(0.82)
        section.bottom_margin = Inches(0.82)
    section.header_distance = Inches(0.38)
    section.footer_distance = Inches(0.38)


def configure_header_footer(section) -> None:
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0]
    p.clear()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.tab_stops.add_tab_stop(
        section.page_width - section.left_margin - section.right_margin,
        WD_TAB_ALIGNMENT.RIGHT,
    )
    r1 = p.add_run("PREFERENTIAL-ATTACHMENT SPECTRAL AUDIT")
    r1.font.name = "Calibri"
    r1.font.size = Pt(8)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor.from_string(NAVY)
    r2 = p.add_run("\tUPDATED 30 AUGUST 2026")
    r2.font.name = "Calibri"
    r2.font.size = Pt(8)
    r2.font.color.rgb = RGBColor.from_string(MUTED)

    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.clear()
    add_page_number(fp)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(TEXT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.18
    normal.paragraph_format.widow_control = True

    title = styles["Title"]
    title.font.name = "Calibri Light"
    title.font.size = Pt(25)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(NAVY)
    title.paragraph_format.space_before = Pt(2)
    title.paragraph_format.space_after = Pt(12)
    title.paragraph_format.keep_with_next = True

    for name, size, before, after, color in (
        ("Heading 1", 16, 16, 6, NAVY),
        ("Heading 2", 13, 13, 5, BLUE),
        ("Heading 3", 11.5, 10, 4, TEXT),
    ):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(10.5)
        style.font.color.rgb = RGBColor.from_string(TEXT)
        style.paragraph_format.left_indent = Inches(0.22)
        style.paragraph_format.first_line_indent = Inches(-0.16)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.12

    code = styles.add_style("Code Inline", WD_STYLE_TYPE.CHARACTER)
    code.font.name = "Consolas"
    code.font.size = Pt(8.6)
    code.font.color.rgb = RGBColor.from_string(NAVY)

    equation = styles.add_style("Equation Display", WD_STYLE_TYPE.PARAGRAPH)
    equation.font.name = "Cambria Math"
    equation.font.size = Pt(10.5)
    equation.font.color.rgb = RGBColor.from_string(TEXT)
    equation.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.space_before = Pt(4)
    equation.paragraph_format.space_after = Pt(7)
    equation.paragraph_format.keep_together = True


INLINE_TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")


def add_inline(paragraph, text: str, *, base_size=None, base_color=None, bold=False) -> None:
    position = 0
    for match in INLINE_TOKEN.finditer(text):
        if match.start() > position:
            run = paragraph.add_run(text[position:match.start()])
            run.bold = bold
            if base_size:
                run.font.size = Pt(base_size)
            if base_color:
                run.font.color.rgb = RGBColor.from_string(base_color)
        token = match.group(0)
        if token.startswith("**"):
            inner = token[2:-2]
            for part in re.split(r"(`[^`]+`)", inner):
                if not part:
                    continue
                if part.startswith("`"):
                    inner_run = paragraph.add_run(part[1:-1])
                    inner_run.style = "Code Inline"
                    inner_run.bold = True
                else:
                    inner_run = paragraph.add_run(part)
                    inner_run.bold = True
                    if base_size:
                        inner_run.font.size = Pt(base_size)
                    if base_color:
                        inner_run.font.color.rgb = RGBColor.from_string(base_color)
            position = match.end()
            continue
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.style = "Code Inline"
        else:
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        if base_size and not token.startswith("`"):
            run.font.size = Pt(base_size)
        if base_color and not token.startswith("`"):
            run.font.color.rgb = RGBColor.from_string(base_color)
        position = match.end()
    if position < len(text):
        run = paragraph.add_run(text[position:])
        run.bold = bold
        if base_size:
            run.font.size = Pt(base_size)
        if base_color:
            run.font.color.rgb = RGBColor.from_string(base_color)


def equation_to_text(latex: str) -> str:
    value = " ".join(part.strip() for part in latex.splitlines() if part.strip())
    known = {
        r"w_i=d_i+\delta,\qquad \delta=m(\alpha-2).": "wᵢ = dᵢ + δ,    δ = m(α − 2).",
        r"\Delta_i=\lambda_{i+1}-\lambda_i, \qquad E_i=(\lambda_i+\lambda_{i+1})/2, \qquad q_i^{\rm local}=(i+1/2)/M.": "Δᵢ = λᵢ₊₁ − λᵢ,    Eᵢ = (λᵢ + λᵢ₊₁)/2,    qᵢˡᵒᶜᵃˡ = (i + 1/2)/M.",
        r"r_i=\frac{\min(\Delta_i,\Delta_{i+1})} {\max(\Delta_i,\Delta_{i+1})},": "rᵢ = min(Δᵢ, Δᵢ₊₁) / max(Δᵢ, Δᵢ₊₁),",
        r"s_i=M\{F_{-f}(\lambda_{i+1})-F_{-f}(\lambda_i)\},": "sᵢ = M{F₋ᶠ(λᵢ₊₁) − F₋ᶠ(λᵢ)},",
        r"F_{GOE}(s)=1-e^{-\pi s^2/4} \quad\text{or}\quad F_P(s)=1-e^{-s}": "FGOE(s) = 1 − e^(−πs²/4)    or    FP(s) = 1 − e^(−s)",
        r"r_i^{unf}=\frac{\min(s_i,s_{i+1})}{\max(s_i,s_{i+1})}": "rᵢᵘⁿᶠ = min(sᵢ, sᵢ₊₁) / max(sᵢ, sᵢ₊₁)",
    }
    if value in known:
        return known[value]
    replacements = {
        r"\Delta": "Δ", r"\lambda": "λ", r"\delta": "δ", r"\alpha": "α",
        r"\zeta": "ζ", r"\pi": "π", r"\qquad": "    ", r"\quad": "  ",
        r"\geq": "≥", r"\leq": "≤", r"\text{or}": "or", r"\rm ": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = value.replace("\\{", "{").replace("\\}", "}")
    return value


def parse_blocks(markdown: str):
    lines = markdown.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == r"\[":
            eq = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                eq.append(lines[i])
                i += 1
            blocks.append(("equation", "\n".join(eq)))
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", c) for c in rows[1]):
                rows.pop(1)
            blocks.append(("table", rows))
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            blocks.append(("heading", len(heading.group(1)), heading.group(2)))
            i += 1
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            blocks.append(("bullet", bullet.group(1)))
            i += 1
            continue
        numbered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if numbered:
            blocks.append(("number", numbered.group(1)))
            i += 1
            continue
        if stripped.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code)))
            i += 1
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt == r"\[" or nxt.startswith("|") or nxt.startswith("```"):
                break
            if re.match(r"^(#{1,4})\s+", lines[i]) or re.match(r"^\s*[-*]\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i]):
                break
            paragraph.append(nxt)
            i += 1
        blocks.append(("paragraph", " ".join(paragraph)))
    return blocks


def column_widths(rows, total_twips: int):
    cols = len(rows[0])
    maxima = []
    for index in range(cols):
        max_len = max(len(re.sub(r"[`*]", "", row[index])) for row in rows)
        maxima.append(max(7, min(max_len, 42)))
    if cols >= 8:
        maxima[0] = max(maxima[0], 24)
        for index in range(1, cols):
            maxima[index] = max(maxima[index], 11)
    weights = [m ** 0.75 for m in maxima]
    widths = [int(total_twips * w / sum(weights)) for w in weights]
    widths[-1] += total_twips - sum(widths)
    return widths


def add_table(doc: Document, rows, total_twips: int):
    cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_width(table, total_twips)
    widths = column_widths(rows, total_twips)
    font_size = 6.9 if cols >= 8 else (7.5 if cols >= 6 else 8.0)

    for row_index, source_row in enumerate(rows):
        target_row = table.rows[row_index]
        prevent_row_split(target_row)
        if row_index == 0:
            set_repeat_table_header(target_row)
        for col_index, value in enumerate(source_row):
            cell = target_row.cells[col_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_width(cell, widths[col_index])
            set_cell_margins(cell)
            if row_index == 0:
                set_cell_shading(cell, NAVY)
            elif row_index % 2 == 0:
                set_cell_shading(cell, PALE_GRAY)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add_inline(
                p,
                value,
                base_size=font_size,
                base_color=WHITE if row_index == 0 else TEXT,
                bold=row_index == 0,
            )
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.space_before = Pt(0)
    return table


def build(source: Path, target: Path) -> None:
    markdown = source.read_text(encoding="utf-8")
    blocks = parse_blocks(markdown)
    doc = Document()
    configure_styles(doc)
    configure_section(doc.sections[0], landscape=False)
    configure_header_footer(doc.sections[0])

    first_heading = True
    current_num_id = None
    wide_section_active = False
    index = 0
    while index < len(blocks):
        block = blocks[index]
        kind = block[0]
        if kind != "number":
            current_num_id = None
        if kind == "heading":
            level, text = block[1], block[2]
            if text == "Why gap and ratio counts can differ" and not wide_section_active:
                landscape = doc.add_section(WD_SECTION.NEW_PAGE)
                configure_section(landscape, landscape=True)
                configure_header_footer(landscape)
                wide_section_active = True
            if level == 1 and first_heading:
                kicker = doc.add_paragraph()
                kicker.paragraph_format.space_before = Pt(8)
                kicker.paragraph_format.space_after = Pt(3)
                run = kicker.add_run("TECHNICAL FORENSIC REPORT")
                run.font.name = "Calibri"
                run.font.size = Pt(9)
                run.font.bold = True
                run.font.color.rgb = RGBColor.from_string(ACCENT)
                title = doc.add_paragraph(style="Title")
                add_inline(title, text)
                set_paragraph_border_bottom(title, color=ACCENT, size=13, space=7)
                first_heading = False
            else:
                style_level = min(level - 1, 3)
                p = doc.add_paragraph(style=f"Heading {style_level}")
                add_inline(p, text)
                if style_level == 1:
                    set_paragraph_border_bottom(p, color=PALE_BLUE, size=5, space=3)
            index += 1
            continue
        if kind == "paragraph":
            p = doc.add_paragraph()
            add_inline(p, block[1])
            if block[1].startswith("**Status:**"):
                set_cell_shading_for_paragraph(p, PALE_BLUE)
                p.paragraph_format.left_indent = Inches(0.12)
                p.paragraph_format.right_indent = Inches(0.12)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(12)
            index += 1
            continue
        if kind in ("bullet", "number"):
            p = doc.add_paragraph(style="List Bullet" if kind == "bullet" else "List Number")
            if kind == "number":
                if current_num_id is None:
                    current_num_id = new_numbering_instance(doc)
                apply_numbering(p, current_num_id)
            add_inline(p, block[1])
            index += 1
            continue
        if kind == "equation":
            p = doc.add_paragraph(style="Equation Display")
            p.add_run(equation_to_text(block[1]))
            index += 1
            continue
        if kind == "code":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.18)
            p.paragraph_format.right_indent = Inches(0.18)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(6)
            set_cell_shading_for_paragraph(p, PALE_GRAY)
            run = p.add_run(block[1])
            run.font.name = "Consolas"
            run.font.size = Pt(8.2)
            index += 1
            continue
        if kind == "table":
            rows = block[1]
            cols = len(rows[0])
            if cols >= 8:
                if not wide_section_active:
                    landscape = doc.add_section(WD_SECTION.NEW_PAGE)
                    configure_section(landscape, landscape=True)
                    configure_header_footer(landscape)
                    wide_section_active = True
                else:
                    landscape = doc.sections[-1]
                width = int((landscape.page_width - landscape.left_margin - landscape.right_margin) / 635)
                add_table(doc, rows, width)
                portrait = doc.add_section(WD_SECTION.NEW_PAGE)
                configure_section(portrait, landscape=False)
                configure_header_footer(portrait)
                wide_section_active = False
            else:
                section = doc.sections[-1]
                width = int((section.page_width - section.left_margin - section.right_margin) / 635)
                add_table(doc, rows, width)
            index += 1
            continue
        raise RuntimeError(f"Unhandled block: {block}")

    # Give the appendix a clean page start without changing its text or order.
    for paragraph in doc.paragraphs:
        if paragraph.text.strip() == "Appendix A: support for every exact fine cell":
            paragraph.paragraph_format.page_break_before = True
            break

    core = doc.core_properties
    core.title = "Forensic report: abnormal extreme-edge behavior in preferential-attachment spectra"
    core.subject = "Updated abnormal-band microscope results and forensic audit"
    core.author = "SW random-graph mobility-edge project"
    core.keywords = "preferential attachment, mobility edge, adjacent-gap ratio, unfolding, spectral audit"
    core.comments = "Updated 30 August 2026; original audit content preserved with the completed microscope results added."

    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target)


def set_cell_shading_for_paragraph(paragraph, fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:fill"), fill)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_extreme_edge_report_docx.py SOURCE.md TARGET.docx")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
