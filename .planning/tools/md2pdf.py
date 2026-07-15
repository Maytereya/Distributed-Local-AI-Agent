"""Markdown -> PDF converter tuned for Russian text (Cyrillic) and simple prose.

Handles: headings (# ## ###), **bold**, *italic*, `code`, > quotes, lists (- / 1.),
tables (|...|), fenced code blocks (```), horizontal rules (---), links [text](url).
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Preformatted,
)

FONT_DIR = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Body", f"{FONT_DIR}/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", f"{FONT_DIR}/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", f"{FONT_DIR}/Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", f"{FONT_DIR}/Arial Bold Italic.ttf"))
pdfmetrics.registerFont(TTFont("Mono", "/System/Library/Fonts/Menlo.ttc"))

from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")

styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontName="Body", fontSize=10.5, leading=15, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"))
H1 = ParagraphStyle("H1", parent=BODY, fontName="Body-Bold", fontSize=20, leading=26, spaceBefore=12, spaceAfter=12, textColor=colors.HexColor("#0b3d91"))
H2 = ParagraphStyle("H2", parent=BODY, fontName="Body-Bold", fontSize=15, leading=20, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#0b3d91"))
H3 = ParagraphStyle("H3", parent=BODY, fontName="Body-Bold", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#333"))
H4 = ParagraphStyle("H4", parent=BODY, fontName="Body-Bold", fontSize=11, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#333"))
QUOTE = ParagraphStyle("Quote", parent=BODY, leftIndent=14, fontName="Body-Italic", textColor=colors.HexColor("#444"), borderPadding=(4, 6, 4, 6))
LIST_ITEM = ParagraphStyle("ListItem", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
TABLE_CELL = ParagraphStyle("TableCell", parent=BODY, fontSize=9.5, leading=13, spaceAfter=0)
TABLE_CELL_BOLD = ParagraphStyle("TableCellBold", parent=TABLE_CELL, fontName="Body-Bold")
CODE = ParagraphStyle("Code", parent=BODY, fontName="Mono", fontSize=9, leading=12, textColor=colors.HexColor("#222"), backColor=colors.HexColor("#f4f4f4"), borderPadding=(6, 8, 6, 8), leftIndent=0)

EMOJI_MAP = {"⚠️": "[!]", "🤝": "[handoff]", "→": " -> ", "≤": "<=", "≥": ">=", "≈": "~"}


def clean_text(s):
    for k, v in EMOJI_MAP.items():
        s = s.replace(k, v)
    return s


def inline_md(s):
    s = clean_text(s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("-&gt;", "->")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r'<font name="Mono" backcolor="#f4f4f4">\1</font>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#0b3d91">\1</link>', s)
    return s


def parse_markdown(md):
    flow = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Preformatted(clean_text("\n".join(buf)), CODE))
            flow.append(Spacer(1, 4))
            continue
        if re.match(r"^---+\s*$", line):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
            flow.append(Spacer(1, 6))
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            text = inline_md(m.group(2))
            style = {1: H1, 2: H2, 3: H3, 4: H4}[level]
            flow.append(Paragraph(text, style))
            i += 1
            continue
        if line.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            text = inline_md(" ".join(x for x in buf if x))
            flow.append(Paragraph(text, QUOTE))
            flow.append(Spacer(1, 4))
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            flow.append(build_table(table_lines))
            flow.append(Spacer(1, 6))
            continue
        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i]) or (lines[i].startswith("  ") and lines[i].strip() and buf)):
                buf.append(lines[i])
                i += 1
            for item in buf:
                m_b = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", item)
                if m_b:
                    indent_n = len(m_b.group(1))
                    marker = m_b.group(2)
                    rest = m_b.group(3)
                    bullet = "•" if marker in ("-", "*") else marker
                    text = inline_md(rest)
                    style = LIST_ITEM
                    if indent_n >= 2:
                        style = ParagraphStyle("ListNested", parent=LIST_ITEM, leftIndent=32, bulletIndent=20)
                    flow.append(Paragraph(f"{bullet}  {text}", style))
                else:
                    flow.append(Paragraph(inline_md(item.strip()), ParagraphStyle("Cont", parent=BODY, leftIndent=22)))
            flow.append(Spacer(1, 4))
            continue
        if not line.strip():
            flow.append(Spacer(1, 4))
            i += 1
            continue
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not is_block_start(lines[i]):
            buf.append(lines[i])
            i += 1
        text = inline_md(" ".join(x.strip() for x in buf))
        flow.append(Paragraph(text, BODY))
    return flow


def is_block_start(line):
    if re.match(r"^#{1,4}\s", line): return True
    if line.startswith(">"): return True
    if line.strip().startswith("```"): return True
    if re.match(r"^---+\s*$", line): return True
    if line.strip().startswith("|"): return True
    if re.match(r"^\s*[-*]\s+", line): return True
    if re.match(r"^\s*\d+\.\s+", line): return True
    return False


def build_table(lines):
    header_cells = split_row(lines[0])
    data_rows = [split_row(r) for r in lines[2:]]
    header = [Paragraph(inline_md(c), TABLE_CELL_BOLD) for c in header_cells]
    rows = [[Paragraph(inline_md(c), TABLE_CELL) for c in row] for row in data_rows]
    table_data = [header] + rows
    n_cols = len(header)
    page_width = A4[0] - 4 * cm
    col_widths = [page_width / n_cols] * n_cols
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
    ]))
    return t


def split_row(line):
    line = line.strip()
    if line.startswith("|"): line = line[1:]
    if line.endswith("|"): line = line[:-1]
    return [c.strip() for c in line.split("|")]


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"стр. {doc.page}")
    canvas.restoreState()


def md_to_pdf(md_path, pdf_path, title):
    md = Path(md_path).read_text(encoding="utf-8")
    flow = parse_markdown(md)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm, title=title, author="Neiry AI")
    doc.build(flow, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"[ok] {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        md_to_pdf(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("usage: md2pdf.py <input.md> <output.pdf> <title>")
