"""Многостраничная презентация результатов бота в стиле Neiry AI.

Берёт живые ответы из docs/demo_cases.json и рендерит PDF
docs/NEIRY_BOT_DEMO.pdf. Запуск:

    venv/bin/python scripts/build_client_slides_v2.py
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


# --- Fonts --------------------------------------------------------------
pdfmetrics.registerFont(TTFont("Body", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-B", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-I", "/System/Library/Fonts/Supplemental/Arial Italic.ttf"))

BODY = "Body"
BOLD = "Body-B"
ITAL = "Body-I"

# --- Neiry palette ------------------------------------------------------
INK = HexColor("#0F1E3D")       # deep navy body
INK_SOFT = HexColor("#374B6A")  # secondary text
MUTED = HexColor("#7A8BA6")     # tertiary
BLUE = HexColor("#2F6BFF")      # primary brand blue
BLUE_SOFT = HexColor("#E2ECFF") # light blue chip bg
GREEN = HexColor("#23D864")     # brand green
GREEN_SOFT = HexColor("#E0F9EA") # light green chip bg
BG = HexColor("#FFFFFF")
CARD_BG = HexColor("#FFFFFF")
CARD_BORDER = HexColor("#E2E8F3")
PATIENT_BUBBLE = HexColor("#F3F5F9")
BOT_BUBBLE = HexColor("#EFF5FF")
RULE = HexColor("#E2E8F3")
ACCENT_GRADIENT_A = HexColor("#2F6BFF")
ACCENT_GRADIENT_B = HexColor("#23D864")

# --- Layout constants ---------------------------------------------------
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 14 * mm
LOGO_PATH = "docs/assets/neiry_logo.png"


@dataclass
class Case:
    case_id: str
    capability: str
    tz_point: int
    turns: list[dict[str, str]]  # [{user, bot, handoff}]
    caption: str


def _load_cases() -> list[Case]:
    data = json.loads(Path("docs/demo_cases.json").read_text())
    cases = []
    for row in data:
        cases.append(
            Case(
                case_id=row["case_id"],
                capability=row["capability"],
                tz_point=row.get("tz_point", 0),
                turns=row["runs"],
                caption=row.get("caption", ""),
            )
        )
    return cases


# =============================================================================
#  SHARED TEMPLATE CHROME
# =============================================================================

def draw_check(c, x, y, size=3.2 * mm, color=None):
    """Vector checkmark inside a filled green circle (Arial lacks the glyph)."""
    color = color or GREEN
    r = size / 2
    cx, cy = x + r, y + r
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.setLineWidth(0)
    c.circle(cx, cy, r, fill=True, stroke=False)
    # white tick
    c.setStrokeColor(HexColor("#FFFFFF"))
    c.setLineCap(1)
    c.setLineJoin(1)
    c.setLineWidth(r * 0.5)
    p = c.beginPath()
    p.moveTo(cx - r * 0.5, cy + r * 0.05)
    p.lineTo(cx - r * 0.1, cy - r * 0.35)
    p.lineTo(cx + r * 0.55, cy + r * 0.35)
    c.drawPath(p, stroke=True, fill=False)


def draw_page_chrome(c: canvas.Canvas, page_num: int, total: int, tag: str | None = None) -> None:
    """Logo top-right, optional tag top-left, footer with page number."""
    # Logo (28mm wide, preserve aspect)
    try:
        from reportlab.lib.utils import ImageReader
        img = ImageReader(LOGO_PATH)
        iw, ih = img.getSize()
        target_w = 28 * mm
        target_h = target_w * ih / iw
        # Paint a white card behind the logo so any PNG decode oddities (alpha
        # channel compositing → black box) disappear.
        pad = 1 * mm
        lx = PAGE_W - MARGIN - target_w
        ly = PAGE_H - MARGIN - target_h
        c.setFillColor(HexColor("#FFFFFF"))
        c.rect(lx - pad, ly - pad, target_w + 2 * pad, target_h + 2 * pad,
               fill=True, stroke=False)
        c.drawImage(
            img, lx, ly, width=target_w, height=target_h,
        )
    except Exception:
        # Fallback text logo
        c.setFillColor(BLUE)
        c.setFont(BOLD, 16)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN - 6 * mm, "Neiry AI")

    # Optional tag (top-left)
    if tag:
        tag_w = c.stringWidth(tag, BOLD, 8.5) + 10 * mm
        tag_h = 7 * mm
        ty = PAGE_H - MARGIN - tag_h + 1 * mm
        c.setFillColor(GREEN_SOFT)
        c.roundRect(MARGIN, ty, tag_w, tag_h, 2 * mm, fill=True, stroke=False)
        c.setFillColor(GREEN)
        c.setFont(BOLD, 8.5)
        c.drawString(MARGIN + 5 * mm, ty + 2.2 * mm, tag.upper())

    # Footer
    c.setStrokeColor(RULE)
    c.setLineWidth(0.3)
    c.line(MARGIN, MARGIN - 3 * mm, PAGE_W - MARGIN, MARGIN - 3 * mm)
    c.setFillColor(MUTED)
    c.setFont(BODY, 8)
    c.drawString(MARGIN, MARGIN - 7 * mm, "Neiry AI · демонстрация возможностей чат-бота клиники")
    c.drawRightString(PAGE_W - MARGIN, MARGIN - 7 * mm, f"{page_num} / {total}")


# =============================================================================
#  CARD / BUBBLE PRIMITIVES
# =============================================================================

def draw_card(c, x, y, w, h, radius=3 * mm, accent: str | None = "blue"):
    # soft shadow
    c.setFillColor(HexColor("#F4F6FB"))
    c.roundRect(x + 0.5 * mm, y - 0.8 * mm, w, h, radius, fill=True, stroke=False)
    # body
    c.setFillColor(CARD_BG)
    c.setStrokeColor(CARD_BORDER)
    c.setLineWidth(0.5)
    c.roundRect(x, y, w, h, radius, fill=True, stroke=True)
    # accent stripe on top
    if accent:
        color = BLUE if accent == "blue" else GREEN
        c.setFillColor(color)
        c.roundRect(x + radius, y + h - 1.1 * mm, w - 2 * radius, 1.1 * mm, 0, fill=True, stroke=False)


def measure_paragraph(text: str, style: ParagraphStyle, width: float) -> float:
    """Return wrap height for a paragraph at given width."""
    p = Paragraph(text, style)
    _, h = p.wrap(width, 500 * mm)
    return h


def draw_paragraph(c, text: str, style: ParagraphStyle, x: float, y_top: float, width: float) -> float:
    """Draw Paragraph at (x, y_top = top-left) and return final y-bottom."""
    p = Paragraph(text, style)
    _, h = p.wrap(width, 500 * mm)
    p.drawOn(c, x, y_top - h)
    return y_top - h


def draw_bubble(c, role: str, text: str, x_left: float, y_top: float, max_width: float) -> float:
    """Draw a chat bubble. role ∈ {'patient','bot'}. Returns y-bottom."""
    if role == "patient":
        bg = PATIENT_BUBBLE
        fg = INK
        align_right = False
        label = "Пациент"
        label_color = MUTED
    else:
        bg = BOT_BUBBLE
        fg = INK
        align_right = False
        label = "Бот"
        label_color = BLUE

    # Slight inner bubble padding
    pad_x = 4 * mm
    pad_y = 3 * mm
    bubble_w = max_width

    # text style
    text_style = ParagraphStyle(
        "bubble", fontName=BODY, fontSize=9.5, textColor=fg, leading=12.5,
    )
    # normalise whitespace (keep line breaks)
    normalised = text.replace("\r\n", "\n").strip()
    # Escape angle brackets for Paragraph
    normalised = (normalised
                  .replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;"))
    # Convert newlines to <br/>
    html = normalised.replace("\n", "<br/>")

    p = Paragraph(html, text_style)
    _, th = p.wrap(bubble_w - 2 * pad_x, 500 * mm)

    bubble_h = th + 2 * pad_y + 4 * mm  # +room for label

    # label above bubble
    c.setFillColor(label_color)
    c.setFont(BOLD, 7.5)
    c.drawString(x_left + 1 * mm, y_top - 2.5 * mm, label.upper())

    # bubble body
    by = y_top - 4 * mm - (bubble_h - 4 * mm)
    c.setFillColor(bg)
    c.setStrokeColor(CARD_BORDER)
    c.setLineWidth(0.4)
    c.roundRect(x_left, by, bubble_w, bubble_h - 4 * mm, 2.5 * mm, fill=True, stroke=True)

    # text inside bubble
    p.drawOn(c, x_left + pad_x, by + pad_y + 1 * mm)

    return by - 2 * mm  # spacing below


# =============================================================================
#  SLIDE 1 — TITLE
# =============================================================================

def slide_title(c, page_num, total):
    draw_page_chrome(c, page_num, total)

    # small tag
    tag = "ДЕМОНСТРАЦИЯ ВОЗМОЖНОСТЕЙ"
    c.setFillColor(BLUE_SOFT)
    tag_w = c.stringWidth(tag, BOLD, 9) + 10 * mm
    c.roundRect(MARGIN, PAGE_H - MARGIN - 20 * mm, tag_w, 8 * mm, 2 * mm, fill=True, stroke=False)
    c.setFillColor(BLUE)
    c.setFont(BOLD, 9)
    c.drawString(MARGIN + 5 * mm, PAGE_H - MARGIN - 20 * mm + 2.6 * mm, tag)

    # Title
    c.setFillColor(INK)
    c.setFont(BOLD, 40)
    c.drawString(MARGIN, PAGE_H - MARGIN - 40 * mm, "Чат-бот клиники")
    c.drawString(MARGIN, PAGE_H - MARGIN - 54 * mm, "в живых сценариях")

    # Subtitle
    sub_style = ParagraphStyle(
        "sub", fontName=BODY, fontSize=13, textColor=INK_SOFT, leading=18,
    )
    draw_paragraph(
        c,
        ("Ниже — реальные ответы бота на запросы пациентов, записанные "
         "с живого сервера клиники. Каждый слайд — один сценарий из ТЗ."),
        sub_style, MARGIN, PAGE_H - MARGIN - 62 * mm, 170 * mm,
    )

    # Right column: 8 capability chips
    right_x = 180 * mm
    right_y = PAGE_H - MARGIN - 26 * mm
    c.setFillColor(INK)
    c.setFont(BOLD, 11)
    c.drawString(right_x, right_y, "Все 8 пунктов ТЗ — в работе")

    items = [
        "Результаты анализов",
        "Стоимость анализов",
        "Услуга + 4 врача + подготовка",
        "Информация о враче по ФИО",
        "Врачи специальности",
        "Расписание врача",
        "Запись к врачу",
        "Справка в налоговую",
    ]
    cy = right_y - 6 * mm
    for item in items:
        draw_check(c, right_x - 0.5 * mm, cy - 7 * mm, size=4 * mm)
        c.setFillColor(INK)
        c.setFont(BODY, 10.5)
        c.drawString(right_x + 6 * mm, cy - 5.3 * mm, item)
        cy -= 7.5 * mm

    # Bottom sticker strip
    strip_y = MARGIN + 3 * mm
    c.setFillColor(GREEN_SOFT)
    c.roundRect(MARGIN, strip_y, PAGE_W - 2 * MARGIN, 12 * mm, 2 * mm, fill=True, stroke=False)
    c.setFillColor(HexColor("#046B3F"))
    c.setFont(BOLD, 10)
    c.drawString(MARGIN + 6 * mm, strip_y + 4.5 * mm,
                 "Все ответы на следующих слайдах получены на живом API — это не макеты.")


# =============================================================================
#  SLIDE 2 — OVERVIEW GRID
# =============================================================================

def slide_overview(c, page_num, total):
    draw_page_chrome(c, page_num, total, tag="обзор")

    c.setFillColor(INK)
    c.setFont(BOLD, 26)
    c.drawString(MARGIN, PAGE_H - MARGIN - 24 * mm, "Что бот умеет делать сейчас")

    sub_style = ParagraphStyle(
        "sub", fontName=BODY, fontSize=11, textColor=INK_SOFT, leading=15,
    )
    draw_paragraph(
        c,
        "8 капабельностей, каждая подтверждена реальным примером из живого сервера на следующих слайдах.",
        sub_style, MARGIN, PAGE_H - MARGIN - 28 * mm, 230 * mm,
    )

    items = [
        ("Результаты анализов", "Выдаёт ссылку на результат; при сбое — аккуратный handoff"),
        ("Стоимость анализов", "Понимает «cito», «на дому», «детский», списки услуг"),
        ("Услуга + 4 врача + подготовка", "Врачи по рейтингу + правила подготовки"),
        ("Информация о враче по ФИО", "Удерживает имя врача на всём диалоге"),
        ("Врачи специальности", "Связывает процедуру со специальностью автоматически"),
        ("Расписание врача", "Различает «врача нет» и «нет свободных слотов»"),
        ("Запись к врачу", "Выбор времени → ФИО → подтверждение → оператор"),
        ("Справка в налоговую", "Готовая инструкция текстом, без лишнего handoff"),
    ]

    grid_top = PAGE_H - MARGIN - 44 * mm
    grid_bottom = MARGIN + 5 * mm
    grid_h = grid_top - grid_bottom
    cols = 4
    rows = 2
    gap_x = 4 * mm
    gap_y = 4 * mm
    card_w = (PAGE_W - 2 * MARGIN - (cols - 1) * gap_x) / cols
    card_h = (grid_h - (rows - 1) * gap_y) / rows

    for idx, (title, desc) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = MARGIN + col * (card_w + gap_x)
        y = grid_top - card_h - row * (card_h + gap_y)
        draw_card(c, x, y, card_w, card_h, accent="blue" if row == 0 else "green")
        # number
        c.setFillColor(MUTED)
        c.setFont(BOLD, 9)
        c.drawString(x + 5 * mm, y + card_h - 7 * mm, f"0{idx+1}")
        # title
        c.setFillColor(INK)
        c.setFont(BOLD, 11)
        c.drawString(x + 5 * mm, y + card_h - 14 * mm, title)
        # description
        desc_style = ParagraphStyle(
            "d", fontName=BODY, fontSize=9, textColor=INK_SOFT, leading=11.5,
        )
        draw_paragraph(c, desc, desc_style, x + 5 * mm, y + card_h - 18 * mm, card_w - 10 * mm)


# =============================================================================
#  SLIDE — ONE SCENARIO (Q/A)
# =============================================================================

def _trim_for_display(text: str, max_chars: int) -> tuple[str, bool]:
    """Hard-trim to max_chars with ellipsis, returning truncated flag."""
    if len(text) <= max_chars:
        return text, False
    # Try to cut at a paragraph break
    cut = text[:max_chars]
    last_break = max(cut.rfind("\n"), cut.rfind(". "))
    if last_break > max_chars * 0.6:
        cut = cut[:last_break + 1]
    return cut.strip() + " …", True


def slide_scenario(
    c, page_num, total, *, tag: str, title: str, case: Case,
    max_bubble_chars: int = 900, turns_subset: tuple[int, int] | None = None,
    caption_override: str | None = None,
):
    draw_page_chrome(c, page_num, total, tag=tag)

    # Header
    c.setFillColor(INK)
    c.setFont(BOLD, 22)
    c.drawString(MARGIN, PAGE_H - MARGIN - 22 * mm, title)

    # Subtitle: TZ point reference (if set and different from title)
    if case.tz_point and case.capability.strip().lower() != title.strip().lower():
        c.setFillColor(INK_SOFT)
        c.setFont(BODY, 10.5)
        c.drawString(MARGIN, PAGE_H - MARGIN - 29 * mm, f"ТЗ №{case.tz_point}: {case.capability}")

    # Chat container
    chat_top = PAGE_H - MARGIN - 35 * mm
    chat_bottom = MARGIN + 20 * mm
    chat_left = MARGIN
    chat_right = PAGE_W - MARGIN
    chat_w = chat_right - chat_left
    bubble_w = chat_w - 10 * mm

    # Pick turns
    if turns_subset:
        start, end = turns_subset
        turns = case.turns[start:end]
    else:
        turns = case.turns

    y = chat_top
    for turn in turns:
        user = turn.get("user", "")
        bot = turn.get("bot", "")
        handoff = turn.get("handoff", False)

        # patient bubble
        y = draw_bubble(c, "patient", user, chat_left + 2 * mm, y, bubble_w)

        # bot bubble — with trim for display
        shown, truncated = _trim_for_display(bot, max_bubble_chars)
        if handoff:
            shown = shown + "\n\n⤷ handoff: передача оператору"
        y = draw_bubble(c, "bot", shown, chat_left + 2 * mm, y, bubble_w)
        if truncated:
            c.setFillColor(MUTED)
            c.setFont(ITAL, 8)
            c.drawString(chat_left + 2 * mm, y - 1 * mm,
                         "(ответ сокращён для слайда; бот возвращает полный текст)")
            y -= 4 * mm

        # small gap between turns
        y -= 3 * mm

        if y < chat_bottom:
            break

    # Caption strip at bottom
    cap_text = caption_override if caption_override is not None else case.caption
    strip_y = MARGIN + 4 * mm
    strip_h = 12 * mm
    c.setFillColor(GREEN_SOFT)
    c.roundRect(MARGIN, strip_y, PAGE_W - 2 * MARGIN, strip_h, 2 * mm, fill=True, stroke=False)
    # vector check
    draw_check(c, MARGIN + 3.5 * mm, strip_y + strip_h / 2 - 2 * mm, size=4 * mm)
    cap_style = ParagraphStyle(
        "cap", fontName=BOLD, fontSize=10, textColor=HexColor("#046B3F"), leading=13,
    )
    draw_paragraph(c, cap_text, cap_style,
                   MARGIN + 11 * mm, strip_y + strip_h - 3 * mm,
                   PAGE_W - 2 * MARGIN - 15 * mm)


# =============================================================================
#  FINAL SLIDE — NUMBERS
# =============================================================================

def slide_numbers(c, page_num, total):
    draw_page_chrome(c, page_num, total, tag="итоги")

    c.setFillColor(INK)
    c.setFont(BOLD, 26)
    c.drawString(MARGIN, PAGE_H - MARGIN - 24 * mm, "Надёжность — в цифрах")

    sub_style = ParagraphStyle(
        "sub", fontName=BODY, fontSize=12, textColor=INK_SOFT, leading=16,
    )
    draw_paragraph(
        c,
        "За 7 недель (3 марта → 22 апреля 2026) бот получил тестовую инфраструктуру и стабильность.",
        sub_style, MARGIN, PAGE_H - MARGIN - 28 * mm, 220 * mm,
    )

    metrics = [
        ("Автоматических проверок поведения бота", "0", "653"),
        ("Эталонных диалогов в регрессионной корзине", "≈10", "62"),
        ("Отдельных доменных модулей кода", "1", "14"),
        ("Общий кэш (колл-центр ↔ мессенджеры)", "—", "CHECK"),
        ("Fallback при сбое внешнего API", "частично", "всегда"),
    ]

    grid_top = PAGE_H - MARGIN - 45 * mm
    row_h = 16 * mm
    col_label_w = 110 * mm
    col_before_w = 40 * mm
    col_after_w = 40 * mm

    # arrow drawing helper
    def _draw_arrow(cx, cy, length=8 * mm):
        c.setStrokeColor(MUTED)
        c.setLineWidth(1.2)
        c.setLineCap(1)
        c.line(cx, cy, cx + length, cy)
        # head
        head = 1.8 * mm
        c.line(cx + length, cy, cx + length - head, cy + head * 0.7)
        c.line(cx + length, cy, cx + length - head, cy - head * 0.7)

    for i, (label, before, after) in enumerate(metrics):
        y = grid_top - i * row_h
        center_y = y - row_h / 2 + 1 * mm
        draw_card(c, MARGIN, y - row_h + 3 * mm, col_label_w + col_before_w + col_after_w + 10 * mm,
                  row_h - 3 * mm, accent=None)
        c.setFillColor(INK)
        c.setFont(BODY, 11)
        c.drawString(MARGIN + 6 * mm, center_y, label)
        # before
        c.setFillColor(MUTED)
        c.setFont(BODY, 14)
        c.drawString(MARGIN + col_label_w, center_y, before)
        # arrow (vector)
        _draw_arrow(MARGIN + col_label_w + col_before_w - 10 * mm, center_y + 1.5 * mm, length=8 * mm)
        # after
        if after == "CHECK":
            draw_check(c, MARGIN + col_label_w + col_before_w, center_y - 1 * mm, size=6 * mm)
        else:
            c.setFillColor(GREEN if after in ("всегда",) or after.isdigit() else BLUE)
            c.setFont(BOLD, 18)
            c.drawString(MARGIN + col_label_w + col_before_w, center_y, after)

    # Bottom caption
    cap_y = MARGIN + 10 * mm
    c.setFillColor(BLUE_SOFT)
    c.roundRect(MARGIN, cap_y, PAGE_W - 2 * MARGIN, 20 * mm, 2 * mm, fill=True, stroke=False)
    c.setFillColor(BLUE)
    c.setFont(BOLD, 11)
    c.drawString(MARGIN + 5 * mm, cap_y + 13 * mm, "Что это значит на практике")
    c.setFillColor(INK)
    c.setFont(BODY, 10)
    c.drawString(MARGIN + 5 * mm, cap_y + 7.5 * mm,
                 "Каждое изменение в боте автоматически проверяется 653 тестами и 62 эталонными диалогами.")
    c.drawString(MARGIN + 5 * mm, cap_y + 3 * mm,
                 "Когда внешние системы клиники недоступны — бот отдаёт последний корректный ответ, а не ошибку.")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    cases = {c.case_id: c for c in _load_cases()}

    # Slide sequence. Each entry is (builder, args).
    slides: list[tuple] = [
        ("title", {}),
        ("overview", {}),
        ("scenario", dict(
            tag="подготовка", title="Подготовка к процедуре",
            case=cases["PREPARE_BIOPSY"], max_bubble_chars=900,
        )),
        ("scenario", dict(
            tag="подготовка", title="Уточнение темы одним словом",
            case=cases["PREPARE_VULVOSCOPY"], max_bubble_chars=560,
        )),
        ("scenario", dict(
            tag="стоимость", title="Стоимость приёма — кардиолог",
            case=cases["PRICE_CONSULT_CARDIO"], max_bubble_chars=760,
        )),
        ("scenario", dict(
            tag="стоимость", title="Стоимость приёма — уролог",
            case=cases["PRICE_UROLOG"], max_bubble_chars=760,
        )),
        ("scenario", dict(
            tag="стоимость", title="Модификатор прайса — «срочно»",
            case=cases["PRICE_CITO"], max_bubble_chars=900,
        )),
        ("scenario", dict(
            tag="специальность", title="Врачи специальности",
            case=cases["SPEC_RATING"], max_bubble_chars=1100,
        )),
        ("scenario", dict(
            tag="специальность", title="Связывание процедуры со специалистом",
            case=cases["PROC_TO_SPECIALTY"], max_bubble_chars=1100,
        )),
        ("scenario", dict(
            tag="филиал", title="Филиал под конкретную процедуру",
            case=cases["ADDRESS_TONSILLECTOMY"], max_bubble_chars=360,
        )),
        ("scenario", dict(
            tag="запись", title="Запись к врачу — шаги 1–2",
            case=cases["APPOINTMENT_FULL"], max_bubble_chars=720,
            turns_subset=(0, 2),
            caption_override="Бот находит кардиологов по запросу, затем уточняет выбранного и показывает расписание.",
        )),
        ("scenario", dict(
            tag="запись", title="Запись к врачу — шаги 3–5",
            case=cases["APPOINTMENT_FULL"], max_bubble_chars=700,
            turns_subset=(2, 5),
            caption_override="Бот удерживает врача и слот, собирает ФИО, готовит карточку записи и передаёт оператору.",
        )),
        ("scenario", dict(
            tag="налоговая", title="Справка в налоговую",
            case=cases["TAX_CERTIFICATE"], max_bubble_chars=400,
        )),
        ("scenario", dict(
            tag="вне профиля", title="Услуга, которой нет в клинике",
            case=cases["STOP_OUT_OF_SCOPE"], max_bubble_chars=300,
        )),
        ("numbers", {}),
    ]

    total_pages = len(slides)
    out_path = Path("docs/NEIRY_BOT_DEMO.pdf")
    out_path.parent.mkdir(exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=landscape(A4))

    for idx, (kind, kwargs) in enumerate(slides, start=1):
        if kind == "title":
            slide_title(c, idx, total_pages)
        elif kind == "overview":
            slide_overview(c, idx, total_pages)
        elif kind == "scenario":
            slide_scenario(c, idx, total_pages, **kwargs)
        elif kind == "numbers":
            slide_numbers(c, idx, total_pages)
        c.showPage()

    c.save()
    print(f"wrote {out_path}  ({total_pages} страниц)")


if __name__ == "__main__":
    main()
