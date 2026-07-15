"""Собирает один слайд-PDF A4 landscape на основе docs/RESULTS_FOR_CLIENT.md.

Контент зашит в скрипт — если меняется MD, меняйте и здесь (файл небольшой).
"""
from __future__ import annotations

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas


# --- Fonts (Cyrillic) ------------------------------------------------
pdfmetrics.registerFont(TTFont("Body", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-B", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))

BODY = "Body"
BOLD = "Body-B"

# --- Palette ---------------------------------------------------------
INK = HexColor("#0F172A")       # near-black text
MUTED = HexColor("#64748B")     # grey secondary
ACCENT = HexColor("#0369A1")    # deep blue accent
CHECK = HexColor("#059669")     # green check
CHIP_BG = HexColor("#F1F5F9")   # light panel
RULE = HexColor("#CBD5E1")      # separator
WARN = HexColor("#92400E")      # amber warning


def draw_slide(path: str) -> None:
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(path, pagesize=landscape(A4))

    margin = 14 * mm
    top = page_h - margin

    # ---- Header strip ------------------------------------------------
    c.setFillColor(ACCENT)
    c.rect(0, top - 1 * mm, page_w, 12 * mm, fill=True, stroke=False)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(BOLD, 18)
    c.drawString(margin, top + 2.5 * mm, "Чат-бот клиники — что изменилось")
    c.setFont(BODY, 11)
    c.drawRightString(page_w - margin, top + 2.5 * mm, "3 марта → 22 апреля 2026  ·  7 недель работы")

    y = top - 5 * mm

    # ---- Section 1: 8 TЗ points (2-column grid) ---------------------
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(margin, y - 6 * mm, "Все 8 пунктов ТЗ — работают")
    y_grid_top = y - 10 * mm

    items = [
        ("Результаты анализов",
         "Отдаёт ссылку на результат; при сбое — аккуратный handoff к оператору"),
        ("Стоимость анализов",
         "Различает варианты, понимает «cito», «на дому», «детский», списки услуг"),
        ("Услуга + 4 врача + подготовка",
         "Врачи по рейтингу, правила подготовки, «как готовиться» vs «где сделать»"),
        ("Информация о враче по ФИО",
         "Удерживает имя врача на всём диалоге, подтягивает розничный прайс"),
        ("Врачи специальности",
         "«Кто делает уретроскопию?» → уролог; составные спец. (травматолог-ортопед)"),
        ("Расписание врача",
         "Различает «врача нет» и «нет свободных слотов»; работает при сбоях API"),
        ("Запись к врачу",
         "Выбор времени → ФИО пациента → подтверждение → оператор"),
        ("Справка в налоговую",
         "Готовая инструкция текстом, не переключает на оператора без повода"),
    ]

    # Build a 4-row × 2-col table of cells
    col_w = (page_w - 2 * margin - 6 * mm) / 2
    cell_h = 14 * mm
    x0 = margin
    x1 = margin + col_w + 6 * mm

    check_style = ParagraphStyle(
        "check", fontName=BOLD, fontSize=10, textColor=CHECK, leading=12,
    )
    title_style = ParagraphStyle(
        "title", fontName=BOLD, fontSize=10.5, textColor=INK, leading=12,
    )
    body_style = ParagraphStyle(
        "body", fontName=BODY, fontSize=9, textColor=MUTED, leading=11,
    )

    y_cell = y_grid_top
    for idx, (title, sub) in enumerate(items):
        col = idx % 2
        row = idx // 2
        cy = y_grid_top - row * cell_h
        cx = x0 if col == 0 else x1

        # chip background
        c.setFillColor(CHIP_BG)
        c.roundRect(cx, cy - cell_h + 2 * mm, col_w, cell_h - 2 * mm, 2 * mm, fill=True, stroke=False)

        # check mark
        c.setFillColor(CHECK)
        c.setFont(BOLD, 13)
        c.drawString(cx + 3 * mm, cy - 5.5 * mm, "✓")

        # title
        c.setFillColor(INK)
        c.setFont(BOLD, 10.5)
        c.drawString(cx + 9 * mm, cy - 5 * mm, title)

        # sub
        p = Paragraph(sub, body_style)
        p.wrapOn(c, col_w - 12 * mm, 10 * mm)
        p.drawOn(c, cx + 9 * mm, cy - 11.5 * mm)

    y = y_grid_top - 4 * cell_h - 2 * mm

    # ---- Divider ----
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.line(margin, y, page_w - margin, y)
    y -= 6 * mm

    # ---- Bottom row: 2 columns (Меньше ошибок | Цифры) ----
    col_w = (page_w - 2 * margin - 10 * mm) / 2
    left_x = margin
    right_x = margin + col_w + 10 * mm
    bottom_y = 14 * mm

    # LEFT column: live scenarios — what the bot does now
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(left_x, y, "Живые сценарии — что бот делает сейчас")

    scenarios = [
        ("Различает смысл похожих запросов об одной услуге",
         "«Как подготовиться к УЗИ щитовидной железы?»  →  правила подготовки\n"
         "«Сколько стоит УЗИ щитовидной железы?»  →  позиции прайса\n"
         "«Где сделать УЗИ щитовидной железы?»  →  адреса филиалов"),
        ("Связывает процедуру со специальностью",
         "«Кто делает уретроскопию?»  →  уролог\n"
         "«Кто снимает ЭКГ?»  →  кардиолог\n"
         "Понимает составные: «травматолог-ортопед», синонимы: «ФГДС» = «гастроскопия»"),
        ("Ведёт запись от первой реплики до подтверждения",
         "«Запишите к Хальметовой на завтра на 12:00»  →  расписание  →  запрос ФИО\n"
         "пациента  →  карточка записи  →  передача оператору.\n"
         "Имя врача, дата и время удерживаются до конца диалога."),
        ("Отказывает вежливо на запросы вне профиля клиники",
         "«Шунтирование желудка», «справка в ГИБДД», «приём по ДМС» — бот\n"
         "сообщает, что это вне его компетенции, и переключает на оператора\n"
         "вместо попытки угадать ответ."),
    ]

    ey = y - 5 * mm
    sub_style = ParagraphStyle(
        "sub", fontName=BODY, fontSize=8.5, textColor=MUTED, leading=10.5,
    )
    q_style = ParagraphStyle(
        "q", fontName=BOLD, fontSize=9.5, textColor=INK, leading=11.5,
    )
    cell_gap = 17 * mm
    for title, body in scenarios:
        pq = Paragraph(title, q_style)
        pq.wrapOn(c, col_w, 6 * mm)
        pq.drawOn(c, left_x, ey - 4 * mm)
        # body with line breaks — convert newlines to <br/>
        body_html = body.replace("\n", "<br/>")
        ps = Paragraph(body_html, sub_style)
        ps.wrapOn(c, col_w, 14 * mm)
        ps.drawOn(c, left_x, ey - 15 * mm)
        ey -= cell_gap

    c.setFillColor(MUTED)
    c.setFont(BODY, 8.5)
    c.drawString(
        left_x, ey + 3 * mm,
        "+ понимает модификаторы прайса: «срочно (cito)», «на дому», «детский», пакеты и комплексы"
    )

    # RIGHT column: numbers table
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(right_x, y, "В цифрах")

    num_data = [
        ["Параметр", "Было 3 марта", "Сейчас"],
        ["Автотестов поведения бота", "0", "653"],
        ["Эталонных диалогов (регрессия)", "≈10", "62"],
        ["Доменных модулей", "1 (монолит)", "14"],
        ["Fallback при сбое внешнего API", "частично", "всегда"],
        ["Общий кэш (колл-центр ↔ мессенджеры)", "—", "есть"],
    ]
    t = Table(num_data, colWidths=[col_w * 0.52, col_w * 0.23, col_w * 0.25])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), BOLD),
        ("FONTNAME", (0, 1), (-1, -1), BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 1), (0, -1), INK),
        ("TEXTCOLOR", (1, 1), (1, -1), MUTED),
        ("TEXTCOLOR", (2, 1), (2, -1), CHECK),
        ("FONTNAME", (2, 1), (2, -1), BOLD),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), CHIP_BG]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.3, HexColor("#FFFFFF")),
    ]))
    tw, th = t.wrapOn(c, col_w, 40 * mm)
    t.drawOn(c, right_x, y - 5 * mm - th)

    # ---- Footer -----------------------------------------------------
    c.setStrokeColor(RULE)
    c.setLineWidth(0.3)
    c.line(margin, bottom_y + 2 * mm, page_w - margin, bottom_y + 2 * mm)
    c.setFillColor(WARN)
    c.setFont(BOLD, 9)
    c.drawString(margin, bottom_y - 3 * mm, "Открытый хвост:")
    c.setFillColor(MUTED)
    c.setFont(BODY, 9)
    c.drawString(margin + 32 * mm, bottom_y - 3 * mm,
                 "3 сценария по прайсу на сложных комбинациях услуг — известны, в плане следующего цикла.")

    c.showPage()
    c.save()


if __name__ == "__main__":
    out = "docs/RESULTS_FOR_CLIENT.pdf"
    draw_slide(out)
    print(f"wrote {out}")
