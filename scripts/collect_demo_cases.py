"""Собирает живые ответы бота для набора презентационных кейсов.

Работает против http://172.16.0.16/api/messenger-generate-once.
Каждый кейс — либо single-turn, либо цепочка с единой session_id.
Результат сохраняется в docs/demo_cases.json для потребителя —
scripts/build_client_slides_v2.py.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

URL = "http://172.16.0.16/api/messenger-generate-once"
TIMEOUT = 45
LLM_MODE = "hybrid"

# Cases to collect — grouped by capability area. Each dict is a self-contained
# demo scenario; `turns` is the list of user messages in order (multi-turn
# scenarios share a session_id so the bot keeps context).
CASES: list[dict[str, Any]] = [
    {
        "case_id": "PREPARE_BIOPSY",
        "capability": "Подготовка к процедуре",
        "tz_point": 3,
        "session_id": "demo_prep_biopsy",
        "turns": ["Как подготовиться к биопсии шейки матки?"],
        "caption": "Бот достаёт медицинскую памятку из базы знаний и выдаёт её пациенту.",
    },
    {
        "case_id": "PREPARE_VULVOSCOPY",
        "capability": "Подготовка к процедуре",
        "tz_point": 3,
        "session_id": "demo_prep_vulv",
        "turns": ["Как подготовиться к вульвоскопии?", "Вульвоскопия"],
        "caption": "На уточняющее сообщение одним словом — тема удержана, выдаётся краткое описание процедуры.",
    },
    {
        "case_id": "PRICE_CONSULT_CARDIO",
        "capability": "Стоимость приёма у специалиста",
        "tz_point": 5,
        "session_id": "demo_price_cardio",
        "turns": ["Сколько стоит приём кардиолога?"],
        "caption": "Розничный прайс и варианты — без случайных «анализов» или «УЗИ», только релевантные позиции.",
    },
    {
        "case_id": "PRICE_UROLOG",
        "capability": "Стоимость приёма у специалиста",
        "tz_point": 5,
        "session_id": "demo_price_urolog",
        "turns": ["Сколько стоит прием уролога?"],
        "caption": "Различает «приём уролога» и «анализы у уролога» — показывает именно розничную стоимость консультации.",
    },
    {
        "case_id": "PRICE_CITO",
        "capability": "Стоимость с модификатором «срочно»",
        "tz_point": 2,
        "session_id": "demo_price_cito",
        "turns": ["Общий анализ крови срочно"],
        "caption": "Бот различает cito-вариант (580 руб., 1–3 часа) и обычные варианты — и показывает все доступные цены и сроки.",
    },
    {
        "case_id": "TAX_CERTIFICATE",
        "capability": "Справка в налоговую",
        "tz_point": 8,
        "session_id": "demo_tax",
        "turns": ["Привет! Получить справку для налоговой"],
        "caption": "Отдаёт готовую текстовую инструкцию — без ненужного переключения на оператора.",
    },
    {
        "case_id": "ADDRESS_TONSILLECTOMY",
        "capability": "Филиалы под конкретную процедуру",
        "tz_point": 3,
        "session_id": "demo_addr_tons",
        "turns": ["Где можно сделать тонзиллэктомию?"],
        "caption": "Понимает, что вопрос про адреса, а не про цену — выдаёт именно филиалы, где процедура доступна.",
    },
    {
        "case_id": "APPOINTMENT_FULL",
        "capability": "Запись к врачу — цепочка без потерь контекста",
        "tz_point": 7,
        "session_id": "demo_appt_full",
        "turns": [
            "Скажите, кто из кардиологов принимает и по какому адресу?",
            "Хальметова, да",
            "2026-04-24 в 12:00!",
            "Рахманов Владимир",
            "да",
        ],
        "caption": "Ведёт диалог: выбор специальности → врача → слота → ФИО пациента → подтверждение → оператор.",
    },
    {
        "case_id": "STOP_OUT_OF_SCOPE",
        "capability": "Корректный отказ на услугу вне профиля",
        "tz_point": 0,
        "session_id": "demo_stop_mri",
        "turns": ["МРТ коленного сустава"],
        "caption": "МРТ в клинике не делают. Бот не выдумывает и сообщает об этом прямо.",
    },
    {
        "case_id": "SPEC_RATING",
        "capability": "Врачи специальности по рейтингу",
        "tz_point": 5,
        "session_id": "demo_spec_rating",
        "turns": ["Кто из кардиологов принимает?"],
        "caption": "Список специалистов с приоритетом по рейтингу и кратким описанием профиля каждого.",
    },
    {
        "case_id": "PROC_TO_SPECIALTY",
        "capability": "Связывание процедуры со специальностью врача",
        "tz_point": 5,
        "session_id": "demo_proc_spec",
        "turns": ["Кто делает уретроскопию?"],
        "caption": "Бот не ищет «уретроскописта» в базе врачей — он понимает, что уретроскопию делает уролог.",
    },
]


def call_bot(session_id: str, text: str) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "text": text,
        "debug": False,
        "llm_mode": LLM_MODE,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"text": f"[ошибка обращения к серверу: {exc}]", "handoff": False}


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "docs" / "demo_cases.json"
    out_path.parent.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in CASES:
        print(f"[{case['case_id']}] collecting {len(case['turns'])} turn(s)…", flush=True)
        turns_out: list[dict[str, Any]] = []
        for text in case["turns"]:
            resp = call_bot(case["session_id"], text)
            turns_out.append(
                {
                    "user": text,
                    "bot": str(resp.get("text") or "").strip(),
                    "handoff": bool(resp.get("handoff", False)),
                }
            )
            time.sleep(0.3)
        results.append({**case, "runs": turns_out})
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nсохранено → {out_path}  ({len(results)} кейсов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
