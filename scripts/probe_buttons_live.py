"""Проверка бота по HTTP: нажатия кнопок меню и обычные вопросы пациентов.

Нужна после деплоя кнопок: убедиться, что кнопки работают, а обычные вопросы
пациентов отвечаются как раньше. Ходит в `/api/messenger-generate-once` с
`debug=true`; каждый запрос — своя сессия, пациентов не задевает.

Нагрузка — около 50 запросов подряд. Не гонять в цикле: прод делит LLM с
пациентами (CLAUDE.md, «Изоляция сети» — зачем гейт герметичен).

    ./venv/bin/python scripts/probe_buttons_live.py --url http://ПРОД/api/messenger-generate-once
    ./venv/bin/python scripts/probe_buttons_live.py --url ... --only buttons --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_button_actions import GATEWAY_V1_TITLES  # noqa: E402 — контракт со шлюзом, версия 1

# Ответ пациента на уточнение кнопки — проверяет, что тема кнопки держится.
FOLLOW_UPS = {
    "menu.price.test": "ферритин",
    "menu.price.doctor": "кардиолог",
    "menu.tests.price": "общий анализ крови",
    "menu.appointment.prepare": "УЗИ брюшной полости",
    "menu.tests.prepare": "ферритин",
    "menu.appointment.book": "кардиолог",
}

# Обычные вопросы пациентов (свип 28.09): до и после деплоя должны совпадать.
TEXT_QUESTIONS = [
    "хочу записаться на приём к врачу", "когда принимает терапевт", "хочу перенести запись",
    "как подготовиться к УЗИ", "где и до скольки можно сдать анализы",
    "нужно ли записываться чтобы сдать кровь", "как подготовиться к анализу крови",
    "какие анализы сдать при усталости", "сколько стоит анализ крови",
    "хочу получить результаты анализов", "результат анализа не пришёл",
    "когда будет готов анализ крови", "сколько стоит ферритин", "сколько стоит приём кардиолога",
    "какие у вас есть акции", "адреса филиалов", "до скольки работаете", "работаете ли в выходные",
    "нужна справка для налогового вычета", "нужна копия договора", "нужен больничный лист",
    "переключите на оператора",
]


def ask(url: str, session_id: str, text: str, button_id: str = "") -> tuple[str, bool, str]:
    payload = {"session_id": session_id, "text": text, "debug": True}
    if button_id:
        payload["button_id"] = button_id
    body = json.dumps(payload).encode()
    last_error = None
    for _ in range(3):  # сеть до сервера бывает нестабильна (VPN)
        try:
            request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            decision = ((data.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            return str(decision.get("label")), bool(data.get("handoff")), data.get("text") or ""
        except Exception as exc:  # noqa: BLE001 — сообщаем и идём дальше
            last_error = exc
            time.sleep(5)
    return f"СБОЙ: {last_error}", False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="…/api/messenger-generate-once")
    parser.add_argument("--only", choices=["buttons", "text"], help="только одна часть проверки")
    parser.add_argument("--limit", type=int, default=0, help="не больше N пунктов в каждой части")
    args = parser.parse_args()
    stamp = int(time.time())
    failures = 0

    if args.only in (None, "buttons"):
        print("=== нажатия кнопок")
        items = list(GATEWAY_V1_TITLES.items())[: args.limit or None]
        for i, (button_id, title) in enumerate(items, 1):
            sid = f"probe_btn_{stamp}_{i}"
            label, handoff, text = ask(args.url, sid, title, button_id)
            failures += label.startswith("СБОЙ")
            print(f"■ {button_id} «{title}» [{label}] оператор={int(handoff)}\n    → {text[:200]!r}")
            if button_id in FOLLOW_UPS:
                label2, handoff2, text2 = ask(args.url, sid, FOLLOW_UPS[button_id])
                failures += label2.startswith("СБОЙ")
                print(f"    ↳ «{FOLLOW_UPS[button_id]}» [{label2}] оператор={int(handoff2)}\n      → {text2[:200]!r}")

    if args.only in (None, "text"):
        print("\n=== обычные вопросы пациентов")
        for i, question in enumerate(TEXT_QUESTIONS[: args.limit or None], 1):
            label, handoff, text = ask(args.url, f"probe_txt_{stamp}_{i}", question)
            failures += label.startswith("СБОЙ")
            print(f"«{question}» [{label}] оператор={int(handoff)} → {text[:120]!r}")

    print(f"\nсбоев сети: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
