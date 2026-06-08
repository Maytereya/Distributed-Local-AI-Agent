"""Multi-turn regression для Stage 3 APPOINTMENT flow.

Эмулирует последовательность сообщений пациента по записи/переносу
и проверяет, что бот проходит шаги ветки без зацикливания.
Ответственность скрипта: проверять устойчивость многошагового APPOINTMENT-flow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field


@dataclass
class Step:
    user_text: str
    expect_any: tuple[str, ...]
    expect_handoff: bool
    reject_any: tuple[str, ...] = field(default_factory=tuple)
    # Динамический ввод: если задан, user_text вычисляется из LIVE-данных в рантайме
    # (например, реальный свободный слот из расписания), чтобы шаг не флакал на
    # изменчивом CRM. Сигнатура: derive_text(url, post_json, session_id) -> str.
    # Любое падение/пустой результат → fallback на статический user_text.
    derive_text: Callable[[str, Callable[..., dict], str], str] | None = None


@dataclass
class Flow:
    flow_id: str
    steps: list[Step]


def _derive_trubin_lenina_slot(url: str, poster: Callable[..., dict], session_id: str) -> str:
    """Динамический шаг 5 HOLTER_RESCHEDULE: подставляет РЕАЛЬНЫЙ свободный слот
    Трубина на пр.Ленина, 5 из live-расписания.

    Раньше шаг хардкодил «завтра после 16:00» и флакал на parity смен (у Ленина-5
    чередуются утренние/вечерние приёмы → «завтра» могло быть без слота после 16:00).
    Читаем актуальное расписание отдельным запросом и берём первый свободный слот в
    секции этого филиала. Падение/нет слота → fallback на исходный хардкод.
    """
    probe = poster(
        url,
        {"session_id": f"{session_id}_slotprobe", "text": "расписание Трубина", "debug": True},
    )
    bot = str(probe.get("text") or "")
    # Якоримся на ЗАГОЛОВОК секции филиала «...Ленина, 5:» (с двоеточием — он
    # уникален; в строке «Адреса приема: …Ленина, 5» двоеточия после адреса нет),
    # иначе нашли бы слот соседнего филиала (Победы 83), указанного выше по тексту.
    anchor = bot.find("Ленина, 5:")
    region_text = bot[anchor:] if anchor >= 0 else bot
    m = re.search(r"•\s*(\d{1,2}\s+[а-яё]+)\s*:\s*свободно в\s*(\d{1,2}:\d{2})", region_text)
    if m:
        return f"{m.group(1).strip()} {m.group(2)}"
    return "завтра после 16:00"


FLOWS: list[Flow] = [
    Flow(
        "APPT_DOCTOR_STD",
        steps=[
            Step("расписание Дразнин", ("свободно", "если нужно записаться"), False),
            Step("20 марта, 16:30", ("фио пациента",), False),
            Step("Иванов Иван Иванович", ("подтверждаете", "запись:"), False),
            Step("да", ("передаю заявку оператору", "соединяю с оператором"), True),
        ],
    ),
    Flow(
        "APPT_HOLTER_RESCHEDULE",
        steps=[
            Step("Можно перенести запись на холтер?", ("фио врача", "нужно перенести"), False),
            Step(
                "Трубин",
                ("фио пациента",),
                False,
            ),
            Step(
                "Петров Петр Петрович",
                ("по адресам", "какой филиал", "филиал вам удобен"),
                False,
                reject_any=("в городе самара доступны филиалы", "телефон:"),
            ),
            # Филиал обновлён 2026-06-08 по live find_doctor_schedule('Трубин'):
            # regions=["ул. Победы, 83", "пр.Ленина, 5"] (Победы-83 ВЕРНУЛСЯ в CRM,
            # «Гагарина 12» неактуальна). ⚠️ Хардкод филиала+времени фрагилен к live-CRM
            # (рецидив 2-й раз) — кандидат на ДИНАМИЧЕСКИЙ шаг (брать филиал/слот из офера
            # бота). Пока берём пр.Ленина, 5.
            Step("пр.Ленина, 5", ("дату и время", "на какую дату", "удобное время"), False),
            # Динамический шаг: реальный свободный слот Трубина на Ленина-5 (не хардкод
            # «завтра после 16:00», который флакал на parity смен). Fallback внутри derive.
            Step(
                "завтра после 16:00",
                ("подтверждаете", "запись:"),
                False,
                derive_text=_derive_trubin_lenina_slot,
            ),
            Step("нет", ("уточните новую дату",), False),
        ],
    ),
    Flow(
        "APPT_HARD_RESET_NE_TUDA",
        steps=[
            Step(
                "Хальметова расписание",
                ("расписан", "если нужно записаться"),
                False,
            ),
            Step(
                "на завтра на 12:00",
                ("фио пациента", "ваше фио"),
                False,
            ),
            Step("не туда", ("отменить текущий процесс записи", "ответьте «да» или «нет»"), False),
            Step("да", ("процесс записи отменён", "процесс отмены записи отменён", "процесс переноса записи отменён"), False),
            Step(
                "Какая стоимость приема у кардиолога?",
                ("кардиолог", "стоим"),
                False,
                reject_any=("сейчас идет оформление записи", "ответьте «да» или «нет»"),
            ),
        ],
    ),
    Flow(
        "APPT_HARD_RESET_RUDE",
        steps=[
            Step(
                "Хальметова расписание",
                ("расписан", "если нужно записаться"),
                False,
            ),
            Step(
                "на завтра на 12:00",
                ("фио пациента", "ваше фио"),
                False,
            ),
            Step("ты несешь бред", ("отменить текущий процесс записи", "ответьте «да» или «нет»"), False),
            Step("да", ("процесс записи отменён", "процесс отмены записи отменён", "процесс переноса записи отменён"), False),
            Step(
                "Какая стоимость приема у кардиолога?",
                ("кардиолог", "стоим"),
                False,
                reject_any=("сейчас идет оформление записи", "ответьте «да» или «нет»"),
            ),
        ],
    ),
]


def post_json(url: str, payload: dict, retries: int = 1) -> dict:
    last_exc: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url=url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=50) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("post_json failed without exception")


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    for p in patterns:
        if re.search(re.escape(p.lower()), t):
            return True
    return False


def _contains_none(text: str, patterns: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    for p in patterns:
        if re.search(re.escape(p.lower()), t):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate stage-3 APPOINTMENT multi-turn flow")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/messenger-generate-once",
        help="Debug endpoint URL",
    )
    parser.add_argument(
        "--session-prefix",
        default="s_eval_stage3_appt",
        help="Session prefix for requests",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id to isolate sessions across repeated launches. Default: current unix timestamp.",
    )
    args = parser.parse_args()
    run_id = args.run_id.strip() or str(int(time.time()))

    total = 0
    passed = 0

    print(f"Evaluating {len(FLOWS)} APPOINTMENT flows against {args.url}")
    print(f"Run id: {run_id}")
    print("-" * 140)
    print(f"{'Flow':<24} {'Step':<4} {'Handoff':<9} {'Pass':<5} {'BOT (trim)'}")
    print("-" * 140)

    for flow in FLOWS:
        session_id = f"{args.session_prefix}_{run_id}_{flow.flow_id}"
        for idx, step in enumerate(flow.steps, 1):
            total += 1
            user_text = step.user_text
            if step.derive_text is not None:
                try:
                    derived = step.derive_text(args.url, post_json, session_id)
                    if derived:
                        user_text = derived
                except Exception:
                    pass  # любое падение derive → fallback на статический user_text
            payload = {"session_id": session_id, "text": user_text, "debug": True}
            try:
                data = post_json(args.url, payload)
                bot_text = str(data.get("text") or "")
                handoff = bool(data.get("handoff", False))
                ok = (
                    _contains_any(bot_text, step.expect_any)
                    and _contains_none(bot_text, step.reject_any)
                    and handoff == step.expect_handoff
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                bot_text = f"ERROR:{type(e).__name__}"
                handoff = False
                ok = False

            if ok:
                passed += 1

            trimmed = re.sub(r"\s+", " ", bot_text).strip()[:80]
            print(f"{flow.flow_id:<24} {idx:<4} {str(handoff):<9} {str(ok):<5} {trimmed}")

    pct = (passed / total) * 100 if total else 0.0
    print("-" * 140)
    print(f"Passed: {passed}/{total} ({pct:.1f}%)")
    print("Target: >= 90% for stage-3 flow stability")

    return 0 if pct >= 90.0 else 2


if __name__ == "__main__":
    sys.exit(main())
