"""Реквизиты результатов анализов без догадок по любому слову (BUG-2026-10-24-RESULT-SLOT-COMMAND).

Ссылку на результат бот строит по четырём реквизитам: фамилия, год рождения, код анализа (в состоянии — `filial`, на сайте — `nom`),
номер анализа. Прежнее быстрое заполнение при недостающем коде брало кодом любые 1–3 слова реплики и город: «Сидорова» становилась и
фамилией, и кодом, «Результаты анализов» — кодом (класс «слот берёт команду как данные», отложенный сиблинг
FEAT-2026-07-15-SERVICE-SLOT-LLM). Справочника кодов в МИС нет — проверить код по данным нечем. Под
`[MESSENGER_ROUTER] result_slots_llm = on`:

- **код анализа** — форма (полный порядок «фамилия, год, код, номер»; «код Бг») или сущность `filial` LLM-классификатора этого хода,
  и только если это слова реплики подряд (модель код не придумала) и не слово другого реквизита хода (одно слово — один реквизит).
  LLM не звали, сбой, кода не назвала — кода нет: бот переспросит (решение владельца 22.10 (а)).

off — как было (`policies._guess_result_code`).
"""

from __future__ import annotations

import re
from typing import Any

from . import turn_telemetry
from .runtime_config import config as _cfg

# Реквизиты хода, чьи слова кодом быть не могут: одно слово реплики — один реквизит.
_OTHER_FIELDS = ("surname", "patient_name", "year", "number")


def on() -> bool:
    """`[MESSENGER_ROUTER] result_slots_llm = on`."""

    return str(getattr(_cfg, "MR_RESULT_SLOTS_LLM", "off") or "off").strip().lower() == "on"


def _words(text: str) -> tuple[str, ...]:
    """Слова без регистра, «ё» и знаков (как в памяти имён хода, `turn_names`)."""

    low = str(text or "").casefold().replace("ё", "е")
    spaced = "".join(ch if ch.isalnum() or ch == "-" else " " for ch in low)
    return tuple(w.strip("-") for w in spaced.split() if w.strip("-"))


def _contains(reply_words: tuple[str, ...], part: tuple[str, ...]) -> bool:
    n = len(part)
    return n > 0 and any(reply_words[i : i + n] == part for i in range(len(reply_words) - n + 1))


def _acted(outcome: str) -> None:
    """Журнал хода `result_slots_acted` — что решил путь `on`."""

    turn_telemetry.append("result_slots_acted", outcome)


def _after_marker(text: str) -> str:
    """Значение формы «код Бг», «Код анализа: бг», «мой код — ву 276»: одно слово с буквы, стоящее отдельно (дальше конец, знак или
    число); «код не помню» — не форма. Цифровой «код 5437» — код услуги (цена, BUG-2026-06-09-07), не код анализа."""

    found = re.search(
        r"\bкод\b(?:\s+анализа\b)?[\s:—–\-№#]*(?!анализ)([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-]{0,19})(?=\s*$|\s*[,;.!?)]|\s+\d)",
        str(text or ""),
        re.I,
    )
    return found.group(1) if found else ""


def marked_code(text: str) -> str:
    """Код анализа из формы «код Бг» (маркер — слово формы, не значение); формы нет — пусто."""

    code = _after_marker(text)
    if code:
        _acted("code:marker")
    return code


def llm_code(value: Any, text: str, entities: dict[str, Any]) -> str:
    """Код анализа из сущности `filial` LLM-классификатора — слова реплики подряд, не слово другого реквизита; иначе пусто."""

    raw = str(value or "").strip(" ,.")
    if not raw:
        return ""
    raw = _after_marker(raw) or raw
    words = _words(raw)
    if not _contains(_words(text), words):
        _acted("code:llm_not_in_reply")
        return ""
    taken = {w for key in _OTHER_FIELDS for w in _words(str(entities.get(key) or ""))}
    if taken.intersection(words):
        _acted("code:llm_other_field")
        return ""
    _acted("code:llm")
    return raw
