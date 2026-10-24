"""Реквизиты результатов анализов без догадок по любому слову (BUG-2026-10-24-RESULT-SLOT-COMMAND).

Ссылку на результат бот строит по четырём реквизитам: фамилия, год рождения, код анализа (в состоянии — `filial`, на сайте — `nom`),
номер анализа. Прежнее быстрое заполнение при недостающем коде брало кодом любые 1–3 слова реплики и город: «Сидорова» становилась и
фамилией, и кодом, «Результаты анализов» — кодом (класс «слот берёт команду как данные», отложенный сиблинг
FEAT-2026-07-15-SERVICE-SLOT-LLM). Справочника кодов в МИС нет — проверить код по данным нечем. Под
`[MESSENGER_ROUTER] result_slots_llm = on`:

- **код анализа** — форма (полный порядок «фамилия, год, код, номер»; «код Бг») или сущность `filial` LLM-классификатора этого хода,
  и только если это слова реплики подряд (модель код не придумала) и не слово другого реквизита хода (одно слово — один реквизит).
  LLM не звали, сбой, кода не назвала — кода нет: бот переспросит (решение владельца 22.10 (а));
- **одно число — один реквизит**: «1985» было и годом рождения, и номером анализа;
- **переспрос называет недостающее** (пункт 5 принципов): часть реквизитов принята — бот просит только остальные, а не те же четыре
  (тот же вопрос второй раз подряд — сразу оффер оператора, хотя пациент продвинулся); ответ в том же порядке («1985, Бг, 123») —
  остаток формы, как полный порядок.

off — как было (`policies._guess_result_code`).
"""

from __future__ import annotations

import re
from typing import Any

from . import turn_telemetry
from .runtime_config import config as _cfg

# Реквизиты хода, чьи слова кодом быть не могут: одно слово реплики — один реквизит.
_OTHER_FIELDS = ("surname", "patient_name", "year", "number")
# Порядок формы — как в вопросе бота (`policies.clarification_question`): фамилия, год рождения, код анализа, номер анализа.
_FORM = ("surname", "year", "filial", "number")
_ASK = {"surname": "фамилию", "year": "год рождения", "filial": "код анализа", "number": "номер анализа"}
_HAVE = {"surname": "фамилия", "year": "год рождения", "filial": "код анализа", "number": "номер анализа"}
_EXAMPLE = {"surname": "Иванов", "year": "1989", "filial": "Бг", "number": "123"}


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


def one_number_one_field(
    text: str, missing: list[str], out: dict[str, Any], *, year_marked: bool, number_marked: bool
) -> None:
    """Одно число реплики стало и годом, и номером — оставить один реквизит: тот, что назван словом («номер 1985», «год рождения
    1985», «заказ 1985»), иначе тот, которого ждём (год раньше номера); не ждём ни того, ни другого — ни один (не перетирать)."""

    year = out.get("year")
    if year is None or year != out.get("number") or _words(text).count(str(year)) > 1:
        return
    if number_marked != year_marked:
        keep = "number" if number_marked else "year"
    else:
        keep = next((field for field in ("year", "number") if field in missing), "")
    for field in ("year", "number"):
        if field != keep:
            out.pop(field, None)
    _acted(f"number:{keep or 'none'}")


def _fits(field: str, part: str) -> bool:
    """Значение подходит реквизиту по виду — как у полного порядка (`policies._QF_RESULT_ORDERED_RE`); код — с буквы."""

    if field == "year":
        return len(part) == 4 and part.isascii() and part.isdigit() and part[:2] in ("19", "20")
    if field == "number":
        return len(part) >= 3 and part.isascii() and part.isdigit()
    if field == "surname":
        return len(part) >= 2 and all(ch.isalpha() or ch == "-" for ch in part)
    return 0 < len(part) <= 20 and part[0].isalpha() and all(ch.isalnum() or ch == "-" for ch in part)


def rest_of_form(text: str, missing: list[str]) -> dict[str, Any]:
    """Остаток формы в порядке вопроса: два-три недостающих реквизита через запятую или пробел, каждый своего вида, и среди них число
    (год или номер) — «1985, Бг, 123», «ву 276». Голое слово формой не считается: «Бг» или «Сидорова» решает LLM хода."""

    rest = [field for field in _FORM if field in missing]
    if not 2 <= len(rest) <= 3 or not {"year", "number"}.intersection(rest):
        return {}
    parts = str(text or "").replace(",", " ").replace(";", " ").strip(" .!").split()
    if len(parts) != len(rest) or not all(_fits(field, part) for field, part in zip(rest, parts)):
        return {}
    # Подпись графы — не значение: «код 5437», «номер 123» — слово формы и число, а не код и номер.
    labels = {word for name in _HAVE.values() for word in _words(name)}
    if labels.intersection(_words(" ".join(parts))):
        return {}
    _acted("form:rest")
    return {
        field: int(part) if field in ("year", "number") else part.capitalize() if field == "surname" else part
        for field, part in zip(rest, parts)
    }


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]


def ask_rest(missing: list[str]) -> str:
    """Переспрос, когда часть реквизитов уже есть: только недостающие, в порядке формы, с примером; нечего назвать — пусто."""

    rest = [field for field in _FORM if field in missing]
    have = [field for field in _FORM if field not in missing]
    if len(rest) < 2 or not have:
        return ""
    return (
        f"Укажите ещё {_listed([_ASK[f] for f in rest])} — через запятую, в таком порядке. "
        f"Пример: «{', '.join(_EXAMPLE[f] for f in rest)}». {_listed([_HAVE[f] for f in have]).capitalize()} уже есть."
    )
