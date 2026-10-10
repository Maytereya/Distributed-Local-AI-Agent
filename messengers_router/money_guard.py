"""Свободный текст LLM не называет денежных сумм (пункт 1 принципов, решение владельца 08.10).

Цену пациенту печатает код из прайса МИС (prebuilt-ответы цен). Свободная генерация
(`orchestrator._render_impl`, ветка `llm_free`) пишет суммы сама: стенд 10.10 — 14 ходов из
1118, в 12 число скопировано из данных хода под пересказанным названием услуги (цена чужой
строки), в 2 числа в данных нет вовсе («скидка 7%», «сертификат на 10 000»). Промпт
`renderer_patient` запрещает суммы — это первая линия; здесь вторая, детерминированная.

Механика: предложение с суммой убирается целиком (частичная редакция «стоит … руб.» даёт
обрубки), заголовок списка, от которого не осталось пунктов, — тоже. Если что-то убрано,
дописывается готовый оффер ценового пути (`CLARIFY_TEXT_MAP["PRICE"]`: «Скажите, пожалуйста,
название услуги/анализа — я уточню стоимость.»). Ответ пациента уйдёт в прайс, и цену
напечатает код.

Применяется ТОЛЬКО к ветке `llm_free`: prebuilt-ответы цен (шаблоны поверх прайса) суммы
сохраняют.
"""

from __future__ import annotations

import re

# Валюта: «₽», «руб», «руб.», «рубль/рубля/рублей/рублях…», «390 р.». «рубеж», «рубрика» — нет.
_CURRENCY_RE = re.compile(r"₽|\bруб(?:л[ьяеиюё]\w*)?\b|\d\s*р\.(?![а-яё\w])", re.IGNORECASE)
# Денежное слово, за которым в том же предложении вплотную идёт число: «стоимость — 1515»,
# «скидка 7%», «сертификат на 10 000». Окно — 20 знаков и не переходит точку: номер дома или
# срок в соседнем предложении («…их стоимость. Адрес: ул. Мира, 5») — не сумма. «стоит/стоят»
# не входят: «стоит прийти за 15 минут».
_PRICE_WORD_NUMBER_RE = re.compile(
    r"\b(?:стоимост|цен[аеуыо]|скидк|сертификат)\w*[^\d.!?\n]{0,20}\d", re.IGNORECASE
)
# Граница предложения: знак конца, пробел и заглавная (или «, ", (, ** перед ней). «390 руб.
# за приём» не режется (дальше строчная), «г. Город», «пр. Мира» — тоже (сокращения адреса),
# как и «1. ОАК» — номер пункта списка в начале строки.
_BOUNDARY_RE = re.compile(r"[.!?…]+\s+(?=[«\"(*]*[A-ZА-ЯЁ])")
_LAST_TOKEN_RE = re.compile(r"(\S+)$")
_ADDRESS_ABBREVIATIONS = frozenset(
    {"г", "гор", "пр", "просп", "ул", "д", "им", "корп", "стр", "пер", "наб", "ш", "пл", "мкр", "обл", "кв", "каб", "т"}
)
_LIST_NUMBER_RE = re.compile(r"\s*\d+")
_LIST_MARKER_ONLY_RE = re.compile(r"^\s*(?:\d+[.)]|[•\-—*])?\s*$")


def has_money(sentence: str) -> bool:
    """Есть ли в предложении денежная сумма (валюта либо денежное слово рядом с числом)."""

    return bool(_CURRENCY_RE.search(sentence) or _PRICE_WORD_NUMBER_RE.search(sentence))


def _sentences(line: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(line):
        last = _LAST_TOKEN_RE.search(line[start:match.start()])
        token = (last.group(1) if last else "").lower()
        if token in _ADDRESS_ABBREVIATIONS:
            continue
        if start == 0 and _LIST_NUMBER_RE.fullmatch(line[: match.start()]):
            continue
        parts.append(line[start:match.end()])
        start = match.end()
    parts.append(line[start:])
    return parts


def _strip_line(line: str) -> tuple[str, int]:
    parts = _sentences(line)
    kept = [part for part in parts if not has_money(part)]
    removed = len(parts) - len(kept)
    if not removed:
        return line, 0
    text = " ".join(part.strip() for part in kept).strip()
    if _LIST_MARKER_ONLY_RE.match(text):
        text = ""
    indent = line[: len(line) - len(line.lstrip())]
    return (indent + text) if text else "", removed


def strip_money(text: str | None) -> tuple[str, int]:
    """Убрать из свободного текста LLM предложения с денежными суммами.

    :param text: полный текст ответа свободной генерации
    :return: (текст без сумм, сколько предложений убрано)
    """

    if not text:
        return text or "", 0
    lines = text.split("\n")
    out: list[str | None] = []  # None — строка ушла целиком
    removed_total = 0
    for line in lines:
        new_line, removed = _strip_line(line)
        removed_total += removed
        out.append(None if removed and not new_line else new_line)
    if not removed_total:
        return text, 0

    # Заголовок списка («Цены:»), от которого не осталось ни одного пункта.
    for i, line in enumerate(out):
        if line is None or not line.rstrip().endswith(":"):
            continue
        block: list[str | None] = []
        for nxt in out[i + 1:]:
            if nxt is not None and not nxt.strip():
                if block:
                    break
                continue  # пустые строки между заголовком и списком
            block.append(nxt)
        if block and all(item is None for item in block):
            out[i] = None

    result = "\n".join(line for line in out if line is not None)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result, removed_total


def guard_free_text(text: str | None) -> tuple[str, int]:
    """Ответ свободной генерации без денежных сумм; если суммы были — оффер ценового пути.

    :param text: полный текст ответа ветки `llm_free`
    :return: (текст пациенту, сколько предложений с суммами убрано)
    """

    from .policies import CLARIFY_TEXT_MAP

    stripped, removed = strip_money(text)
    if not removed:
        return stripped, 0
    offer = CLARIFY_TEXT_MAP["PRICE"]
    if offer in stripped:
        return stripped, removed
    return (f"{stripped}\n\n{offer}" if stripped else offer), removed
