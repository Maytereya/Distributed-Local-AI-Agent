"""Цена, этап 2 «без стоп-слов»: читатели цены берут фразу услуги из разбора хода (DESIGN-2026-10-25-PRICE-STAGE2).

Решения владельца 22.10: что в реплике услуга — решает LLM разметки (`_turn_spans`, этап 1, на проде с деплоя 17); 28 списков
стоп-слов удаляются; без LLM понять, какую услугу назвал пациент, бот не пытается — дословное название или код клиники, иначе
«уточните название» или оффер. До этапа 2 читатели цены резали реплику сами вырезателем
(`_prices_helpers._extract_price_service_from_query`, 11 списков), и вырезатель играл четыре роли: фраза услуги, «названа ли
услуга», ключ сильного совпадения, слова поиска. Здесь — один источник фразы на ход: разбор этой реплики.

Выключатель `[MESSENGER_ROUTER] price_by_parse = off | on` — один на читателей П1–П8 (дизайн §5). Off — путь как есть.
"""

from __future__ import annotations

from .. import shown_context, turn_parse
from ..runtime_config import config as _cfg

# Разбор этой реплики был, но не решён: LLM не ответила, ответ нарушил контракт, реплика — письмо (больше 40 слов), сбой кода.
# Без LLM услугу не угадываем (решение 22.10 (а)): действует только реплика, которая сама — название или код (`strong_whole`).
_FAILED = frozenset(
    {
        turn_parse.ParseStatus.UNAVAILABLE,
        turn_parse.ParseStatus.INVALID,
        turn_parse.ParseStatus.MERGED,
        turn_parse.ParseStatus.TOO_MANY,
        turn_parse.ParseStatus.ERROR,
    }
)


def enabled() -> bool:
    """`[MESSENGER_ROUTER] price_by_parse = on`: читатели цены берут фразу разбора хода; мусор — off (как было)."""

    return str(getattr(_cfg, "MR_PRICE_BY_PARSE", "off") or "").strip().lower() == "on"


def phrase(text: str) -> str | None:
    """Фраза одной услуги, которую пациент назвал в этой реплике, — по разбору хода (срез реплики: границы поставила LLM
    разметки или данные МИС).

    :param text: вход читателя цены — реплика хода
    :return: фраза; "" — разбор решил, что услуги в реплике нет (`none_fit`, `empty`); None — одной фразы нет: разбора этой
        реплики нет (не ценовой ход, вход — не реплика), он не решён (`failed`) или назван список (его ведёт корзина)
    """

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    if turn_understanding.parse_status(text) == turn_parse.ParseStatus.EMPTY:
        return ""
    parse = turn_understanding.parse_for(text)
    if parse is None or not parse.decided:
        return None
    if parse.status == turn_parse.ParseStatus.NONE_FIT or not parse.items:
        return ""
    return str(parse.items[0].text or "") if len(parse.items) == 1 else None


def failed(text: str) -> bool:
    """Разбор этой реплики был, но не решён (`_FAILED`): без LLM — только дословное название или код (решение 22.10 (а))."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    return turn_understanding.parse_status(text) in _FAILED


def number_is_choice(text: str) -> bool:
    """Число — номер из показанного прошлым ответом, а не код услуги (находка № 8 дизайна): «1» после списка цен — первая
    строка списка, а не услуга с кодом 1 (коды 1–20 в прайсе все заняты). Мерило — данные хода: ответ показал пациенту
    столько строк (`shown_context.previous()`), и число не больше их числа."""

    raw = str(text or "").strip()
    return raw.isdigit() and 0 < int(raw) <= len(shown_context.previous())
