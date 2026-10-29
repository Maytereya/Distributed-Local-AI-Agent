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

# LLM разметки не ответила (сбой модели, таймаут, отказ лимита решений хода) — без LLM услугу не угадываем (решение 22.10 (а)):
# действует только реплика, которая сама — название или код (`strong_whole`). Прочие нерешённые исходы — не «без LLM»: ответ не по
# контракту (`invalid`, `merged`), письмо длиннее 40 слов (`too_many`), сбой кода (`error`) — модель работает, строку по-прежнему
# выбирает LLM (выбор строки, корзина); карточка Ш7 (25.10): «уточните» на них стоило 4 верных ответов dev из 867 (два длинных
# списка по 15 позиций, «глюкоза в плазме», ВЭБ).
_FAILED = frozenset({turn_parse.ParseStatus.UNAVAILABLE})


def enabled() -> bool:
    """`[MESSENGER_ROUTER] price_by_parse = on`: читатели цены берут фразу разбора хода; мусор — off (как было)."""

    return str(getattr(_cfg, "MR_PRICE_BY_PARSE", "off") or "").strip().lower() == "on"


def phrase(text: str) -> str | None:
    """Фраза одной услуги, которую пациент назвал в этой реплике, — по разбору хода (срез реплики: границы поставила LLM
    разметки или данные МИС).

    :param text: вход читателя цены — реплика хода
    :return: фраза; "" — в реплике нет слов (`empty`); None — одной фразы нет: разбора этой реплики нет (не ценовой ход, вход —
        не реплика), он не решён, назван список (его ведёт корзина) или разметка сказала «ни одна фраза не услуга» (`none_fit`):
        выбор строки при этом бывает прав («посев на хламидии», «анализы для ревматолога» — replay 21.10: 5 потерь верной цены из
        1068 ходов; карточка Ш7: 4 потери dev), поэтому поиск идёт по реплике, как раньше; печатать и писать клинике — нечего
    """

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    if turn_understanding.parse_status(text) == turn_parse.ParseStatus.EMPTY:
        return ""
    parse = turn_understanding.parse_for(text)
    if parse is None or not parse.decided:
        return None
    if parse.status == turn_parse.ParseStatus.NONE_FIT or not parse.items:
        return None
    return str(parse.items[0].text or "") if len(parse.items) == 1 else None


def failed(text: str) -> bool:
    """LLM разметки этой реплики не ответила (`_FAILED`): без LLM — только дословное название или код (решение 22.10 (а))."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    return turn_understanding.parse_status(text) in _FAILED


def insurance_marked(text: str) -> bool:
    """Разметка этой реплики пометила оплату страховкой — ОМС, ДМС, полис (`position_qualifiers = on`): у позиции или ни к одной.
    Метка — сведение, не вердикт (карточка 28.10: живая LLM метила так и «цену»): перевод на оператора — только вместе с судом
    условия, который сказал «условия в строке нет» (решение владельца 28.10: цену по ОМС и ДМС называет оператор). С вердиктом по
    метке (`insurance_confirmed`) это правило — запасное: действует, когда отдельный вопрос об оплате не ответил."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(text)
    if parse is None or not parse.decided or parse.insurance == "denied":  # вопрос об оплате сказал «не страховка» — метки нет
        return False
    insurance = turn_parse.QualifierKind.INSURANCE
    return any(q.kind == insurance for q in parse.loose) or any(item.has(insurance) for item in parse.items)


def insurance_confirmed(text: str) -> bool:
    """Пациент спрашивает цену по ОМС / ДМС / страховке — так сказали оба вызова LLM: разметка пометила оплату, отдельный вопрос
    подтвердил (`_turn_spans._insurance_verdict`). Решение владельца 28.10: такую цену называет оператор — перевод хода до выбора
    строк (суд условия корзины пропускал «по ДМС» и «по полису»)."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(text)
    return parse is not None and parse.decided and parse.insurance == "confirmed"


def loose_qualifier(text: str) -> bool:
    """Разметка этой реплики отметила уточнение ни к одной услуге (DESIGN-2026-10-28-POSITION-QUALIFIERS, этап 2; есть только под
    `position_qualifiers = on`): «а сколько стоит по омс?» после цены — вопрос об условии, а не «та же строка ещё раз»."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — импорт на вызове

    parse = turn_understanding.parse_for(text)
    return parse is not None and parse.decided and bool(parse.loose)


def number_is_choice(text: str) -> bool:
    """Число — номер из показанного прошлым ответом, а не код услуги (находка № 8 дизайна): «1» после списка цен — первая
    строка списка, а не услуга с кодом 1 (коды 1–20 в прайсе все заняты). Мерило — данные хода: ответ показал пациенту
    столько строк (`shown_context.shown_count()`), и число не больше их числа."""

    raw = str(text or "").strip()
    return raw.isdigit() and 0 < int(raw) <= shown_context.shown_count()
