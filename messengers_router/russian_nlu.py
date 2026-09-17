"""Russian NLU utilities — single source of truth.

Every module in messengers_router MUST import from here instead of
inlining `.lower().replace("ё", "е")` and related patterns.
This eliminates 20+ scattered copies of the same normalization idiom.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4096)
def normalize_ru(text: str | None) -> str:
    """Lowercase + ё→е normalization. Returns ``""`` for None.

    Use everywhere in place of the inline idiom:
        str(x).lower().replace("ё", "е").strip()
    """
    return str(text or "").lower().replace("ё", "е").strip()


# Canonical entity whitelist — single source of truth for all three former copies:
#   prompt_contracts.ALLOWED_ENTITY_KEYS
#   classifier._ALLOWED_ENTITY_KEYS
#   entity_grounder._LABEL_ENTITY_WHITELIST keys (union)
ENTITY_WHITELIST: frozenset[str] = frozenset({
    "doctor_name",
    "doctor_id",
    "specialty",
    "branch_name",
    "branch_id",
    "city",
    "service_name",
    "appointment_action",
    "patient_name",
    "test_name",
    "test_goal",
    "surname",
    "year",
    "filial",
    "number",
    "order_id",
    "result_action",
    "lang",
    "insurance_type",
    "accepts_children",
    "child_age",
    "date_hint",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "secondary_intents",
    "include_promos",
    "time_flexible",
})


def common_prefix_len(a: str, b: str) -> int:
    """Длина общего начала двух строк.

    Был объявлен ДВАЖДЫ — в `services/_prices_helpers` и в `policies`, — двумя
    разными реализациями одного и того же (цикл `while` против `zip`). Модули
    друг друга не импортируют, поэтому копия жила незаметно.

    :param a: первая строка
    :param b: вторая строка
    :return: сколько первых символов совпадает
    """

    n = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        n += 1
    return n


def tokens_share_stem(a: str, b: str, *, length: int) -> bool:
    """Считать ли два слова одним словом по общему началу.

    Единственное место, где живёт правило «одно ли это слово». До 17.09 оно было
    переписано шестью способами с тремя разными порогами: 4 символа в
    `_token_covered_by`, в расширении семейства, в сегментации корзины и в
    подборе врача; 5 — в скорере прайса; 6 — в гарде отображения цены. Пороги
    остаются разными там, где это обосновано, но правило теперь одно и
    проверяется в одном месте.

    Порог передаётся ЯВНО и по умолчанию не подставляется: «сколько символов
    достаточно, чтобы счесть слова одинаковыми» — решение вызывающей стороны, и
    молчаливое умолчание здесь уже приводило к ошибкам в обе стороны («сахар» ↔
    «сахарная» слиплись, «дерматолог» ↔ «дерматовенеролог» разошлись).

    :param a: первое слово
    :param b: второе слово
    :param length: сколько первых символов обязаны совпасть
    :return: True, если оба слова не короче порога и их начала совпадают
    """

    if a == b:
        # Одинаковые слова — одно слово всегда, даже если они короче порога.
        # Без этой строки «оак» перестал бы совпадать сам с собой: сверка со
        # старыми формами правила поймала 725 таких пар на живом словаре.
        return bool(a)
    if len(a) < length or len(b) < length:
        return False
    return a[:length] == b[:length]
