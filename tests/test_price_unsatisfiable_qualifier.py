"""П2 (прод 13.08): значащее уточнение молча отбрасывалось, запрос садился на
обычную услугу.

Симптом: «А возможна ли онлайн консультация?» → «Онлайн-консультации в нашей
клинике доступны. Уточните специалиста». Проверено офлайн на живом каталоге —
5 из 10 фраз теряли уточнение:

| Запрос | Резолвился в | Потеряно |
|---|---|---|
| онлайн консультация | Прием врача-хирурга | «онлайн» |
| приём кардиолога онлайн | Прием врача-кардиолога | «онлайн» |
| **приём терапевта по ОМС** | **Прием врача-терапевта (платный)** | **«ОМС»** |
| консультация по телефону | Прием врача-хирурга | «телефон» |
| удалённая консультация | Прием врача-хирурга | «удалённо» |

Строка про ОМС опаснее всех: пациент спрашивает про полис, получает платную цену.

Корень: `_scorer_match_has_distinctive_overlap` требует ОДИН совпавший
различающий токен («консультация») и не проверяет, что остальные уточнения
запроса вообще чем-то удовлетворены.

Инвариант (класс `unsatisfiable_qualifier_dropped`, тот же архитектурный класс,
что «гепатит С→В»): если в запросе есть различающий токен, которого нет ни в
найденной услуге, ни ВООБЩЕ НИ В ОДНОЙ строке каталога — каталог такой запрос
удовлетворить не может; совпадение не засчитывается (→ None → честное «не нашёл»
/ оператор), а не подменяется услугой без этого свойства.

Механизм catalog-derived: словарь допустимых токенов = сам прайс. Ни списка
«онлайн/ОМС/телефон» в коде, ни списка услуг.
"""

from __future__ import annotations

import random

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    resolve_price_service_name_from_catalog,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _resolve(q: str) -> str | None:
    return resolve_price_service_name_from_catalog(q, rows=ROWS)


def test_catalog_sanity_has_consultations():
    """Гард окружения: приёмы врачей в каталоге есть (иначе тест бессмысленен)."""
    names = " | ".join(str(r.get("serviceName") or "") for r in ROWS).lower()
    assert "врача-терапевта" in names and "врача-кардиолога" in names


# --- Класс: неудовлетворимое уточнение не отбрасывается --------------------

@pytest.mark.parametrize(
    "query",
    [
        "онлайн консультация",
        "приём кардиолога онлайн",
        "приём терапевта по ОМС",
        "консультация по телефону",
        "удалённая консультация",
        "консультация по видеосвязи",
        "приём гинеколога по полису ОМС",
    ],
    ids=[
        "online_consult",
        "cardio_online",
        "therapist_oms",
        "consult_by_phone",
        "remote_consult",
        "video_consult",
        "gyn_oms_policy",
    ],
)
def test_unsatisfiable_qualifier_blocks_match(query):
    """Уточнения «онлайн/ОМС/по телефону/удалённо/видеосвязь» не встречаются
    НИ В ОДНОЙ строке каталога → услугу-без-этого-свойства подставлять нельзя."""
    assert _resolve(query) is None, f"{query!r} по-прежнему приземляется на услугу"


def test_oms_query_never_returns_paid_service():
    """Самый опасный кейс: спрашивают про ОМС — платную цену выдавать нельзя."""
    resolved = _resolve("приём терапевта по ОМС")
    assert resolved is None, resolved


# --- Второй стык: family-режим (как у дискриминатора гепатитов) ------------

@pytest.mark.parametrize(
    "query",
    [
        "приём терапевта по ОМС",
        "приём гинеколога по полису ОМС",
        "приём кардиолога онлайн",
    ],
    ids=["therapist_oms", "gyn_oms_policy", "cardio_online"],
)
def test_family_mode_does_not_offer_list_instead_of_answer(query):
    """Одного гарда в single-резолвере мало: без гарда в family-builder запрос
    «по ОМС» проваливался в семейный режим и получал список ЧУЖИХ платных приёмов
    (хирург, мануальный терапевт) — дезинформация вместо честного «не нашёл»."""
    from messengers_router.services._prices_helpers import _build_price_family_payload

    assert _build_price_family_payload(query, ROWS) is None


def test_cart_keeps_honest_unrecognized_item():
    """Граница со «списком анализов» (BUG-B): нераспознанная позиция —
    САМОСТОЯТЕЛЬНЫЙ пункт списка, S3 честно называет её пациенту. Гард такой
    список глушить не имеет права; глушит только уточнение, поглощённое ВНУТРИ
    фразы («приём терапевта по ОМС» — один пункт, а не список)."""
    from messengers_router.services._prices_helpers import _build_multi_price_payload

    payload = _build_multi_price_payload("ОАК\nферритин\nКВАНТОВЫЙ АНАЛИЗ АУРЫ\nглюкоза", ROWS)
    assert payload is not None
    assert "аур" in str(payload.get("unrecognized_note") or "").lower()

    assert _build_multi_price_payload("приём терапевта по ОМС", ROWS) is None


def test_guard_silent_on_service_code_lookup():
    """Запрос услуги ПО КОДУ резолвится по homecode, а не по названию —
    обрамляющие слова там не уточнение услуги (BUG-C, «Код 5437»)."""
    from messengers_router.services._prices_helpers import (
        _match_drops_unsatisfiable_qualifier,
    )

    assert _match_drops_unsatisfiable_qualifier("Код 5437", "", ROWS) is False
    assert _match_drops_unsatisfiable_qualifier("услуга 5437", "", ROWS) is False


def test_guard_silent_on_small_synthetic_catalog():
    """Утверждение «такого слова в прайсе нет» опирается на ПОЛНЫЙ прайс.
    На синтетическом срезе из пары строк гард обязан молчать — иначе он рубил бы
    любой обычный запрос (тест-срезы соседних сьютов)."""
    from messengers_router.services._prices_helpers import (
        _match_drops_unsatisfiable_qualifier,
    )

    tiny = [{"serviceName": "УЗДГ сосудов шеи", "cost": 1800}, {"serviceName": "ЛПНП", "cost": 450}]
    assert (
        _match_drops_unsatisfiable_qualifier(
            "обследование уздг сосудов шеи и сдать кровь на ЛПНП, какова стоимость?",
            "УЗДГ сосудов шеи",
            tiny,
        )
        is False
    )


def test_family_mode_still_works_for_normal_family_query():
    """Анти-over-trigger: обычный семейный запрос по-прежнему отдаёт варианты."""
    from messengers_router.services._prices_helpers import _build_price_family_payload

    payload = _build_price_family_payload("гепатит", ROWS)
    assert payload is not None
    assert payload.get("service_kind") == "family_query"


# --- Анти-over-trigger: обычные запросы продолжают резолвиться -------------

@pytest.mark.parametrize(
    "query, expect_fragment",
    [
        ("приём кардиолога", "кардиолога"),
        ("консультация уролога", "уролога"),
        ("приём терапевта", "терапевта"),
        ("спермограмма", "спермограмма"),
        ("общий анализ крови", "общий анализ крови"),
        ("С-реактивный белок", "реактивный белок"),
        ("ферритин", "ферритин"),
        ("липидограмма", "липидограмма"),
        ("УЗИ щитовидной железы", "щитовидной железы"),
    ],
    ids=[
        "cardio",
        "urolog",
        "therapist",
        "spermogram",
        "oak",
        "crp",
        "ferritin",
        "lipid",
        "uzi_thyroid",
    ],
)
def test_normal_queries_still_resolve(query, expect_fragment):
    resolved = _resolve(query)
    assert resolved is not None, f"{query!r} перестал резолвиться"
    assert expect_fragment.lower() in resolved.lower(), (query, resolved)


@pytest.mark.parametrize(
    "query",
    [
        "сколько стоит спермограмма",
        "подскажите пожалуйста цену на ферритин",
        "хочу узнать стоимость общего анализа крови",
        "мне нужен анализ на ферритин",
        "цена приёма кардиолога",
        "сколько будет стоить УЗИ щитовидной железы",
    ],
    ids=["price_verb", "polite", "wordy", "need_test", "price_noun", "uzi_wordy"],
)
def test_conversational_noise_does_not_block_match(query):
    """Разговорная обвязка («мне нужен», «подскажите», «хочу узнать») не должна
    трактоваться как неудовлетворимое уточнение — иначе гард убьёт живые запросы.

    Проверено ДО фикса: все эти запросы резолвились, значит падение здесь = вина
    гарда, а не предсуществующее поведение матчера."""
    assert _resolve(query) is not None, f"{query!r} заблокирован разговорной обвязкой"


# --- E2E: пациентский ответ на всех четырёх стыках ------------------------

@pytest.mark.parametrize(
    "query",
    [
        "приём терапевта по ОМС",
        "приём гинеколога по полису ОМС",
        "приём кардиолога онлайн",
        "консультация по телефону",
        "онлайн консультация",
        "удалённая консультация",
    ],
    ids=[
        "therapist_oms",
        "gyn_oms_policy",
        "cardio_online",
        "consult_by_phone",
        "online_consult",
        "remote_consult",
    ],
)
def test_price_info_returns_no_price_for_unsatisfiable_qualifier(query):
    """Главная проверка класса: пациент НЕ получает цену услуги без запрошенного
    свойства. Гард нужен на ЧЕТЫРЁХ стыках (single-резолвер, family-builder,
    multi/«корзина», fallback на сырую фразу в price_info) — каждый по отдельности
    обходился остальными; тест держит все сразу через живой price_info."""
    import asyncio

    from messengers_router.services import Services

    payload = asyncio.run(Services().price_info(query, {}))
    assert not payload.get("prices"), payload.get("prices")
    assert not payload.get("family_variants"), payload.get("family_variants")


@pytest.mark.parametrize(
    "query, expect_fragment",
    [
        ("приём кардиолога", "кардиолога"),
        ("сколько стоит спермограмма", "спермограмма"),
        ("мне нужен анализ на ферритин", "ферритин"),
        ("гепатит с", "гепатит с"),
    ],
    ids=["cardio", "spermogram", "ferritin", "hepatitis_c"],
)
def test_price_info_still_answers_normal_queries(query, expect_fragment):
    """Анти-over-trigger на E2E-уровне: обычные запросы по-прежнему дают цену."""
    import asyncio

    from messengers_router.services import Services

    payload = asyncio.run(Services().price_info(query, {}))
    rows = list(payload.get("prices") or []) + list(payload.get("family_variants") or [])
    assert rows, f"{query!r}: пустой ответ"
    names = " | ".join(str(r.get("serviceName") or "") for r in rows).lower()
    assert expect_fragment.lower().split()[0] in names, (query, names[:200])


def test_typo_still_resolves():
    """Опечатка покрывается найденной услугой (префиксное совпадение) и потому
    не считается неудовлетворённым уточнением."""
    assert _resolve("спермограма") is not None


# --- Класс-sweep: гард не ломает сам каталог ------------------------------

def test_guard_never_fires_on_catalog_names_against_themselves():
    """Верхняя граница blast radius, дёшево: словарь допустимых токенов = сам
    каталог, поэтому НИ ОДНО название услуги не может содержать токен, которого
    в каталоге нет. Проверяем предикат напрямую по всем 3600+ строкам — без
    ранжирования, поэтому быстро."""
    from messengers_router.services._prices_helpers import (
        _match_drops_unsatisfiable_qualifier,
    )

    offenders = []
    for row in ROWS:
        name = str(row.get("serviceName") or row.get("name") or "").strip()
        if not name:
            continue
        if _match_drops_unsatisfiable_qualifier(name, name, ROWS):
            offenders.append(name)
    assert not offenders, f"гард сработал на самих названиях каталога: {offenders[:10]}"


def test_guard_ignores_token_that_catalog_knows():
    """Гард НЕ его дело — различать варианты внутри каталога. Если токен запроса
    в каталоге есть (пусть и в другой услуге), совпадение не блокируется: этим
    занимаются family-дискриминатор и scorer-overlap."""
    from messengers_router.services._prices_helpers import (
        _match_drops_unsatisfiable_qualifier,
    )

    # «крови» каталогу известно, хотя в «Спермограмма» его нет → не наш класс.
    assert _match_drops_unsatisfiable_qualifier("спермограмма крови", "Спермограмма", ROWS) is False
    # «онлайн» каталогу неизвестно вовсе → наш класс.
    assert _match_drops_unsatisfiable_qualifier(
        "онлайн консультация", "Прием (осмотр, консультация) врача-хирурга", ROWS
    ) is True


def test_catalog_names_verbatim_still_resolve():
    """E2E-срез (маленькая выборка — полный резолв дорогой): дословные названия
    услуг продолжают резолвиться."""
    random.seed(20260814)
    sample = [
        str(r.get("serviceName") or "").strip()
        for r in random.sample(ROWS, 25)
        if str(r.get("serviceName") or "").strip()
    ]
    broken = [name for name in sample if _resolve(name) is None]
    # Каталог содержит строки, которые матчер не берёт и БЕЗ гарда (мусорные,
    # сверхдлинные, с кодами) — фиксируем порог, чтобы поймать именно регрессию гарда.
    assert len(broken) <= len(sample) * 0.12, (
        f"гард срезал {len(broken)}/{len(sample)} дословных названий каталога: {broken[:10]}"
    )
