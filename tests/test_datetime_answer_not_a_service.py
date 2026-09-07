"""Прод, диалог #1109 (07.09.2026): «Да, можем записать на приём к ортопед
(пр.Ленина, 5), **стоимость 850 руб**» — при реальной цене приёма 2700 руб.

Цена появлялась НА ВТОРОМ ходе, после того как пациент назвал дату.

Цепочка (воспроизведена офлайн на живом каталоге):

1. на шаге APPOINTMENT бот спрашивает «На какую дату и время вам удобно?»;
2. ответ «6.09.2026» уходит в `price_info` как текст запроса;
3. `resolve_price_service_name_from_catalog` честно возвращает None;
4. **fallback `_extract_price_service_from_query` отдаёт дату как имя услуги** —
   «6 09 2026»;
5. `_rank_price_rows` матчит цифры на `serviceHomecode` каталога: у услуги
   «Индейка (F284)» (аллерген, 850 руб) homecode равен **2026** — году из даты;
6. `extract_price_rub` берёт `prices[0]` вслепую → «стоимость 850 руб».

Дефект не единичный — цену подставлял ЛЮБОЙ ответ датой или временем:

| ответ пациента | что подставлялось |
|---|---|
| «6.09.2026»  | Индейка (F284) — 850 руб. |
| «15.10.2026» | Индейка (F284) — 850 руб. |
| «10:30»      | Абонемент на 10 сеансов массажа — 21 250 руб. |

Инварианты класса `datetime_answer_as_service` (SEVERE, дезинформация ценой):

* **И1.** Реплика, состоящая только из даты/времени, не является запросом
  услуги ни на одном стыке — включая `or query_text`-fallback'и family-ветки,
  которые возвращали сырой текст обратно.
* **И2.** Гард структурный, а не словарный: он опирается на ФОРМУ даты
  («д.м.гггг», «чч:мм»), которой нет у кода услуги («Код 5437» — целое число),
  и молчит, если кроме даты в реплике есть значащие слова.
* **И3.** Цена называется только за ТУ услугу, которую бот предлагает записать:
  строка прайса без общего различающего токена с услугой не цитируется.

И3 — вторая линия обороны на стыке отображения: даже если какой-то будущий
резолвер снова ошибётся, оффер записи не сможет назвать цену чужой услуги.
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.policies import extract_price_rub
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _extract_price_service_from_query,
    _is_datetime_only_query,
    _price_raw_query_fallback,
    _rank_price_rows,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _service_query_reaching_catalog(text: str) -> str:
    """Ровно то, что доходит до матчинга по каталогу на family-стыках."""

    return str(_extract_price_service_from_query(text) or _price_raw_query_fallback(text)).strip()


# --- гард окружения ---------------------------------------------------------

def test_catalog_sanity() -> None:
    """Каталог загружен и содержит строку-виновника (иначе тест бессмысленен)."""
    assert len(ROWS) > 1000
    assert any(str(r.get("serviceHomecode") or "") == "2026" for r in ROWS), (
        "в каталоге нет услуги с homecode '2026' — репро-условие исчезло"
    )


# --- И1: дата/время не запрос услуги ----------------------------------------

_DATETIME_ANSWERS = [
    # инстанс из прод-диалога #1109
    "6.09.2026",
    # тот же класс: любой день/месяц/год, разделители, ISO, время, комбинации
    "15.10.2026", "1.12.2026", "20.11.2026", "06.09.26", "6/9/2026", "6-9-2026",
    "2026-09-06", "10:30", "14:00", "09:05", "6.09.2026 14:00", "15.10 в 10:30",
    "завтра", "послезавтра", "сегодня", "завтра утром", "в 14 часов",
    "6 сентября", "20 ноября 2026", "в понедельник", "на 15.10",
]


@pytest.mark.parametrize("answer", _DATETIME_ANSWERS)
def test_datetime_answer_is_not_a_service_query(answer: str) -> None:
    """И1: реплика-дата не доходит до каталога ни как имя услуги, ни сырым текстом."""
    assert _is_datetime_only_query(answer) is True, answer
    assert _service_query_reaching_catalog(answer) == "", answer


@pytest.mark.parametrize("answer", _DATETIME_ANSWERS)
def test_datetime_answer_never_lands_on_a_price_row(answer: str) -> None:
    """И1 на живом каталоге: ни один ответ датой не выбирает строку прайса.

    Это и есть исходный симптом: «6.09.2026» → «Индейка (F284)» 850 руб.
    """
    reaching = _service_query_reaching_catalog(answer)
    matched = _rank_price_rows(ROWS, reaching, limit=3) if reaching else []
    assert not matched, (
        f"ответ датой {answer!r} приземлился на "
        f"{[str(m.get('serviceName')) for m in matched]}"
    )


# --- И2: гард не глушит нормальные запросы ----------------------------------

_MUST_NOT_BE_BLOCKED = [
    # дата/время ПРИСУТСТВУЕТ, но есть и значащие слова — это реальный запрос
    "сколько стоит приём ортопеда завтра",
    "цена УЗИ 15.10.2026",
    "можно сдать кровь завтра в 10:30",
    # код услуги — целое число, формы даты у него нет
    "Код 5437", "услуга 5437", "5437",
    # обычные услуги с числами в названии
    "УЗИ 3 триместр", "витамин Д 25", "омега 3",
    "приём ортопеда", "стоимость приёма травматолога",
]


@pytest.mark.parametrize("text", _MUST_NOT_BE_BLOCKED)
def test_guard_does_not_block_real_queries(text: str) -> None:
    """И2: гард срабатывает только на чистых датах/времени."""
    assert _is_datetime_only_query(text) is False, text
    assert _price_raw_query_fallback(text) == text, text


def test_homecode_query_still_resolves() -> None:
    """И2 регресс-гард: запрос услуги по коду продолжает работать."""
    matched = _rank_price_rows(ROWS, "Код 5437", limit=3)
    assert matched, "homecode-запрос перестал находить услугу"


# --- И3: цена только за предлагаемую услугу ---------------------------------

def _row_named(fragment: str) -> dict:
    for row in ROWS:
        if fragment.lower() in str(row.get("serviceName") or "").lower():
            return row
    raise AssertionError(f"в каталоге нет строки с {fragment!r}")


def test_offer_quotes_price_of_the_offered_service() -> None:
    """И3: цена своей услуги называется."""
    row = _row_named("травматолога-ортопеда")
    payload = {"prices": [row]}
    assert extract_price_rub(payload, expected_service="приём к ортопед") is not None


@pytest.mark.parametrize(
    "foreign_fragment",
    ["Индейка (F284)", "Кукуруза", "Медь в крови", "Абонемент на 10 сеансов"],
)
def test_offer_never_quotes_price_of_a_foreign_service(foreign_fragment: str) -> None:
    """И3: цена чужой услуги в оффер записи не попадает.

    Вторая линия обороны — независимо от того, как строка попала в evidence.
    """
    payload = {"prices": [_row_named(foreign_fragment)]}
    assert extract_price_rub(payload, expected_service="приём к ортопед") is None


@pytest.mark.parametrize(
    "specialty, row_fragment",
    [
        ("ортопед", "травматолога-ортопеда"),
        # «дерматолог» и «дерматовенеролога» полным префиксом НЕ совпадают
        # (расходятся на 8-м символе) — строгий startswith глушил свою же цену.
        ("дерматолог", "дерматовенеролога"),
        ("кардиолог", "врача-кардиолога"),
        ("гинеколог", "акушера-гинеколог"),
        ("аллерголог", "аллерголога-имм"),
        ("оториноларинголог", "оториноларинголог"),
    ],
)
def test_offer_price_survives_related_catalog_wording(specialty: str, row_fragment: str) -> None:
    """И3 анти-over-block: своя цена не глушится формулировкой каталога.

    Гард ПОДТВЕРЖДАЮЩИЙ, а не выбирающий: строку уже выбрал резолвер. Ложное
    подавление скрыло бы законную цену, поэтому родственные термины проходят.
    """
    row = _row_named(row_fragment)
    price = extract_price_rub({"prices": [row]}, expected_service=f"приём к {specialty}")
    assert price is not None, f"цена {specialty} подавлена строкой {row.get('serviceName')!r}"


def test_price_without_expected_service_is_unchanged() -> None:
    """Совместимость: без `expected_service` поведение прежнее (цена берётся)."""
    payload = {"prices": [_row_named("Индейка (F284)")]}
    assert extract_price_rub(payload) == "850"
