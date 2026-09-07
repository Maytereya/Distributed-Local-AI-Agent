"""Класс `guard_over_blocks_silently`: гард, поставленный против дезинформации,
сам молча гасит КОРРЕКТНЫЙ результат.

Это второй по счёту случай в проекте, поэтому здесь закрепляется не инстанс,
а **жанр теста**: у каждого гарда, который что-то запрещает, должен быть
встречный свип «а не запрещает ли он лишнего» по живым данным.

## Почему обычный гейт этого не ловит

Тест на гард пишется от его цели: «„по ОМС“ не должно отдавать платную цену»,
«цена чужой услуги не должна цитироваться». Такой тест зелёный и когда гард
работает, и когда он лупит по площадям. Ущерб от over-block **молчаливый**:
пациент видит не ошибку, а отсутствие ответа, и в логах это неотличимо от
«услуги нет».

## Общий механизм всех трёх известных инстансов

Гард сравнивает **пациентский словарь** с **поверхностными формами каталога**
и делает вывод «клиника такого не делает» из того, что слова нет в прайсе:

| пациент сказал | резолвер верно нашёл | гард убил, потому что |
|---|---|---|
| «холестерин» | Липидограмма | слова «холестерин» в каталоге нет |
| «оам» | Общий анализ мочи | аббревиатуры в каталоге нет |
| «узи» | «…с ультразвуковым исследованием» | нет общего токена |

Резолвер эти синонимы ЗНАЕТ — гард пересматривает его решение по словарю,
в котором пациентской лексики нет по построению.

## Инварианты

* **И-A (ценовой гард оффера).** Если проектный экстрактор читает в строке
  прайса специальность, то гард отображения обязан пропустить цену этой
  строки для этой же специальности. Свип по всему каталогу — именно он
  поймал бы over-block «дерматолог» ↔ «дерматовенеролога», которого не
  видел ни один тест.
* **И-Б (гард мульти-расчёта).** Если корзина распозналась полностью
  (≥2 позиции, нераспознанных нет), расчёт обязан быть выдан. Сейчас
  НАРУШЕН — отмечено `xfail`, см. журнал багов.
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.policies import _price_row_matches_service
from messengers_router.services._doctors_helpers import (
    _extract_specialty_from_text,
    _is_clean_consultation_row_name,
)
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _build_multi_price_payload,
    _resolve_multi_price_items,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _consultation_specialty_pairs() -> list[tuple[str, str]]:
    """Пары (специальность, строка прайса), выведенные ИЗ каталога.

    Специальность читается из самой строки проектным экстрактором — ручного
    списка специальностей здесь нет, поэтому свип автоматически расширяется
    вместе с прайсом.
    """

    pairs: list[tuple[str, str]] = []
    for row in ROWS:
        name = str(row.get("serviceName") or "")
        if not _is_clean_consultation_row_name(name):
            continue
        specialty = _extract_specialty_from_text(name)
        if specialty:
            pairs.append((specialty, name))
    return pairs


def test_catalog_sanity() -> None:
    """Гард окружения: свип имеет на чём работать."""
    pairs = _consultation_specialty_pairs()
    assert len(pairs) >= 100, f"консультационных строк слишком мало: {len(pairs)}"
    assert len({p[0] for p in pairs}) >= 20


# --- И-A: ценовой гард оффера не глушит свою же цену ------------------------

def test_price_guard_never_suppresses_own_specialty_price() -> None:
    """И-A, сплошной свип по каталогу.

    Именно этот тест поймал бы over-block «дерматолог» ↔ «дерматовенеролога»:
    строгое сравнение по префиксу молча прятало цену приёма дерматолога,
    а целевые тесты гарда при этом оставались зелёными.
    """

    suppressed = [
        (specialty, name)
        for specialty, name in _consultation_specialty_pairs()
        if not _price_row_matches_service(name, f"приём к {specialty}")
    ]
    assert not suppressed, (
        "гард отображения глушит цену СВОЕЙ услуги "
        f"({len(suppressed)} строк): {suppressed[:5]}"
    )


@pytest.mark.parametrize(
    "row_name, specialty",
    [
        # Аббревиатура пациента против развёрнутой формулировки каталога.
        ("Прием (осмотр, консультация) врача-акушера-гинеколога  (прием с "
         "ультразвуковым гинекологическим исследованием одним датчиком)", "узи"),
        # Родственный термин, полным префиксом не совпадающий.
        ("Приём (осмотр, консультация) врача дерматовенеролога", "дерматолог"),
    ],
)
def test_price_guard_handles_patient_vocabulary(row_name: str, specialty: str) -> None:
    """И-A: пациентское слово ≠ форма каталога — цену это глушить не должно."""
    assert _price_row_matches_service(row_name, f"приём к {specialty}") is True


def test_price_guard_still_blocks_foreign_services() -> None:
    """Анти-under-block: смягчения не должны открыть дорогу чужой услуге."""
    for foreign in (
        "Индейка (F284)",
        "Кукуруза (F8)",
        "Медь в крови",
        "Абонемент на 10 сеансов медицинского массажа",
        "Липосакция одной области",
    ):
        assert _price_row_matches_service(foreign, "приём к ортопед") is False, foreign


@pytest.mark.parametrize(
    "offered_specialty, foreign_row",
    [
        # Самый вероятный реальный мисматч — не «Индейка», а консультация
        # ДРУГОГО врача: строки похожи почти целиком, различает одно слово.
        ("кардиолог", "Приём эндокринолога"),
        ("кардиолог", "Прием (осмотр, консультация) врача-терапевта"),
        ("ортопед", "Прием (осмотр, консультация) врача-уролога"),
        ("невролог", "Прием (осмотр, консультация) врача-онколога"),
        ("терапевт", "Прием (осмотр, консультация) врача-кардиолога"),
    ],
)
def test_price_guard_blocks_other_specialty_consultation(
    offered_specialty: str, foreign_row: str
) -> None:
    """Анти-under-block на консультациях: служебные слова совпадают почти все.

    «Прием (осмотр, консультация) врача-» есть у обеих строк — если бы они
    участвовали в сравнении, гард пропустил бы цену чужого врача. Отсюда
    исключение служебных токенов в `_PRICE_MATCH_GENERIC_TOKENS`.
    """
    assert _price_row_matches_service(foreign_row, f"приём к {offered_specialty}") is False


# --- И-Б: гард мульти-расчёта (НАРУШЕН, зафиксирован xfail) -----------------

@pytest.mark.parametrize(
    "cart",
    [
        "общий анализ крови, общий анализ мочи, глюкоза",
        "ОАК, ферритин",
    ],
)
def test_multi_price_survives_when_cart_fully_resolved(cart: str) -> None:
    """И-Б: полностью распознанная корзина отдаёт расчёт."""
    items, unrecognized = _resolve_multi_price_items(cart, ROWS)
    assert len(items) >= 2 and not unrecognized, f"предусловие не выполнено: {cart!r}"
    assert _build_multi_price_payload(cart, ROWS) is not None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ОТКРЫТЫЙ БАГ: гард `_match_drops_unsatisfiable_qualifier` (aa6b7de) "
        "считает пациентский синоним неудовлетворимым уточнением, потому что "
        "самого слова нет в каталоге, и гасит полностью распознанную корзину. "
        "Блокирует калькулятор анализов. См. BUG-2026-09-07-GUARD-OVER-BLOCK. "
        "Когда починят — тест станет XPASS и потребует убрать этот маркер."
    ),
)
@pytest.mark.parametrize(
    "cart",
    [
        # Полные названия, БЕЗ аббревиатур: «холестерин» верно резолвится в
        # «Липидограмма», но самого слова в каталоге нет — корзина гибнет.
        "ферритин, глюкоза, холестерин",
        # Аббревиатура «оам» → «Общий анализ мочи»: то же самое.
        "ОАК ОАМ глюкоза ферритин",
    ],
)
def test_multi_price_survives_patient_synonyms(cart: str) -> None:
    """И-Б для пациентской лексики. Сейчас нарушен — расчёт гасится."""
    items, unrecognized = _resolve_multi_price_items(cart, ROWS)
    assert len(items) >= 2 and not unrecognized, f"предусловие не выполнено: {cart!r}"
    assert _build_multi_price_payload(cart, ROWS) is not None
