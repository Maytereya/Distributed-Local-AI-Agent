"""Класс: каждая услуга списка получает свою цену; вопрос о сроке готовности цену не получает.

BUG-2026-10-05-PRICE-LIST-COLLAPSES-TO-ONE: список в вопросе цены сводится к одной услуге раньше
разбора корзины — подстановка каталога и грундер приземляют реплику целиком на одну строку
(«холестерин» → «Липидограмма»), а планировщик с известной услугой зовёт `service_bundle_info`,
где корзины нет. Сборщик корзины верен (`test_multi_price_survives_patient_synonyms`), но
живой путь до него не доходит — поэтому тест идёт через обработчик эндпоинта.

BUG-2026-10-05-READINESS-AS-PRICE: «когда будет готов анализ на ферритин» — подбор анализов
приземляется на услугу каталога и переклеивается в цену; переклейка смотрит только флаг
доставки результата, а не то, о каком свойстве спросили.

Инварианты класса:
- список из названий, которые каталог узнаёт, — цена каждой позиции, текстом и ответом на
  кнопку цены, со словом «сколько стоит» и без него, в новом диалоге и поверх старой услуги;
- вопрос о сроке готовности не получает цену.

Ход (б), шаг 1 (L-14): `услуги[]` + `свойство` на метке «цена», под выключателем
`turn_understanding_price`. Тесты включают его явно (`on`): по умолчанию шаг в тени.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

# Список → названия услуг каталога, которые обязаны прозвучать в ответе.
LISTS = {
    "ферритин, глюкоза, холестерин": ("ферритин", "глюкоза", "липидограмма"),
    "оак, оам": ("общий анализ крови", "общий анализ мочи"),
    "ферритин и глюкоза": ("ферритин", "глюкоза"),
    "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА": ("общий анализ крови", "общий анализ мочи", "ферритин", "глюкоза"),
}

# Как пациент задаёт список: (ходы до списка, шаблон реплики).
PATHS = {
    "text": ((), "сколько стоит {}"),
    "button": ((("menu.price.test", "Цена анализа"),), "{}"),
    "bare": ((), "{}"),
    "over_context": (("сколько стоит ТТГ",), "а {}?"),
}

# Сегодня на этих местах класс проходит и без шага (б) — xfail(strict) их не держит.
_PASSING_TODAY = {("text", "ферритин и глюкоза"), ("button", "ферритин и глюкоза")}

READINESS = [
    "когда будет готов анализ на ферритин",
    "через сколько будет готов ферритин",
    "сколько делается анализ на ферритин",
    "сколько готовится ферритин",
    "сколько готовятся анализы",
]
_READINESS_PRICED_TODAY = {"когда будет готов анализ на ферритин", "сколько делается анализ на ферритин"}


@pytest.fixture
def understanding_on(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "on", raising=False)


def _cases():
    for path in sorted(PATHS):
        for items in LISTS:
            marks = () if (path, items) in _PASSING_TODAY else (pytest.mark.xfail(strict=True, reason="PRICE-LIST-COLLAPSES-TO-ONE: до шага (б)-1"),)
            yield pytest.param(path, items, id=f"{path}-{items}", marks=marks)


@pytest.mark.parametrize(("path", "items"), list(_cases()))
def test_every_listed_service_gets_its_price(path, items, mini_prod_world, understanding_on):  # noqa: F811
    before, template = PATHS[path]
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, template.format(items)], f"list-{path}-{abs(hash(items))}")
    answer = record[-1]["text"].lower()
    missing = [name for name in LISTS[items] if name not in answer]
    assert missing == [], record[-1]["text"]
    assert "руб" in answer, record[-1]["text"]


@pytest.mark.parametrize(
    "question",
    [
        pytest.param(q, marks=(pytest.mark.xfail(strict=True, reason="READINESS-AS-PRICE: до шага (б)-1"),) if q in _READINESS_PRICED_TODAY else ())
        for q in READINESS
    ],
)
def test_readiness_question_gets_no_price(question, mini_prod_world, understanding_on):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        (turn,) = run_dialog([question], f"ready-{abs(hash(question))}")
    assert "руб" not in turn["text"].lower(), turn["text"]
