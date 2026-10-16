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
- вопрос о сроке готовности не получает цену и срока свободным текстом LLM: срок печатает код из
  справочника МИС, а нет срока или услуга не названа — честный перевод на оператора.

Ход (б), шаг 1 (L-14): `услуги[]` + `свойство` на метке «цена», под выключателем
`turn_understanding_price`. Тесты включают его явно (`on`). Разбор реплики на фразы (`turn_parse`) — единственный
режим (T7 шаг 3б, 16.10).
До шага 14 из 16 мест списка и 2 из 5 формулировок срока были красными.
"""

from __future__ import annotations

import re

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

# Вопрос о сроке → срок из МИС кодом (в мини-прод у ферритина и глюкозы срок «1-2»).
READINESS_KNOWN = {
    "когда будет готов анализ на ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
    "через сколько будет готов ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
    "сколько делается анализ на ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
    "сколько готовится ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
}
# Услуга не названа, срока в МИС нет (у липидограммы поле пустое) или это список (строку каждой
# позиции выберет choose(), ход (в)) — честный перевод.
READINESS_UNKNOWN = ["сколько готовятся анализы", "когда будет готов анализ на липидограмму", "когда будет готов анализ на ферритин и глюкозу"]


@pytest.fixture
def understanding_on(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "on", raising=False)


# Разбор реплики — единственный режим (T7 шаг 3б). Сплошной текст без разделителей при недоступной LLM режут только
# данные МИС (T7 шаг 3а): в мини-проде у «ОАК ОАМ» нет синонимов МИС — слова вне прайса в промежутке, без LLM
# неотличимые от условия П2, — реплика целиком (на проде — синонимы, разрез данными). Тот же случай —
# `test_price_one_or_many_class.py::list_caps_run_on_bare`.
_RUN_ON_BARE = "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА"
_RUN_ON_REASON = (
    "LLM недоступна, сплошной текст: «ОАК ОАМ» без синонимов МИС в мини-проде — слова вне прайса в промежутке, реплика "
    "целиком одной услугой (T7 шаг 3а, П2 строже покрытия); с синонимами МИС (прод) — разрез данными"
)


def _list_params():
    for items in LISTS:
        marks = [pytest.mark.xfail(strict=True, reason=_RUN_ON_REASON)] if items == _RUN_ON_BARE else []
        yield pytest.param(items, marks=marks, id=items)


@pytest.mark.parametrize("items", list(_list_params()))
@pytest.mark.parametrize("path", sorted(PATHS))
def test_every_listed_service_gets_its_price(path, items, mini_prod_world, understanding_on):  # noqa: F811
    before, template = PATHS[path]
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, template.format(items)], f"list-{path}-{abs(hash(items))}")
    answer = record[-1]["text"].lower()
    missing = [name for name in LISTS[items] if name not in answer]
    assert missing == [], record[-1]["text"]
    assert "руб" in answer, record[-1]["text"]


_BEFORE = [(), (("menu.price.test", "Цена анализа"),)]


@pytest.mark.parametrize("before", _BEFORE, ids=["text", "after_price_button"])
@pytest.mark.parametrize("question", sorted(READINESS_KNOWN))
def test_readiness_question_gets_the_deadline_from_mis(question, before, mini_prod_world, understanding_on):  # noqa: F811
    """Срок печатает код из поля `deadline` строки прайса: без единицы — дни (решение 11.10)."""

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, question], f"ready-{len(before)}-{abs(hash(question))}")
    turn = record[-1]
    assert (turn["text"], turn["handoff"]) == (READINESS_KNOWN[question], False)


@pytest.mark.parametrize("before", _BEFORE, ids=["text", "after_price_button"])
@pytest.mark.parametrize("question", READINESS_UNKNOWN)
def test_readiness_without_deadline_is_honest_operator_not_price(question, before, mini_prod_world, understanding_on):  # noqa: F811
    """Срока нет или услуга не названа: не цена и не срок свободным текстом LLM, а честный перевод —
    тот же текст, что у справочного вопроса о сроках в пути результатов (прод #675)."""

    from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, question], f"ready-none-{len(before)}-{abs(hash(question))}")
    turn = record[-1]
    assert not re.search(r"\d\s*руб", turn["text"]), turn["text"]
    assert (turn["text"], turn["handoff"]) == (RESULT_TIMING_OPERATOR_TEXT, True)
