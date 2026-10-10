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
- вопрос о сроке готовности не получает цену — и сроков свободным текстом LLM тоже: срока бот
  не знает, поэтому честный перевод на оператора (как кнопка «Когда будет готов»).

Ход (б), шаг 1 (L-14): `услуги[]` + `свойство` на метке «цена», под выключателем
`turn_understanding_price`. Тесты включают его явно (`on`): по умолчанию шаг в тени, и тогда
поведение прежнее — его держит эталон `test_price_turns_golden.py` (режимы off и shadow).
До шага 14 из 16 мест списка и 2 из 5 формулировок срока были красными.
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

READINESS = [
    "когда будет готов анализ на ферритин",
    "через сколько будет готов ферритин",
    "сколько делается анализ на ферритин",
    "сколько готовится ферритин",
    "сколько готовятся анализы",
    "когда будет готов анализ на ферритин и глюкозу",
]


@pytest.fixture
def understanding_on(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "on", raising=False)


@pytest.mark.parametrize("items", list(LISTS))
@pytest.mark.parametrize("path", sorted(PATHS))
def test_every_listed_service_gets_its_price(path, items, mini_prod_world, understanding_on):  # noqa: F811
    before, template = PATHS[path]
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, template.format(items)], f"list-{path}-{abs(hash(items))}")
    answer = record[-1]["text"].lower()
    missing = [name for name in LISTS[items] if name not in answer]
    assert missing == [], record[-1]["text"]
    assert "руб" in answer, record[-1]["text"]


@pytest.mark.parametrize("before", [(), (("menu.price.test", "Цена анализа"),)], ids=["text", "after_price_button"])
@pytest.mark.parametrize("question", READINESS)
def test_readiness_question_gets_honest_operator_not_price(question, before, mini_prod_world, understanding_on):  # noqa: F811
    """Срок бот не знает (NO-TURNAROUND): не цена и не сроки свободным текстом LLM, а честный
    перевод — тот же текст, что у справочного вопроса о сроках в пути результатов (прод #675)."""

    from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, question], f"ready-{len(before)}-{abs(hash(question))}")
    turn = record[-1]
    assert "руб" not in turn["text"].lower(), turn["text"]
    assert (turn["text"], turn["handoff"]) == (RESULT_TIMING_OPERATOR_TEXT, True)
