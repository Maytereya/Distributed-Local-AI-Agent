"""T7: защиты запасного пути цены K1 и K3 снимаются выключателями (облако 19.10, задача B; по умолчанию off — как было).

- K1 — защиты `services/_prices_helpers.price_list_services`: «одно название прайса покрывает реплику» (`one_catalog_name`)
  и «сплошной список — только распознанный целиком» (`segment_unrecognized`). Выключатель `t7_drop_k1`.
- K3 — гард третьего стыка корзины (`_multi_price_items`: условие П2 в остатке реплики гасит корзину). Выключатель
  `t7_drop_k3`.

Под разбором хода (`turn_parse`, единственный режим) обе эвристики решают только на пути правил (`turn_parse.acting_for`
= None); сбой LLM разбора с 16.10 ведёт политика сбоя разбора (`_turn_phrases.fallback_split`: разрез данными МИС, иначе
реплика целиком) — Q2 отменено. Риски снятия из брифа: «когда будет готов анализ на ферритин» → корзина из двух позиций,
«Глюкозотолерантный тест (глюкоза+ C-пептид)» → две позиции (K1); «по ОМС» → платные позиции (K3, П2).

Инварианты (через обработчик эндпоинта, мини-прод, LLM недоступна, сплошной текст — «сначала тест, потом снятие»):
- под on ответы на реплики рисков и на весь корпус эталона запасного пути (`price_turns_parse_fallback.json`) — те же,
  что под off: до эвристик эти реплики больше не доходят;
- выключатель действительно снимает эвристику (`price_list_services` / `_multi_price_items` напрямую) — это и есть
  оставшийся путь правил (вход читателя — не реплика хода: дозаполнение, контекст, срок готовности; `too_many`).
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router.services import _prices_helpers as H
from tests import test_turn_parse_fallback_class as fallback
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_K1_RISKS = [
    "когда будет готов анализ на ферритин",
    "Глюкозотолерантный тест (глюкоза+ C-пептид)",
    "сколько стоит глюкозотолерантный тест (глюкоза+ C-пептид)",
    "сколько стоит Билирубин общий прямой непрямой",
    "сколько стоит прием, осмотр и консультация терапевта",
    "сколько стоит сдать общий анализ крови ферритин глюкоза натощак",
    "сколько стоит оак оам ферритин глюкоза узи",
]
_K3_RISKS = [
    "сколько стоит ферритин глюкоза по омс",
    "сколько стоит приём терапевта по ОМС",
    "сколько стоит мазок на микрофлору с поверхности задней стенки глотки",
    "ферритин и глюкоза по омс сколько стоят",
]


def _answer(text: str, config: str, extra: dict[str, str], monkeypatch, world_dir, *, case: str = "") -> dict:
    """Ход через эндпоинт обвязкой эталона запасного пути: набор `config` плюс выключатели `extra`; `case` — случай
    эталона (с его оракулом), иначе — временный случай с этой репликой (LLM недоступна)."""

    name = case or f"t7-{len(text)}-{config}"
    sets = f"t7-{config}-{sorted(extra.items())}"
    if not case:
        fallback.CASES[name] = fallback.Case(text, "any", "T7")
    fallback._SETS[sets] = {**fallback._SETS[config], **extra}
    try:
        [turn] = fallback._dialog(name, sets, [text], monkeypatch, world_dir)
    finally:
        if not case:
            fallback.CASES.pop(name, None)
        fallback._SETS.pop(sets, None)
    return turn


@pytest.mark.parametrize("config", ["deploy_parse", "prod"])
@pytest.mark.parametrize(("key", "text"), [*(("MR_T7_DROP_K1", t) for t in _K1_RISKS), *(("MR_T7_DROP_K3", t) for t in _K3_RISKS)])
def test_llm_down_solid_text_answers_the_same_without_the_heuristic(key, text, config, mini_prod_world, monkeypatch):  # noqa: F811
    off = _answer(text, config, {key: "off"}, monkeypatch, mini_prod_world)
    on = _answer(text, config, {key: "on"}, monkeypatch, mini_prod_world)

    assert on == off


@pytest.mark.parametrize("key", ["MR_T7_DROP_K1", "MR_T7_DROP_K3"])
@pytest.mark.parametrize("name", sorted(fallback.CASES))
def test_fallback_golden_corpus_is_unchanged_without_the_heuristic(key, name, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _answer(fallback.CASES[name].text, "prod", {key: "on"}, monkeypatch, mini_prod_world, case=name)

    assert [turn] == fallback._load()[f"{name}-prod"]


def _catalog(world_dir):
    with ev_world.FrozenWorld(world_dir, "day"):
        return H._catalog_rows(None)


def test_k1_switch_removes_both_guards_on_the_rules_path(mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    rows = _catalog(mini_prod_world)
    covering = "сколько стоит прием, осмотр и консультация терапевта"
    run_on = "сколько стоит оак оам ферритин глюкоза узи"
    monkeypatch.setattr(cfg, "MR_T7_DROP_K1", "off", raising=False)
    assert H.price_list_services(covering, rows=rows) == ((), "one_catalog_name")
    assert H.price_list_services(run_on, rows=rows) == ((), "segment_unrecognized")
    monkeypatch.setattr(cfg, "MR_T7_DROP_K1", "on", raising=False)
    assert H.price_list_services(covering, rows=rows)[1] == "list_split"
    assert H.price_list_services(run_on, rows=rows)[1] == "list_segment"


def test_k3_switch_removes_the_third_seam_guard(mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    rows = _catalog(mini_prod_world)
    text = "сколько стоит ферритин глюкоза по омс"
    monkeypatch.setattr(cfg, "MR_T7_DROP_K3", "off", raising=False)
    assert H._multi_price_items(text, rows) == ([], [])
    monkeypatch.setattr(cfg, "MR_T7_DROP_K3", "on", raising=False)
    items, _unrecognized = H._multi_price_items(text, rows)
    assert [item["service_name"] for item in items] == ["Ферритин", "Глюкоза"]
