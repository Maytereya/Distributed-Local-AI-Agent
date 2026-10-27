"""T7: защиты запасного пути цены K1 и K3 снимаются выключателями (облако 19.10, задача B; по умолчанию off — как было).

- K1 — защиты `services/_prices_helpers.price_list_services`: «одно название прайса покрывает реплику» (`one_catalog_name`)
  и «сплошной список — только распознанный целиком» (`segment_unrecognized`). Выключатель `t7_drop_k1`.
- K3 — гард третьего стыка корзины (`_multi_price_items`: условие П2 в остатке реплики гасит корзину). Выключатель
  `t7_drop_k3`.

Под разбором хода обе эвристики решают только на пути правил (`turn_parse.acting_for` = None). Риски снятия из брифа:
«когда будет готов анализ на ферритин» → корзина из двух позиций, «Глюкозотолерантный тест (глюкоза+ C-пептид)» → две
позиции (K1); «по ОМС» → платные позиции (K3, П2).

Инварианты (умолчания кода = прод: разметка `turn_spans`, цена читает разбор хода; STEP-2026-10-27-DEFAULTS-PROD):
- под on ответ на реплики рисков при лежащей LLM — тот же, что под off (через обработчик эндпоинта, мини-прод): до
  эвристик реплика хода не доходит — без LLM цена только дословному названию, иначе «Уточните название»;
- выключатель действительно снимает эвристику (`price_list_services` / `_multi_price_items` напрямую) — это и есть
  оставшийся путь правил (вход читателя — не реплика хода: дозаполнение, контекст, срок готовности; `too_many`).

Тесты запасного пути старого генератора окон (`turn_spans_llm = off`: эталон `price_turns_parse_fallback.json`) удалены
вместе с `test_turn_parse_fallback_class` при перевороте умолчаний на прод (27.10).
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router.services import _prices_helpers as H
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

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


def _answer(text: str, key: str, value: str, monkeypatch, world_dir) -> list[dict]:
    """Ход через эндпоинт при лежащей LLM (мир гейта) с выключателем `key` = `value` — что видит пациент: ответ, перевод,
    висящий вопрос, услуга сессии (инструмент плана эвристика списка меняет: корзина или одна услуга — ответ тот же)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, key, value, raising=False)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        record = run_dialog([text], f"t7-{key}-{value}-{len(text)}")
        attempts = list(world.net.attempts)
    assert attempts == []
    return [{key: turn[key] for key in ("text", "handoff", "pending", "service_name")} for turn in record]


@pytest.mark.parametrize(("key", "text"), [*(("MR_T7_DROP_K1", t) for t in _K1_RISKS), *(("MR_T7_DROP_K3", t) for t in _K3_RISKS)])
def test_llm_down_solid_text_answers_the_same_without_the_heuristic(key, text, mini_prod_world, monkeypatch):  # noqa: F811
    off = _answer(text, key, "off", monkeypatch, mini_prod_world)
    on = _answer(text, key, "on", monkeypatch, mini_prod_world)

    assert on == off


def _catalog(world_dir):
    with ev_world.FrozenWorld(world_dir, "day"):
        return H._catalog_rows(None)


def test_k1_switch_removes_both_guards_on_the_rules_path(mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    rows = _catalog(mini_prod_world)
    covering = "сколько стоит глюкозотолерантный тест, глюкоза и C-пептид"
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
