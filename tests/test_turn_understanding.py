"""Понимание хода на метке «цена» (ход (б), шаг 1, L-14): производитель, выключатель, телеметрия.

Сквозной класс (список → корзина, срок → не цена) — `test_price_list_class.py`; эталон режимов
off / shadow / on — `test_price_turns_golden.py`. Здесь — то, чего сквозной тест не видит:
- встречный свип: полное название прайса как запрос — не список. Сплиттер режет по `,` `;` `и`
  `+` `/`, а они живут и внутри названий (110 из 404 строк мини-прод с запятой);
- свойство решают существующие детекторы, и «за сколько» без слова о сроке — цена;
- тень не действует, понимание не переживает ход, телеметрия — только числа и ключи.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from evalbench import world as ev_world
from messengers_router import turn_understanding as tu
from tests.test_price_turns_golden import FIXTURE, mini_prod_world  # noqa: F401 — фикстура модуля


@pytest.fixture
def mode(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    def _set(value: str) -> None:
        monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", value, raising=False)

    return _set


def _names() -> list[str]:
    path = FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl"
    return sorted({json.loads(line)["serviceName"].strip() for line in path.open(encoding="utf-8")})


def test_full_price_name_is_one_service_not_a_list(mini_prod_world):  # noqa: F811
    from messengers_router.services._prices_helpers import price_list_services

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        chopped = {
            query: got
            for name in _names()
            for query in (name, f"сколько стоит {name}")
            if (got := price_list_services(query)[0])
        }
    assert chopped == {}


@pytest.mark.parametrize(
    ("text", "facet"),
    [
        ("когда будет готов анализ на ферритин", "readiness"),
        ("сколько делается анализ на ферритин", "readiness"),
        ("за сколько дней будет готов ферритин", "readiness"),
        # «За сколько» без слова о сроке — вопрос о цене, хотя регулярка снятия фразы его берёт.
        ("за сколько можно сдать ферритин", None),
        ("сколько стоит и когда будет готов ферритин", "price"),
        ("как подготовиться к ферритину", "prepare"),
        ("сколько стоит ферритин", "price"),
    ],
)
def test_facet_is_decided_by_existing_detectors(text, facet):
    assert tu.price_turn_facet(text)[0] == facet


def test_readiness_question_is_one_service_not_a_basket(mini_prod_world):  # noqa: F811
    """Слова о сроке не становятся позициями корзины («Вакцинация … готов ли ваш организм?»)."""

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        got = tu.understand_price_turn("когда будет готов анализ на ферритин и глюкоза")
    assert got.facet == "readiness"
    assert set(got.service_names) <= {"Ферритин", "Глюкоза"}


def test_list_is_understood_once_with_catalog_names(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        got = tu.understand_price_turn("сколько стоит ферритин, глюкоза, холестерин")
    assert got.is_list and got.services_rule == "list_split"
    assert got.service_names == ("Ферритин", "Глюкоза", "Липидограмма")
    assert got.facet == "price"


def _turns(texts: list[str], world: ev_world.FrozenWorld) -> list[tuple[dict, dict | None]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run():
        memory, services, out = MemoryStore(), Services(), []
        for text in texts:
            resp = await messenger_generate_once(MessengerGenerateRequest(session_id="tu", text=text, debug=True), memory=memory, services=services)
            body = json.loads(resp.body)
            out.append(((body.get("state_update") or {}).get("debug") or {}, world.take_telemetry()))
        return out

    return asyncio.run(run())


def test_shadow_writes_numbers_and_keys_and_does_not_act(mini_prod_world, mode):  # noqa: F811
    mode("shadow")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        ((dbg, row),) = _turns(["сколько стоит ферритин, глюкоза, холестерин"], world)
    assert dbg["understanding"]["services"] == ["Ферритин", "Глюкоза", "Липидограмма"]
    assert dbg["understanding"]["acted"] == []
    assert (row["tu_list"], row["tu_facet"], row["tu_rule"]) == (3, "price", "list_split")
    assert "tu_acted" not in row
    # В телеметрии — ни одного значения сущности (решение 08.10).
    assert "Ферритин" not in json.dumps(row, ensure_ascii=False)


def test_on_records_which_readers_acted(mini_prod_world, mode):  # noqa: F811
    mode("on")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        ((dbg, row),) = _turns(["сколько стоит ферритин, глюкоза, холестерин"], world)
    assert "planner_list_to_cart" in row["tu_acted"]
    assert dbg["understanding"]["acted"] == row["tu_acted"]


def test_understanding_does_not_outlive_its_turn(mini_prod_world, mode):  # noqa: F811
    mode("on")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        (_, first), (dbg, row) = _turns(["сколько стоит ферритин, глюкоза", "привет"], world)
    assert first["tu_list"] == 2
    assert dbg.get("understanding") is None
    assert "tu_list" not in row and "tu_acted" not in row


def test_off_builds_nothing(mini_prod_world, mode):  # noqa: F811
    mode("off")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        ((dbg, row),) = _turns(["сколько стоит ферритин, глюкоза"], world)
    assert dbg.get("understanding") is None
    assert "tu_list" not in row


def test_unknown_switch_value_falls_back_to_shadow(mode):
    mode("да")
    assert tu.mode() == "shadow"


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("1-2", "1–2 дня"),  # без единицы — дни (решение владельца 11.10, без «рабочих»)
        ("11-13", "11–13 дней"),
        ("1", "1 день"),
        ("21", "21 день"),
        ("до 7", "до 7 дней"),
        ("3-4 часа", "3–4 часа"),  # явная единица — дословно
        ("До 30 дней", "До 30 дней"),
        ("1-3 ч", "1–3 ч"),
        (" ", None),  # пусто — честный перевод
    ],
)
def test_deadline_is_printed_by_code_in_days_unless_unit_given(raw, shown):
    from messengers_router.services._prices_helpers import format_deadline

    assert format_deadline(raw) == shown

