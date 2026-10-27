"""Понимание хода на метке «цена» (ход (б), шаг 1, L-14): производитель, выключатель, телеметрия.

Сквозной класс (список → корзина, срок → не цена) — `test_price_list_class.py`; эталон режимов
off / shadow / on — `test_price_turns_golden.py`. Здесь — то, чего сквозной тест не видит:
- встречный свип: полное название прайса как запрос — не список. Сплиттер режет по `,` `;` `и`
  `+` `/`, а они живут и внутри названий (110 из 404 строк мини-прод с запятой);
- разрез понимания = разрез корзины: понимание режет сырую реплику, корзина — после подстановок
  синонимов кода (L-15b, шаг 1);
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


@pytest.mark.xfail(
    strict=True,
    reason="STEP-2026-10-27-DEFAULTS-PROD, находка «явный перечень своих названий»: под `turn_understanding_one = on` (прод с "
    "22.10) правило K1 снимается у «явного перечня» (решение 21.10), и 8 из 398 названий мини-прод с запятой правило понимания "
    "режет на позиции с чужими строками («Собака, IgE» → «Кошка,IgE»). Пациент этого не видит (цена читает разметку; без LLM — "
    "«Уточните»), неверна телеметрия понимания (`tu_list`, `tu_rule`)",
)
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


# --- Разрез понимания = разрез корзины (L-15b, шаг 1; STEP-2026-10-13-L15B-1-PARSE-CACHE) ---------
#
# Понимание решает «реплика — список» по сырому тексту (`price_list_services(user_text)`), корзина
# режет тот же текст после подстановок синонимов кода (`_multi_price_items(_apply_service_synonyms
# (query))`, `price_info`). Защиты понимания только сужают (список → не список), поэтому на списке
# разрезы обязаны совпасть по составу: иначе понимание шлёт в корзину список, который она соберёт
# иначе или не соберёт, и выбор LLM получает одну строку на всю реплику («маршрут ≠ корзина», совет
# сессии 13.10). На стенде совпали все 83 хода-списка; расхождение возможно только через
# `_SERVICE_SYNONYM_SUBS` — поэтому каждая подстановка проверяется внутри позиции списка.

# Как пациенты пишут слова, которые подстановки кода меняют («забор» → «взятие», «электромиография»
# → «ЭМГ»). Новая подстановка без фразы здесь — красный `test_every_code_synonym_has_a_phrase_here`.
_SUBSTITUTED_PHRASES = (
    "забор крови из вены",
    "Забор крови",
    "заборы крови из пальца",
    "электромиография",
    "электромиографию нижних конечностей",
    "ЭЛЕКТРОМИОГРАФИЯ",
)
_LIST_AROUND = ("сколько стоит ферритин, {}, глюкоза", "{} и ферритин", "оак, оам, {}", "цена: глюкоза; {}")
_LISTS = (
    "сколько стоит ферритин, глюкоза, холестерин",
    "оак, оам",
    "ферритин и глюкоза",
    "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА",
    "сколько стоит ттг, т4 свободный",
    "Нужны цены на список анализов: ТТГ, ферритин, витамин D, глюкоза",
)


def _sub_patterns() -> list:
    from messengers_router.services._prices_helpers import _SERVICE_SYNONYM_SUBS

    return [pattern for pattern, _ in _SERVICE_SYNONYM_SUBS]


def _cuts(text: str) -> tuple[list[str], list[str]]:
    """(разрез понимания, разрез корзины) — названия позиций каталога; понимание: пусто — не список."""

    from messengers_router.services import _prices_helpers as helpers

    names, _ = helpers.price_list_services(text)
    items, _ = helpers._multi_price_items(helpers._apply_service_synonyms(text), helpers._catalog_rows(None))
    return list(names), [str(item["service_name"]) for item in items]


def _catalog_lists() -> list[str]:
    """Списки из коротких названий мини-прод (детерминированно): по три через запятую и через «и»."""

    plain = [name for name in _names() if len(name) < 30 and not any(mark in name for mark in (",", "(", " и ", "+", "/"))]
    step = max(1, len(plain) // 18)
    picked = plain[::step][:18]
    triples = [picked[i : i + 3] for i in range(0, len(picked) - 2, 3)]
    return [f"сколько стоит {', '.join(t)}" for t in triples] + [" и ".join(t[:2]) for t in triples]


def test_every_code_synonym_has_a_phrase_here():
    uncovered = [p.pattern for p in _sub_patterns() if not any(p.search(phrase) for phrase in _SUBSTITUTED_PHRASES)]
    assert uncovered == []


@pytest.mark.parametrize("pattern", _sub_patterns(), ids=lambda p: p.pattern)
def test_code_synonym_inside_a_list_does_not_change_the_cut(pattern, mini_prod_world):  # noqa: F811
    replies = [around.format(phrase) for phrase in _SUBSTITUTED_PHRASES if pattern.search(phrase) for around in _LIST_AROUND]
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        cuts = {text: _cuts(text) for text in replies}
    assert all(understood for understood, _ in cuts.values()), "реплика перестала быть списком — тест пуст"
    diverged = {text: cut for text, cut in cuts.items() if sorted(cut[0]) != sorted(cut[1])}
    assert diverged == {}


def test_list_cut_of_understanding_is_the_cut_of_the_basket(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        cuts = {text: _cuts(text) for text in (*_LISTS, *_catalog_lists())}
    lists = {text: cut for text, cut in cuts.items() if cut[0]}
    assert len(lists) >= len(cuts) * 3 // 4, "мало реплик-списков — тест пуст"
    diverged = {text: cut for text, cut in lists.items() if sorted(cut[0]) != sorted(cut[1])}
    assert diverged == {}


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


def test_shadow_writes_numbers_and_keys_and_does_not_act(mini_prod_world, mode, monkeypatch):  # noqa: F811
    from messengers_router import llm_runtime
    from tests.spans_oracle import SpansOracle

    mode("shadow")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(("ферритин", "глюкоза", "холестерин")))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        ((dbg, row),) = _turns(["сколько стоит ферритин, глюкоза, холестерин"], world)
    # Разбор реплики — единственный режим (T7 шаг 3б): услуги хода — фразы разбора (разметка `turn_spans` — оракул, прод с
    # деплоя 17; без LLM разреза по разделителям на пути прода нет), а не названия каталога, подставленные разрезом правил.
    assert dbg["understanding"]["services"] == ["ферритин", "глюкоза", "холестерин"]
    assert dbg["understanding"]["acted"] == []
    assert (row["tu_list"], row["tu_facet"], row["tu_rule"]) == (3, "price", "parse_picked")
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

