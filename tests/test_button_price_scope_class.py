"""Ответ на вопрос кнопки цены ищется в её области прайса (BUG-2026-10-25-BUTTON-PRICE-SCOPE, выключатель
`button_price_scope`; ТЗ 17.08, «сужение поиска по теме»).

Класс: «Цена приёма врача» → «кардиограмма» → «С-реактивный белок (кардио)», «щитовидка» → профиль анализов: строка
чужого вида услуги, хотя пациент нажатием сказал, что спрашивает о приёме. Вид услуги — данные прайса МИС: у анализов
есть срок готовности, у приёмов, процедур и УЗИ — нет.

Инвариант класса (свип по всем названиям мини-прайса фикстуры): у ответа на вопрос кнопки цены с областью — если в
области есть кандидат на фразу, ни одной строки вне области; нет — ответ тот же, что без кнопки (весь прайс). Без области
(выключатель, режим `doctor` для «Цены анализа», вопрос не кнопки) — как было. Встречный: дословное название вне области
(синоним МИС, точное имя) — по-прежнему его строки.

Свип — при лежащей LLM, как в мире гейта. Под `price_by_parse = on` (прод с 26.10, Ш4) без решения LLM строки слабой фразы с
ценой не показываются вовсе, и свип проверяет в основном сильные фразы. Экземпляры класса — с подделкой выбора строки «выбрала всё,
что ей показали» (`picks_all`): ответ — ровно строки, которые область пустила к выбору LLM, т. е. проверяется сама область.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import button_flow, choice, llm_runtime
from messengers_router.services import prices
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import FIXTURE, mini_prod_world  # noqa: F401 — фикстура

_ROWS = [json.loads(line) for line in next(Path(FIXTURE).rglob("price_region_3_*.jsonl")).open(encoding="utf-8") if line.strip()]
# Фразы свипа: первые два слова каждого названия (как пациент называет услугу коротко), без повторов.
_PHRASES = sorted({" ".join(str(r["serviceName"]).split()[:2]).strip(" ,.(") for r in _ROWS} - {""})


def _price(world_dir, query: str, scope: str = "") -> dict[str, Any]:
    from messengers_router.services import Services

    entities: dict[str, Any] = {"__price_question": True, **({"__price_scope": scope} if scope else {})}

    async def run() -> dict[str, Any]:
        return await Services().price_info(query, entities)

    with ev_world.FrozenWorld(world_dir, "day"):
        llm_runtime._BREAKER = llm_runtime._Breaker()
        choice._CACHE.clear()
        return asyncio.run(run())


def _ids(payload: dict[str, Any]) -> set[int]:
    rows = list(payload.get("prices") or []) + list(payload.get("family_variants") or [])
    for item in payload.get("items") or []:
        rows += list((item or {}).get("rows") or [])
    return {int(r["serviceId"]) for r in rows if isinstance(r, dict) and r.get("serviceId") is not None}


_LAB = {int(r["serviceId"]) for r in _ROWS if str(r.get("deadline") or "").strip()}
_PRICE_SELECT_MARK = "ответить пациенту на вопрос о цене"


class _PicksAll:
    """LLM выбора строки цены отмечает все показанные ей строки; прочие места — «недоступна»."""

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        from ollama import GenerateResponse

        if _PRICE_SELECT_MARK not in f"{system}\n\n{prompt}":
            raise ConnectionError("тест: LLM недоступна для этого места")
        picks = [int(head) for head, dot, _rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks, "reason": "none", "search_terms": []}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


@pytest.fixture
def picks_all(monkeypatch):
    from tests.spans_oracle import keep_llm_breaker_closed

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PicksAll())
    keep_llm_breaker_closed(monkeypatch)


@pytest.mark.parametrize("scope", ["doctor", "tests"])
def test_scoped_answer_has_no_row_of_the_other_kind(scope, mini_prod_world):  # noqa: F811
    narrowed = fallback = 0
    for phrase in _PHRASES:
        query = f"сколько стоит {phrase}"
        base = _price(mini_prod_world, query)
        got = _price(mini_prod_world, query, scope)
        rows = [r for r in _ROWS if prices._in_price_scope(r, scope)]
        if prices._button_scope_rows({"__price_scope": scope}, query, phrase, phrase, _ROWS) is None or not rows:
            fallback += 1
            assert _ids(got) == _ids(base), (scope, phrase)  # в области нет кандидата — весь прайс, как без кнопки
            continue
        narrowed += 1
        foreign = _ids(got) & (_LAB if scope == "doctor" else ({int(r["serviceId"]) for r in _ROWS} - _LAB))
        assert not foreign, (scope, phrase, sorted(foreign))
    assert narrowed and fallback, (narrowed, fallback)  # свип видит обе ветки


def test_no_scope_keeps_the_answer(mini_prod_world):  # noqa: F811
    for phrase in _PHRASES[:40]:
        query = f"сколько стоит {phrase}"
        assert _ids(_price(mini_prod_world, query, "")) == _ids(_price(mini_prod_world, query)), phrase


def test_class_instances(mini_prod_world, picks_all):  # noqa: F811
    """Экземпляры класса: «кардио…» на «Цену приёма врача» — не анализ «С-реактивный белок (кардио)»; «щитовидная железа» —
    УЗИ, не профиль анализов; на «Цену анализа» — наоборот; «ЭКГ» на «Цену анализа» — ЭКГ (в анализах кандидата нет)."""

    crp = 13267
    assert crp not in _ids(_price(mini_prod_world, "сколько стоит кардио", "doctor"))
    thyroid_doctor = _ids(_price(mini_prod_world, "сколько стоит щитовидная железа", "doctor"))
    thyroid_tests = _ids(_price(mini_prod_world, "сколько стоит щитовидная железа", "tests"))
    assert thyroid_doctor and not thyroid_doctor & _LAB, thyroid_doctor
    assert thyroid_tests and thyroid_tests <= _LAB, thyroid_tests
    assert 1455 in _ids(_price(mini_prod_world, "сколько стоит ЭКГ", "tests"))


def test_exact_name_outside_scope_is_not_narrowed():
    """Встречный (свип живого прайса 25.10: «Фгдс» на «Цену анализа» → «профиль анализов перед ФГДС» вместо ФГДС): услуга
    названа дословно (синоним МИС, точное имя, код) только вне области — область не сужает, даже если по словам в ней
    нашёлся анализ; без такого совпадения — сужает."""

    rows = [
        {"serviceId": 1, "serviceHomecode": "9.1", "serviceName": "Колоноскопия", "deadline": " ", "cost": 5000.0, "priceUnitId": 291},
        {"serviceId": 2, "serviceHomecode": "9.2", "serviceName": "Профиль анализов перед колоноскопией", "deadline": "1", "cost": 1500.0, "priceUnitId": 145},
        {"serviceId": 3, "serviceHomecode": "9.3", "serviceName": "Ферритин", "deadline": "1", "cost": 500.0, "priceUnitId": 135},
    ]
    assert prices._button_scope_rows({"__price_scope": "tests"}, "сколько стоит колоноскопия", "колоноскопия", "", rows) is None
    narrowed = prices._button_scope_rows({"__price_scope": "tests"}, "сколько стоит профиль перед колоноскопией", "профиль перед колоноскопией", "", rows)
    assert narrowed is not None and {r["serviceId"] for r in narrowed} == {2, 3}


# --------------------------------------------------------------------------- область из нажатия (планировщик)

_MARK_ANY = "Сообщение пациента"


def _dialog(world_dir, monkeypatch, turns, *, mode: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"scope-{mode}-{json.dumps(turns, ensure_ascii=False)}"

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append({"text": str(body.get("text") or ""), "telemetry": world.take_telemetry() or {}})
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {"MR_PENDING_BUTTONS": "on", "MR_CHOOSE_CACHE_TTL_S": 0, "MR_BUTTON_PRICE_SCOPE": mode}.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM({_MARK_ANY: json.dumps({"picks": []})}))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


@pytest.mark.parametrize(
    ("mode", "button_id", "scope"),
    [
        ("doctor", "menu.price.doctor", "doctor"),
        ("doctor", "menu.price.test", None),  # «Цену анализа» режим `doctor` не сужает
        ("all", "menu.price.test", "tests"),
        ("all", "menu.tests.price", "tests"),
        ("off", "menu.price.doctor", None),
    ],
)
def test_button_question_answer_gets_its_scope(mode, button_id, scope, mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), "щитовидная железа"], mode=mode)
    assert on[1]["telemetry"].get("price_scope") in ({scope, "fallback"} if scope else {None}), on[1]["telemetry"]
    if scope:
        assert button_flow.price_scope({"button": button_id, "label": "PRICE"}) in {scope, ""}


def test_typed_price_question_has_no_scope(mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, ["сколько стоит прием", "щитовидная железа"], mode="all")
    assert "price_scope" not in on[1]["telemetry"], on[1]["telemetry"]
