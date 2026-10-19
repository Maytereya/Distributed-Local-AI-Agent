"""«Где сделать …»: услугу инструмента адресов выбирает LLM, а не буквы (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, остаток).

Было: при пустой сущности услуги инструмент адресов (`services/addresses.py::address_info`) сам выводил её из реплики —
`extract_service_phrase` и лексический отбор строк прайса (`_select_address_price_rows`) — и места оказания решали
буквы: «где сделать магнитотерапию» → адрес лаборатории «Mg+ (магний)» (общий корень «магн»). Разбор хода к этому моменту
услугу уже снял (LLM её не подтвердила) или правила её не нашли.

Стало под `[MESSENGER_ROUTER] address_service_llm = on` (умолчание off — как было):
- услугу, которую LLM разбора хода уже не подтвердила, инструмент заново по буквам не ищет и LLM второй раз не зовёт —
  общий список филиалов, как у снятой услуги;
- иначе строки прайса, найденные правилами по фразе реплики, — кандидаты; какие из них спросил пациент, решает модуль
  выбора (`choose`, задача `address_service`): выбраны — места оказания только по ним; «ни одна» — услуга снимается,
  общий список филиалов; сбой LLM — как было.

Инвариант класса: место строки, совпавшей с репликой буквами, пациент получает только если её выбрала LLM; под off и при
сбое LLM — как было. Мир — `mini_prod` (в нём есть «Mg+ (магний)») плюс строки процедур и филиалы; места оказания — по
названию строки. LLM — подделка настоящего типа (`llm_runtime._OLLAMA_CLIENT`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import address_service, choice, llm_runtime
from messengers_router.choice import Option
from messengers_router.services import Services, _prices_helpers
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_ADDRESS_MARK = "место оказания которой спрашивает пациент"
_PRICE_MARK = "ответить пациенту на вопрос о цене"
_LAB = "г.Самара, ул.Победы, 83"  # место сдачи «Mg+ (магний)» в этом мире
_PHYSIO = "г.Самара, пр.Ленина, 5"  # физиотерапия и массаж спины
_EYE = "г.Самара, ул.Ново-Садовая, 180А"  # массаж век — у офтальмологов
_MASSAGE_BACK = "Массаж спины (1 сеанс)"
_MASSAGE_EYELID = "Массаж век (1 сеанс)"
_MAGNETO = "Магнитотерапия (1 процедура)"
_MG = "Mg+ (магний)"
_PLACES = {_MG: _LAB, _MASSAGE_BACK: _PHYSIO, _MASSAGE_EYELID: _EYE, _MAGNETO: _PHYSIO}
_BRANCHES = [_LAB, _PHYSIO, _EYE, "г.Самара, пр.Кирова, 223"]


class _LLM:
    """Отвечает на выбор услуги места (`address_service`: `decide(названия) → названия`) и, если задано, на выбор строки
    прайса разбора хода (`price(строки) → номера`); прочие места — «недоступна», как в мире гейта."""

    def __init__(self, decide: Callable[[list[str]], list[str]] | None, *, price: Callable[[dict[str, int]], list[int]] | None = None):
        self.decide = decide
        self.price = price
        self.address_calls: list[list[str]] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _ADDRESS_MARK in seen and self.decide is not None:
            # Варианты — в данных хода (роль user); в инструкции свои нумерованные правила.
            names = [rest for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
            self.address_calls.append(names)
            picks = [names.index(name) + 1 for name in self.decide(names) if name in names]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if _PRICE_MARK in seen and self.price is not None:
            rows: dict[str, int] = {}
            for line in seen.splitlines():
                head, dot, rest = line.partition(". ")
                if dot and head.isdigit() and " — " in rest:
                    rows[rest.rsplit(" — ", 1)[0]] = int(head)
            picks = self.price(rows)
            answer = {"picks": picks, "reason": "none" if picks else "other_service", "search_terms": []}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _only(*names: str) -> Callable[[list[str]], list[str]]:
    return lambda shown: [name for name in shown if name in names]


def _world(patch, switch: str, llm: _LLM | None) -> None:
    """Строки процедур с местом оказания (по названию) и филиалы Самары; выключатель; LLM."""

    from agent_logic_2.nayka_api import api_price
    from messengers_router.runtime_config import config as cfg

    real_load = api_price.load_price_by_region
    extra = [
        {"serviceId": 990001 + i, "priceUnitId": 155, "serviceHomecode": f"T{i}", "serviceName": name, "deadline": " ", "cost": 900.0}
        for i, name in enumerate((_MASSAGE_BACK, _MASSAGE_EYELID, _MAGNETO))
    ]

    def load(region_id, *args, **kwargs):
        return list(real_load(region_id, *args, **kwargs) or []) + [dict(row) for row in extra]

    def annotate(rows):
        out = []
        for row in rows or []:
            if isinstance(row, dict):
                place = _PLACES.get(str(row.get("serviceName") or ""))
                out.append({**row, "care_setting_address": place or "", "care_setting_addresses": [place] if place else []})
        return out

    async def regions(self):
        return [
            {"id": i, "name": f"Самара {addr}", "addressForSite": addr, "city": "Самара", "analysis": True, "doctorService": True}
            for i, addr in enumerate(_BRANCHES, 1)
        ]

    patch.setattr(api_price, "load_price_by_region", load)
    patch.setattr(_prices_helpers, "_annotate_price_rows_with_care_context", annotate)
    patch.setattr(Services, "_ensure_regions_loaded", regions)
    patch.setattr(cfg, "MR_ADDRESS_SERVICE_LLM", switch, raising=False)
    patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    if llm is not None:
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    choice._CACHE.clear()


def _turn(world_dir, monkeypatch, question: str, *, switch: str, llm: _LLM | None) -> dict[str, Any]:
    """Один ход через обработчик эндпоинта."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=f"addr-{switch}-{question}", text=question, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        dbg = (body.get("state_update") or {}).get("debug") or {}
        items = (dbg.get("evidence") or {}).get("items") or {}
        address = next((v for v in items.values() if isinstance(v, dict) and "addresses" in v), {})
        return {
            "text": str(body.get("text") or ""),
            "label": (dbg.get("decision") or {}).get("label"),
            "flags": (dbg.get("decision") or {}).get("flags") or [],
            "addresses": list(address.get("addresses") or []),
            "note": str(address.get("note") or ""),
        }

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        _world(patch, switch, llm)
        return asyncio.run(run())


def _tool(world_dir, monkeypatch, question: str, *, switch: str, llm: _LLM | None, entities: dict[str, Any] | None = None) -> dict[str, Any]:
    """Инструмент адресов при пустой сущности услуги — как его зовёт шаг ADDRESS, когда услугу разбор хода не нашёл или
    сняли правила (без решения LLM)."""

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        _world(patch, switch, llm)
        return asyncio.run(Services().address_info(question, dict(entities or {})))


# --- класс: «где сделать X» при пустой сущности — места только по строкам, выбранным LLM --------------------------------

_CASES = [
    # (реплика, что выбирает LLM, ожидаемые места)
    ("где сделать магнитотерапию", (_MAGNETO,), [_PHYSIO]),  # омоним: «Mg+ (магний)» — общий корень «магн»
    ("где можно сделать массаж спины", (_MASSAGE_BACK,), [_PHYSIO]),
    ("где сделать массаж век", (_MASSAGE_EYELID,), [_EYE]),
]


@pytest.mark.parametrize(("question", "picked", "places"), _CASES)
def test_places_come_only_from_the_rows_the_llm_picked(question, picked, places, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM(_only(*picked))
    out = _tool(mini_prod_world, monkeypatch, question, switch="on", llm=llm)

    assert len(llm.address_calls) == 1 and set(picked) <= set(llm.address_calls[0]), llm.address_calls
    assert out["addresses"] == places, out
    assert out["note"] == "address_info: priceUnits care-setting (llm)"


def test_letters_decided_the_place_before(mini_prod_world, monkeypatch):  # noqa: F811
    # Как было (и под off): «магнитотерапию» по буквам — строки и магнитотерапии, и анализа на магний: адрес лаборатории.
    out = _tool(mini_prod_world, monkeypatch, "где сделать магнитотерапию", switch="off", llm=_LLM(_only(_MAGNETO)))

    assert _LAB in out["addresses"], out


@pytest.mark.parametrize("question", [case[0] for case in _CASES] + ["где можно сделать массаж"])
def test_none_fit_drops_the_service_and_shows_all_branches(question, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM(lambda shown: [])
    out = _tool(mini_prod_world, monkeypatch, question, switch="on", llm=llm)

    assert llm.address_calls, "кандидаты были — выбор LLM должен был прозвучать"
    assert sorted(out["addresses"]) == sorted(_BRANCHES), out
    assert "care-setting" not in out["note"], out


@pytest.mark.parametrize("question", [case[0] for case in _CASES] + ["где можно сделать массаж"])
def test_llm_failure_and_switch_off_are_as_before(question, mini_prod_world, monkeypatch):  # noqa: F811
    off = _tool(mini_prod_world, monkeypatch, question, switch="off", llm=_LLM(lambda shown: []))
    down = _tool(mini_prod_world, monkeypatch, question, switch="on", llm=_LLM(None))

    assert (down["addresses"], down["note"]) == (off["addresses"], off["note"]), question
    assert off["note"] == "address_info: priceUnits care-setting", off


def test_service_entity_from_the_turn_is_not_judged_again(mini_prod_world, monkeypatch):  # noqa: F811
    # Сущность услуги есть (разбор хода её нашёл и, если надо, судил) — инструмент её не перепроверяет.
    llm = _LLM(lambda shown: [])
    entities = {"service_name": _MASSAGE_BACK}
    on = _tool(mini_prod_world, monkeypatch, "где сделать массаж спины", switch="on", llm=llm, entities=entities)
    off = _tool(mini_prod_world, monkeypatch, "где сделать массаж спины", switch="off", llm=None, entities=entities)

    assert llm.address_calls == [] and on == off, on


def test_general_address_question_does_not_call_the_llm(mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM(lambda shown: [])
    out = _tool(mini_prod_world, monkeypatch, "адреса филиалов", switch="on", llm=llm)

    assert llm.address_calls == [] and sorted(out["addresses"]) == sorted(_BRANCHES), out


# --- через эндпоинт: услугу, снятую LLM разбора хода, инструмент заново по буквам не ищет --------------------------------


def test_turn_llm_rejection_is_final_for_the_address_tool(mini_prod_world, monkeypatch):  # noqa: F811
    # LLM выбора строки прайса (разбор хода) не подтвердила услугу — снята. Под on: общий список филиалов, LLM места не
    # зовут; под off: инструмент по буквам находит строки снова — среди мест лаборатория анализа на магний.
    on_llm = _LLM(_only(_MAGNETO), price=lambda rows: [])
    on = _turn(mini_prod_world, monkeypatch, "где сделать магнитотерапию", switch="on", llm=on_llm)
    off = _turn(mini_prod_world, monkeypatch, "где сделать магнитотерапию", switch="off", llm=_LLM(None, price=lambda rows: []))

    assert on["label"] == off["label"] == "ADDRESS"
    assert address_service.llm_dropped_service(on["flags"]) and address_service.llm_dropped_service(off["flags"]), on["flags"]
    assert _LAB in off["addresses"], off
    assert on_llm.address_calls == []
    assert sorted(on["addresses"]) == sorted(_BRANCHES), on
    assert "__service_llm_dropped" not in json.dumps(on, ensure_ascii=False)


@pytest.mark.parametrize("question", ["где сделать магнитотерапию", "где можно сделать массаж спины"])
def test_endpoint_without_llm_is_as_before(question, mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(mini_prod_world, monkeypatch, question, switch="on", llm=_LLM(None))
    off = _turn(mini_prod_world, monkeypatch, question, switch="off", llm=_LLM(None))

    assert (on["text"], on["addresses"]) == (off["text"], off["addresses"]), question


def test_llm_dropped_service_needs_a_verdict():
    assert address_service.llm_dropped_service({"entity_dropped_unverified_service_name", "booking_service_none_fit"})
    assert not address_service.llm_dropped_service({"entity_dropped_unverified_service_name"})  # сняли правила, не LLM
    assert not address_service.llm_dropped_service({"entity_dropped_unverified_service_name", "booking_service_unavailable"})
    assert not address_service.llm_dropped_service({"booking_service_picked"})  # не снята


def test_prompt_renders_question_and_rows():
    system, user = choice.render(address_service.ADDRESS_SERVICE_TASK, "где сделать магнитотерапию", [Option(id="0", text=_MG)])

    assert "не похожие буквы" in system
    assert "«где сделать магнитотерапию»" in user and "1. Mg+ (магний)" in user
