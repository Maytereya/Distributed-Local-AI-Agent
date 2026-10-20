"""«Где сделать X»: выбор LLM доходит до места оказания (BUG-2026-10-20-ADDRESS-MULTI-PICK-AS-DROPPED).

Было (карточка прод-LLM 20.10, `address_service_llm = on`): «где сделать магнитотерапию», «где сделать массаж спины» —
общий список всех филиалов (под off — «пр. Ленина, 5»). LLM разбора хода выбрала несколько строк одной услуги (варианты:
«Массаж … (30 / 45 / 60 мин)»), для записи это выбор пациента (`judge_booking_service` → `choose`), а путь «где сделать»
(`entity_grounder._apply_service_verdict`) считал любое «не одно название» снятой услугой — и под on инструмент адресов
отдавал все филиалы, хотя LLM услугу нашла.

Инвариант класса: услугу, названную пациентом и выбранную LLM (разбором хода — одной строкой или несколькими — или судом
инструмента, в том числе вторым кругом по терминам), бот показывает местами оказания ЭТИХ строк — не адресами другой
услуги (анализ «Mg+ (магний)» на «магнитотерапию») и не общим списком всех филиалов. Общий список — только вопрос без
услуги или «ни одна» LLM в обоих кругах (решение владельца 19.10, п. 2). Под off и при сбое LLM — как было.

Мир — `mini_prod` (в нём есть «Mg+ (магний)») плюс строки процедур с местами оказания по названию и филиалы Самары. LLM —
подделка настоящего типа (`llm_runtime._OLLAMA_CLIENT`, `ollama.GenerateResponse`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from messengers_router.services import Services, _prices_helpers
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_ADDRESS_MARK = "место оказания которой спрашивает пациент"
_PRICE_MARK = "ответить пациенту на вопрос о цене"
_LAB = "г.Самара, ул.Победы, 83"  # место сдачи «Mg+ (магний)»
_LENINA = "г.Самара, пр.Ленина, 5"
_KIROVA = "г.Самара, пр.Кирова, 223"
_EYE = "г.Самара, ул.Ново-Садовая, 180А"
_BRANCHES = [_LAB, _LENINA, _EYE, _KIROVA, "г.Самара, ул.Солнечная, 53"]
_MG = "Mg+ (магний)"
_PLACES = {
    _MG: _LAB,
    "Магнитотерапия (1 поле)": _LENINA,
    "Магнитотерапия (2 поля)": _KIROVA,
    "Массаж спины (30 мин)": _LENINA,
    "Массаж спины (60 мин)": _LENINA,
    "Массаж век (1 сеанс)": _EYE,
    "Электрофорез лекарственный (1 процедура)": _KIROVA,
    "Электрофорез лекарственный (10 процедур)": _KIROVA,
}


class _LLM:
    """Выбор строки прайса разбора хода: `price(названия) → названия`; выбор места инструмента: `address(названия) →
    (названия, термины)`; прочие места — «недоступна», как в мире гейта. Ответы — по схеме модуля выбора."""

    def __init__(
        self,
        *,
        price: Callable[[list[str]], list[str]] | None = None,
        address: Callable[[list[str]], tuple[list[str], list[str]]] | None = None,
    ):
        self.price = price
        self.address = address
        self.address_calls: list[list[str]] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        lines = [line.partition(". ") for line in prompt.splitlines()]
        numbered = [(int(head), rest) for head, dot, rest in lines if dot and head.isdigit()]
        if _ADDRESS_MARK in seen and self.address is not None:
            names = [rest for _, rest in numbered]
            self.address_calls.append(names)
            picked, terms = self.address(names)
            answer = {"picks": [names.index(n) + 1 for n in picked if n in names], "search_terms": terms}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        if _PRICE_MARK in seen and self.price is not None:
            rows = {rest.rsplit(" — ", 1)[0]: index for index, rest in numbered if " — " in rest}
            picks = sorted(rows[name] for name in self.price(list(rows)) if name in rows)
            answer = {"picks": picks, "reason": "none" if picks else "other_service", "search_terms": []}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _starting(prefix: str) -> Callable[[list[str]], list[str]]:
    return lambda shown: [name for name in shown if name.startswith(prefix)]


def _world(patch, switch: str, llm: _LLM | None) -> None:
    from agent_logic_2.nayka_api import api_price
    from messengers_router.runtime_config import config as cfg

    real_load = api_price.load_price_by_region
    extra = [
        {"serviceId": 991001 + i, "priceUnitId": 155, "serviceHomecode": f"A{i}", "serviceName": name, "deadline": " ", "cost": 900.0}
        for i, name in enumerate(n for n in _PLACES if n != _MG)
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
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=f"addr-class-{switch}-{question}", text=question, debug=True)
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
            "entities_used": address.get("entities_used") or {},
            "state": {k: v for k, v in (body.get("state_update") or {}).items() if k != "debug"},
        }

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        _world(patch, switch, llm)
        return asyncio.run(run())


def _tool(world_dir, monkeypatch, question: str, *, switch: str, llm: _LLM | None) -> dict[str, Any]:
    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        _world(patch, switch, llm)
        return asyncio.run(Services().address_info(question, {}))


# --- класс: LLM разбора хода выбрала несколько строк услуги — места оказания этих строк ----------------------------------

_MULTI = [
    # (реплика, семейство строк, которое выбирает LLM разбора хода, места этих строк)
    ("где сделать магнитотерапию", "Магнитотерапия", [_LENINA, _KIROVA]),  # омоним «Mg+ (магний)»
    ("где сделать массаж спины", "Массаж спины", [_LENINA]),  # рядом «Массаж век» — у офтальмологов
    ("где сделать электрофорез лекарственный", "Электрофорез лекарственный", [_KIROVA]),
]


@pytest.mark.parametrize(("question", "family", "places"), _MULTI)
def test_several_rows_picked_by_the_turn_give_their_places(question, family, places, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM(price=_starting(family), address=lambda shown: ([], []))
    out = _turn(mini_prod_world, monkeypatch, question, switch="on", llm=llm)

    assert out["label"] == "ADDRESS", out["flags"]
    assert "booking_service_picked" in out["flags"], out["flags"]  # выбрала LLM, несколько названий
    assert sorted(out["addresses"]) == sorted(places), out
    assert sorted(out["addresses"]) != sorted(_BRANCHES), "общий список филиалов вместо мест услуги"
    assert _LAB not in out["addresses"] and _EYE not in out["addresses"], out
    assert all(place.split(", ", 1)[1] in out["text"] for place in places), out["text"]
    assert llm.address_calls == [], "выбор хода уже решил — второй вызов LLM не нужен"
    # Служебный ключ — вход хода, не сущность диалога: в состояние и в `entities_used` (промпт ответа) не попадает.
    assert "_address_service_picks" not in json.dumps([out["state"], out["entities_used"]], ensure_ascii=False)


@pytest.mark.parametrize(("question", "family"), [case[:2] for case in _MULTI])
def test_switch_off_is_as_before(question, family, mini_prod_world, monkeypatch):  # noqa: F811
    # Под off выбор хода инструмент не видит — места по буквам реплики, как было.
    off = _turn(mini_prod_world, monkeypatch, question, switch="off", llm=_LLM(price=_starting(family)))
    rules = _tool(mini_prod_world, monkeypatch, question, switch="off", llm=None)

    assert off["addresses"] == rules["addresses"], (off, rules)


# --- класс: суд инструмента «ни одна» среди строк по буквам — второй круг по терминам LLM ---------------------------------


def test_round_two_by_llm_terms_finds_the_service(mini_prod_world, monkeypatch):  # noqa: F811
    # «магнитолечение»: по буквам правила находят только анализы с магнием («Mg+ (магний)», литогенные субстанции мочи), а
    # строк магнитотерапии — нет. LLM отвергает кандидатов и называет термин; по нему правила находят магнитотерапию, LLM
    # выбирает — места оказания её, а не общий список и не лаборатория.
    def address(shown: list[str]) -> tuple[list[str], list[str]]:
        magneto = [name for name in shown if name.startswith("Магнитотерапия")]
        return (magneto, []) if magneto else ([], ["Магнитотерапия"])

    question = "где сделать магнитолечение"
    llm = _LLM(address=address)
    out = _tool(mini_prod_world, monkeypatch, question, switch="on", llm=llm)
    no_terms = _tool(mini_prod_world, monkeypatch, question, switch="on", llm=_LLM(address=lambda shown: ([], [])))
    off = _tool(mini_prod_world, monkeypatch, question, switch="off", llm=None)

    assert len(llm.address_calls) == 2, llm.address_calls
    first, second = llm.address_calls
    assert _MG in first and not any(n.startswith("Магнитотерапия") for n in first), first
    assert not set(first) & set(second), "второй круг повторил варианты первого"
    assert sorted(out["addresses"]) == sorted([_LENINA, _KIROVA]), out
    assert out["note"] == "address_info: priceUnits care-setting (llm)"
    # Без терминов «ни одна» — общий список, как решил владелец 19.10; под off — места по буквам (лаборатория).
    assert sorted(no_terms["addresses"]) == sorted(_BRANCHES), no_terms
    assert _LAB in off["addresses"], off


def test_round_two_without_a_fit_keeps_the_general_list(mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM(address=lambda shown: ([], ["Несуществующая процедура"]))
    out = _tool(mini_prod_world, monkeypatch, "где сделать массаж спины", switch="on", llm=llm)

    assert len(llm.address_calls) == 1, "по термину правила ничего не нашли — второго вопроса к LLM нет"
    assert sorted(out["addresses"]) == sorted(_BRANCHES), out


def test_llm_down_is_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    for question, _family, _places in _MULTI:
        on = _turn(mini_prod_world, monkeypatch, question, switch="on", llm=_LLM())
        off = _turn(mini_prod_world, monkeypatch, question, switch="off", llm=_LLM())
        assert (on["text"], on["addresses"]) == (off["text"], off["addresses"]), question


def test_task_asks_for_round_two_terms():
    from messengers_router import address_service
    from messengers_router.choice import Option

    options = [Option(id="0", text=_MG)]
    schema = choice.answer_schema(address_service.ADDRESS_SERVICE_TASK, options)
    system, _user = choice.render(address_service.ADDRESS_SERVICE_TASK, "где сделать магнитолечение", options)

    assert schema["properties"]["search_terms"]["maxItems"] == 3 and "search_terms" in schema["required"]
    assert '"search_terms"' in system and "без названия услуги" in system
