"""Реплика после адресного листа — ответ по данным филиалов, а не общий список (BUG-2026-10-25-ADDRESS-FOLLOWUP-TEMPLATE, часть 2
класса BUG-2026-10-25-PASS-BUTTON-FOLLOWUP-BRANCH-TEMPLATE; выключатель `address_followup_llm`).

Замер `buttons_v1` (деплой 18): на вопрос сразу после «Адреса и часы работы» / «Где и до скольки принимают» («По адресу аминева 29
можно сдать анализы?», «Режим работы в праздники», «Можно ли сдать кровь ребенку 10 месяцев?») бот повторял общий список всех
филиалов — 67 провалов на этих двух листах.

Инвариант класса: под on реплика после любого адресного листа, в которой LLM узнала названный филиал, получает карточку именно
его (адрес, телефон, график, что там делают — из данных), а не список остальных; «нет в данных» — честный ответ и оффер оператора;
«выходные» — филиалы, работающие в выходные (на «Где и до скольки принимают» — заборные). Встречные: общий вопрос, другая тема,
сбой LLM, вторая реплика после кнопки, реплика после другой кнопки, off — как было. Филиалы — подделка `/site/regions`, LLM —
подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import address_flow, button_menu, choice, llm_runtime
from messengers_router.services import Services
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_MARK = "бот показал список филиалов, а пациент спросил ещё"
_HOURS = {"weekdaysFrom": "07:00:00", "weekdaysTo": "18:30:00", "saturdayFrom": "08:00:00", "saturdayTo": "14:00:00", "sundayFrom": "08:00:00", "sundayTo": "14:00:00"}
_REGIONS = [
    {"addressForSite": "ул.Аминева, 29", "analysis": True, "ecg": True, "usi": False, **_HOURS},
    {"addressForSite": "ул.Пензенская, 71", "analysis": True, "ecg": False, "usi": False, **_HOURS},
    {"addressForSite": "пр.Ленина, 5", "analysis": True, "ecg": True, "usi": True, **_HOURS, "weekdaysTo": "20:00:00"},
    {"addressForSite": "ул.Ново-Садовая, 106, корпус 82", "analysis": False, "ecg": False, "usi": False, **_HOURS},
]
_ADDRESS_LEAVES = sorted(b for b, a in button_menu.BUTTONS.items() if a.mode == "pass" and a.label == "ADDRESS")


def _answer(outcome: str, *picks: str) -> str:
    return json.dumps({"picks": [[r["addressForSite"] for r in _REGIONS].index(p) + 1 for p in picks], "outcome": outcome})


def _dialog(world_dir, monkeypatch, turns, *, switch: str, followup: str | None) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg

    async def regions(self):
        return [{"id": i, "name": f"Самара {r['addressForSite']}", "city": "Самара", "phone": f"+7 846 000 00 0{i}", **r} for i, r in enumerate(_REGIONS, 1)]

    answers: dict[str, Any] = {}
    if followup is not None:
        answers[_MARK] = followup
    session = f"addr-followup-{switch}-{followup}-{json.dumps(turns, ensure_ascii=False)}"

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append({"text": str(body.get("text") or ""), "state": dict(memory.get(session).last_entities or {}), "telemetry": world.take_telemetry() or {}})
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(Services, "_ensure_regions_loaded", regions)
        patch.setattr(cfg, "MR_ADDRESS_FOLLOWUP_LLM", switch, raising=False)
        patch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def _listed(text: str) -> set[str]:
    return {r["addressForSite"] for r in _REGIONS if r["addressForSite"] in text}


@pytest.mark.parametrize("button_id", _ADDRESS_LEAVES)
def test_named_branch_gets_its_own_card(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press(button_id), "по адресу аминева 29 можно сдать анализы?"]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=_answer("branch", "ул.Аминева, 29"))
    text = on[1]["text"]
    assert _listed(text) == {"ул.Аминева, 29"}, text
    assert "Телефон: +7 846 000 00 01" in text and "будни 07:00–18:30" in text, text
    assert "Здесь можно сдать анализы, сделать ЭКГ." in text and "Здесь нельзя сделать УЗИ." in text, text
    assert on[1]["telemetry"].get("address_followup") == "branch" and on[1]["telemetry"].get("nlu_llm") is not True


@pytest.mark.parametrize("button_id", _ADDRESS_LEAVES)
def test_question_without_data_is_honest_and_offers_operator(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), "как работаете 12 июня?"], switch="on", followup=_answer("no_data"))
    assert on[1]["text"] == address_flow.NO_DATA_TEXT and on[1]["state"].get("_operator_offer_pending"), on[1]
    named = _dialog(mini_prod_world, monkeypatch, [_press(button_id), "на Ленина 5 ребёнку можно?"], switch="on", followup=_answer("no_data", "пр.Ленина, 5"))
    assert _listed(named[1]["text"]) == {"пр.Ленина, 5"} and named[1]["text"].endswith(address_flow.NO_DATA_TEXT)


def test_weekend_lists_open_branches_and_tests_scope(mini_prod_world, monkeypatch):  # noqa: F811
    general = _dialog(mini_prod_world, monkeypatch, [_press("menu.address"), "в субботу работаете?"], switch="on", followup=_answer("weekend"))
    assert general[1]["text"].startswith("В выходные работают:") and _listed(general[1]["text"]) == {r["addressForSite"] for r in _REGIONS}
    tests = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.where"), "в субботу работаете?"], switch="on", followup=_answer("weekend"))
    assert "ул.Ново-Садовая, 106, корпус 82" not in tests[1]["text"] and _listed(tests[1]["text"]), tests[1]["text"]


@pytest.mark.parametrize("followup", [_answer("list"), _answer("other_topic"), None, "не JSON"])
def test_general_question_other_topic_and_failure_keep_general_path(followup, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press("menu.address"), "где вы находитесь?"]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=followup)
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", followup=followup)
    assert on[1]["text"] == off[1]["text"]
    assert "address_followup" not in on[1]["telemetry"] or on[1]["telemetry"]["address_followup"] in {"list", "other_topic", "none"}


def test_only_the_reply_right_after_the_address_button(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречные: вторая реплика после кнопки и реплика после другой кнопки — общий путь."""

    late = _dialog(mini_prod_world, monkeypatch, [_press("menu.address"), "спасибо", "по адресу аминева 29?"], switch="on", followup=_answer("branch", "ул.Аминева, 29"))
    assert "address_followup" not in late[2]["telemetry"], late[2]["telemetry"]
    other = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.promo"), "по адресу аминева 29?"], switch="on", followup=_answer("branch", "ул.Аминева, 29"))
    assert "address_followup" not in other[1]["telemetry"], other[1]["telemetry"]


def test_off_keeps_the_general_answer(mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press("menu.address"), "по адресу аминева 29 можно сдать анализы?"]
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", followup=_answer("branch", "ул.Аминева, 29"))
    assert "address_followup" not in off[1]["telemetry"] and address_flow.ADDRESS_BUTTON_KEY not in off[0]["state"]


def test_branch_not_named_in_the_reply_is_not_carded(mini_prod_world, monkeypatch):  # noqa: F811
    """Карточка 25.10: LLM выбирала филиалы, которых пациент не называл («рядом с метро …», «рядом с моим адресом», пустой выбор
    — три одних и тех же): выбор сужает правило «слово адреса филиала есть в реплике»; без названного филиала — общий путь."""

    turns = [_press("menu.address"), "есть ли филиал рядом с метро Московская?"]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=_answer("branch", "ул.Аминева, 29"))
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", followup=None)
    assert on[1]["text"] == off[1]["text"] and on[1]["telemetry"].get("address_followup_vetoed") == 1, on[1]["telemetry"]
