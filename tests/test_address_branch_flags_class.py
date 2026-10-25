"""Список «где сдать» — только там, где по данным филиалов это делают (BUG-2026-10-25-PASS-BUTTON-FOLLOWUP-BRANCH-TEMPLATE, часть 1,
выключатель `address_branch_flags`).

Замер `buttons_v1` (деплой 18): во всех 80 ответах на «Где и до скольки принимают» и в 26 ответах «Анализы выполняются без записи,
в порядке живой очереди» в списке был филиал без забора анализов (Ново-Садовая, 106 к.82: `analysis=false`).

Инвариант класса: под on в ответе, где бот говорит, что анализы (ЭКГ, УЗИ) делают без записи, и на нажатии «Где и до скольки
принимают» (раздел «Сдать анализы») с ближайшей репликой после него, каждый филиал списка — с нужным флагом данных филиалов;
общий вопрос об адресах («Адреса и часы работы») — все филиалы; off — как было. Филиалы — подделка `/site/regions` с флагами,
LLM недоступна (путь правил, как в мире гейта).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import address_flow, choice, llm_runtime
from messengers_router.services import Services
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

# Флаги филиалов как в `/site/regions`: заборные точки, приём врачей без забора, УЗИ без забора.
_REGIONS = [
    *({"addressForSite": f"ул. Лабораторная, {n}", "analysis": True, "ecg": n % 2 == 0, "usi": False, "doctorService": False} for n in range(1, 7)),
    {"addressForSite": "ул. Ново-Садовая, 106, корпус 82", "analysis": False, "ecg": False, "usi": False, "doctorService": True},
    {"addressForSite": "ул. Врачебная, 5", "analysis": False, "ecg": True, "usi": True, "doctorService": True},
]
_NO_LAB = {r["addressForSite"] for r in _REGIONS if not r["analysis"]}
_NO_ECG = {r["addressForSite"] for r in _REGIONS if not r["ecg"]}


def _dialog(world_dir, monkeypatch, turns, *, switch: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg

    async def regions(self):
        return [{"id": i, "name": f"Самара {r['addressForSite']}", "city": "Самара", "phone": "+7", **r} for i, r in enumerate(_REGIONS, 1)]

    session = f"branch-flags-{switch}-{json.dumps(turns, ensure_ascii=False)}"

    async def run() -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append({"text": str(body.get("text") or ""), "state": dict(memory.get(session).last_entities or {})})
        return out

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(Services, "_ensure_regions_loaded", regions)
        patch.setattr(cfg, "MR_ADDRESS_BRANCH_FLAGS", switch, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM({}))
        choice._CACHE.clear()
        return asyncio.run(run())


def _listed(text: str) -> set[str]:
    return {r["addressForSite"] for r in _REGIONS if r["addressForSite"] in text}


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


@pytest.mark.parametrize("question", ["хочу сдать кровь", "где сдать анализы", "можно сдать анализы без записи?"])
def test_walkin_analysis_answer_lists_only_lab_branches(question, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [question], switch="off")[0]["text"]
    on = _dialog(mini_prod_world, monkeypatch, [question], switch="on")[0]["text"]
    if "без записи" not in off:
        pytest.skip(f"«{question}» и без выключателя отвечают не фразой «без записи»")
    assert _listed(off) & _NO_LAB, "off — как было: в списке филиал без забора"
    assert _listed(on) and not _listed(on) & _NO_LAB, on


def test_walkin_ecg_answer_lists_only_ecg_branches(mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, ["где сделать ЭКГ без записи?"], switch="on")[0]["text"]
    if "без записи" not in on:
        pytest.skip("ответ не фразой «без записи»")
    assert _listed(on) and not _listed(on) & _NO_ECG, on


@pytest.mark.parametrize("follow", ["а в субботу работаете?", "а где поближе к центру?"])
def test_tests_where_button_and_next_reply_list_lab_branches(follow, mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.where"), follow], switch="on")
    off = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.where"), follow], switch="off")
    assert _listed(off[0]["text"]) & _NO_LAB, "off — как было"
    for turn in on:
        if _listed(turn["text"]):
            assert not _listed(turn["text"]) & _NO_LAB, turn["text"]
    assert _listed(on[0]["text"]), on[0]["text"]
    assert on[0]["state"].get(address_flow.ADDRESS_BUTTON_KEY, {}).get("id") == "menu.tests.where"


def test_mark_lives_one_reply(mini_prod_world, monkeypatch):  # noqa: F811
    """Метка кнопки — на нажатии и одной реплике после; вторая реплика и другое нажатие её снимают."""

    turns = [_press("menu.tests.where"), "а в субботу?", "хочу сдать кровь"]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on")
    marks = [t["state"].get(address_flow.ADDRESS_BUTTON_KEY) for t in on]
    assert marks[0] == {"id": "menu.tests.where", "age": 0} and marks[1] is None and marks[2] is None, marks
    again = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.where"), _press("menu.price.test")], switch="on")
    assert address_flow.ADDRESS_BUTTON_KEY not in again[1]["state"]


def test_general_address_button_keeps_every_branch(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: «Адреса и часы работы» — общий вопрос об адресах, фильтра нет: все филиалы, как без выключателя."""

    on = _dialog(mini_prod_world, monkeypatch, [_press("menu.address")], switch="on")[0]["text"]
    off = _dialog(mini_prod_world, monkeypatch, [_press("menu.address")], switch="off")[0]["text"]
    assert on == off and _listed(on) & _NO_LAB


def test_off_keeps_state_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.where"), "а в субботу?"], switch="off")
    assert all(address_flow.ADDRESS_BUTTON_KEY not in t["state"] for t in off)


@pytest.mark.parametrize(
    ("hint", "wanted"),
    [("анализы", {"analysis"}), ("ЭКГ", {"ecg"}), ("анализы и ЭКГ", {"analysis", "ecg"}), ("УЗИ", {"usi"}), ("", set())],
)
def test_needed_flags_follow_the_answer(hint, wanted):
    """Что без записи назвал ответ — те флаги филиала и нужны; фильтр без флагов в данных и пустой фильтр — как было."""

    assert address_flow.needed_flags(hint) == frozenset(wanted)
    rows = [{"address": r["addressForSite"], **{f: r[f] for f in ("analysis", "ecg", "usi")}} for r in _REGIONS]
    kept = address_flow.filter_branches(rows, frozenset(wanted))
    if wanted:
        assert kept and all(all(row[f] for f in wanted) for row in kept), kept
    else:
        assert kept is None
    assert address_flow.filter_branches([{"address": "x"}], frozenset({"analysis"})) is None  # флагов в данных нет
