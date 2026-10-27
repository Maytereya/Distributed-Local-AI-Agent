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
_REGIONS_CITY = "Энск"  # город подделки — у настоящих филиалов он записан и в адресе
_ADDRESS_LEAVES = sorted(b for b, a in button_menu.BUTTONS.items() if a.mode == "pass" and a.label == "ADDRESS")


def _answer(outcome: str, *picks: str, regions: list[dict[str, Any]] = _REGIONS) -> str:
    return json.dumps({"picks": [[r["addressForSite"] for r in regions].index(p) + 1 for p in picks], "outcome": outcome})


def _dialog(
    world_dir, monkeypatch, turns, *, switch: str, followup: str | None, regions: list[dict[str, Any]] = _REGIONS, named_street: str = "off"
) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg

    async def fake_regions(self):
        return [{"id": i, "name": f"Самара {r['addressForSite']}", "city": "Самара", "phone": f"+7 846 000 00 0{i}", **r} for i, r in enumerate(regions, 1)]

    answers: dict[str, Any] = {}
    if followup is not None:
        answers[_MARK] = followup
    session = f"addr-followup-{switch}-{named_street}-{len(regions)}-{followup}-{json.dumps(turns, ensure_ascii=False)}"

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append({"text": str(body.get("text") or ""), "state": dict(memory.get(session).last_entities or {}), "telemetry": world.take_telemetry() or {}})
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(Services, "_ensure_regions_loaded", fake_regions)
        patch.setattr(cfg, "MR_ADDRESS_FOLLOWUP_LLM", switch, raising=False)
        patch.setattr(cfg, "MR_ADDRESS_NAMED_STREET", named_street, raising=False)
        patch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def _listed(text: str, regions: list[dict[str, Any]] = _REGIONS) -> set[str]:
    return {r["addressForSite"] for r in regions if r["addressForSite"] in text}


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


def test_branch_not_named_in_the_reply_is_not_carded(mini_prod_world, monkeypatch):  # noqa: F811
    """Карточка 25.10: LLM выбирала филиалы, которых пациент не называл («рядом с метро …», «рядом с моим адресом», пустой выбор
    — три одних и тех же): выбор сужает правило «слово адреса филиала есть в реплике»; без названного филиала — общий путь."""

    turns = [_press("menu.address"), "есть ли филиал рядом с метро Московская?"]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=_answer("branch", "ул.Аминева, 29"))
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", followup=None)
    assert on[1]["text"] == off[1]["text"] and on[1]["telemetry"].get("address_followup_vetoed") == 1, on[1]["telemetry"]


def test_city_word_in_an_address_does_not_name_the_branch(mini_prod_world, monkeypatch):  # noqa: F811
    """Карточка 25.10: город в реплике называл филиал, у которого город записан в адресе, — город филиала (`city` его данных) и
    слова, общие для многих адресов, филиал не называют."""

    city = _REGIONS_CITY
    addresses = [f"г. {city}, ул. Победы, 126", "ул. Мичурина, 112", "ул.Аминева, 29"]
    stems = address_flow._common_words(addresses) | frozenset(address_flow._address_words(city))
    assert not address_flow.named_in(f"как работаете в праздники в {city}е?", addresses[0], common=stems)
    assert address_flow.named_in("на Победы 126 в субботу?", addresses[0], common=stems)


@pytest.mark.parametrize("button_id", _ADDRESS_LEAVES)
def test_when_to_bring_biomaterial_gets_lab_hours(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """BUG-2026-10-26-ADDRESS-FOLLOWUP-HOURS: «кал в 15:00?», «ХГЧ вечером?» после адресного листа получали «нет в данных» и
    лишний оффер оператора — ответ есть в данных: пункты забора (`analysis`) и их график; время сдачи, которое задаёт подготовка
    анализа, графиком не подменяется — отсылка к подготовке."""

    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), "я смогу принести кал на анализ в 15.00?"], switch="on", followup=_answer("hours"))
    text = on[1]["text"]
    assert text.startswith(address_flow.HOURS_HEAD) and text.endswith(address_flow.HOURS_NOTE), text
    assert _listed(text) == {r["addressForSite"] for r in _REGIONS if r["analysis"]}, text
    assert "будни 07:00–18:30" in text and not on[1]["state"].get("_operator_offer_pending"), on[1]
    assert on[1]["telemetry"].get("address_followup") == "hours"


# BUG-2026-10-26-ADDRESS-FOLLOWUP-NAMED-BRANCH-HOURS: на одной улице несколько филиалов — как «Победы» (126, 95, 83) в `/site/regions`.
_STREET = [
    {"addressForSite": "ул.Луговая, 12", "analysis": True, "ecg": True, "usi": False, **_HOURS},
    {"addressForSite": "ул.Луговая, 40", "analysis": True, "ecg": False, "usi": True, **_HOURS, "weekdaysTo": "20:00:00"},
    {"addressForSite": "ул.Луговая, 77", "analysis": True, "ecg": False, "usi": False, **_HOURS},
]
_STREET_REGIONS = _REGIONS + _STREET


@pytest.mark.parametrize("button_id", _ADDRESS_LEAVES)
@pytest.mark.parametrize("outcome", ["hours", "branch"])
def test_time_at_a_named_street_gets_the_cards_of_its_branches(button_id, outcome, mini_prod_world, monkeypatch):  # noqa: F811
    """Смоук деплоя 18 на проде: «а на Победы до скольки?» → часы всех пунктов забора. LLM выбрала три филиала на Победы, а
    правило «слово есть в адресах больше чем двух филиалов — филиал не назван» их вычеркнуло. Инвариант: вопрос о времени работы
    названного филиала (или улицы, на которой их несколько) получает карточки именно их, с графиком из данных, а не список всех;
    исход LLM — `hours` (так она отвечает на живой пробе) или `branch` — не важен. off — как было: выбор вычеркнут."""

    turns = [_press(button_id), "а на Луговой до скольки?"]
    street = [r["addressForSite"] for r in _STREET]
    answer = _answer(outcome, *street, regions=_STREET_REGIONS)
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=answer, regions=_STREET_REGIONS, named_street="on")
    text = on[1]["text"]
    assert _listed(text, _STREET_REGIONS) == set(street) and not text.startswith(address_flow.HOURS_HEAD), text
    assert text.count("График: будни 07:00–18:30") == 2 and "График: будни 07:00–20:00" in text, text
    assert "address_followup_vetoed" not in on[1]["telemetry"] and on[1]["telemetry"].get("address_followup_picks") == 3
    assert not on[1]["state"].get("_operator_offer_pending"), on[1]
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=answer, regions=_STREET_REGIONS, named_street="off")
    assert off[1]["telemetry"].get("address_followup_vetoed") == 3, off[1]["telemetry"]


@pytest.mark.parametrize("button_id", _ADDRESS_LEAVES)
def test_house_number_on_a_shared_street_gets_one_card(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Улица названа с номером дома — LLM выбирает один филиал, и карточка — только его (основа улицы общая для трёх адресов)."""

    turns = [_press(button_id), "на Луговой 40 во сколько открываетесь?"]
    answer = _answer("hours", "ул.Луговая, 40", regions=_STREET_REGIONS)
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=answer, regions=_STREET_REGIONS, named_street="on")
    assert _listed(on[1]["text"], _STREET_REGIONS) == {"ул.Луговая, 40"} and "Здесь можно сдать анализы, сделать УЗИ." in on[1]["text"], on[1]["text"]


def test_word_of_more_branches_than_cards_does_not_name_them(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: слово, общее для большего числа филиалов, чем карточек в ответе (`max_picks`), филиалов не называет — все не
    покажем, и одна-три карточки были бы чужим графиком; ответ — общий путь (здесь — часы пунктов забора), как без выключателя."""

    regions = _STREET_REGIONS + [{"addressForSite": "ул.Луговая, 101", "analysis": True, "ecg": False, "usi": False, **_HOURS}]
    assert len([r for r in regions if "Луговая" in r["addressForSite"]]) > address_flow.FOLLOWUP_TASK.max_picks
    turns = [_press("menu.address"), "а на Луговой до скольки?"]
    answer = _answer("hours", *[r["addressForSite"] for r in _STREET], regions=regions)
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=answer, regions=regions, named_street="on")
    assert on[1]["text"].startswith(address_flow.HOURS_HEAD) and on[1]["telemetry"].get("address_followup_vetoed") == 3, on[1]


@pytest.mark.parametrize("button_id", _ADDRESS_LEAVES)
def test_when_to_bring_without_a_named_branch_keeps_lab_hours(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный (BUG-2026-10-26-ADDRESS-FOLLOWUP-HOURS): «когда принести» без названного филиала под выключателем — тот же
    ответ, что без него: пункты забора с графиком и отсылка к подготовке."""

    turns = [_press(button_id), "когда можно принести кал на анализ?"]
    answers = [
        _dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=_answer("hours", regions=_STREET_REGIONS), regions=_STREET_REGIONS, named_street=s)
        for s in ("on", "off")
    ]
    assert answers[0][1]["text"] == answers[1][1]["text"], answers[0][1]["text"]
    text = answers[0][1]["text"]
    assert text.startswith(address_flow.HOURS_HEAD) and text.endswith(address_flow.HOURS_NOTE), text
    assert _listed(text, _STREET_REGIONS) == {r["addressForSite"] for r in _STREET_REGIONS if r["analysis"]}, text


def test_city_word_does_not_name_the_branch_under_named_street():
    """Встречный: порог «общего слова» поднят до числа карточек, но город филиала (`city` его данных) по-прежнему не называет
    филиал, у которого город записан в адресе (карточка 25.10)."""

    city = _REGIONS_CITY
    addresses = [f"г. {city}, ул. Победы, 126", "ул. Победы, 95", "ул. Победы, 83", "ул.Аминева, 29"]
    stems = address_flow._common_words(addresses, more_than=address_flow.FOLLOWUP_TASK.max_picks) | frozenset(address_flow._address_words(city))
    assert not address_flow.named_in(f"как работаете в праздники в {city}е?", addresses[0], common=stems)
    assert all(address_flow.named_in("а на Победы до скольки?", a, common=stems) for a in addresses[:3])
    assert not address_flow.named_in("а на Победы до скольки?", addresses[3], common=stems)
