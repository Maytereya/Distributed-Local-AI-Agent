"""«Слово → роль» по каталогу МИС, режим `on` (ход (г), этап 3; STEP-2026-10-21-G-ROLE-PICK-ON).

Инвариант класса (решения владельца 20.10 (г) 1–6, 08.10 «похожий, но другой врач — никогда автоматически»): на метках врача
и записи роль хода решает LLM по каталогу ролей из среза МИС — специальность правил только кандидат: «окулист» →
офтальмолог, «лор» → оториноларинголог, «сердечно-сосудистый хирург» — не хирург, «мануальный терапевт» — не терапевт; роли
нет у врачей региона — честное «такого специалиста нет» + оффер оператора (`role_absent`), а не общий «нет данных»; отказ
стоп-листа на роль, которая в каталоге есть, снимается; «детский X» — оператору (`role_qualifier`); цена приёма, сбой LLM и
пустой каталог — как было. Под `off` и `shadow` решение хода не меняется. LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import choice, llm_runtime, role_pick
from messengers_router.mess_types import Evidence, RouteDecision
from messengers_router.response_builder import build_unsupported_catalog_response
from messengers_router.services._role_catalog import build_role_catalog
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_role_catalog_class import _doctor, _here
from tests.test_role_pick_shadow_class import _CATALOG, _MARK, _down, _picks

_SERVICES = object()  # каталог подменяется — сервисный слой не читается


def _apply(monkeypatch, text: str, answer, *, label: str = "APPOINTMENT", rules: str = "", flags=(), mode: str = "on", catalog=_CATALOG):
    from messengers_router.runtime_config import config as cfg

    fake = _LLM({_MARK: answer})
    decision = RouteDecision(label=label, entities={"specialty": rules} if rules else {}, flags=set(flags))

    async def cat(_services):
        return catalog

    with monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_ROLES_CATALOG_LLM", mode, raising=False)
        patch.setattr(role_pick, "catalog", cat)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
        choice._CACHE.clear()
        out = asyncio.run(role_pick.apply(decision, text, _SERVICES))
    return decision, out, fake.calls


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_off_and_shadow_do_not_touch_the_decision(monkeypatch, mode):
    before, after, calls = _apply(monkeypatch, "нужен окулист", _picks("офтальмолог"), mode=mode)
    assert after is before and not calls


@pytest.mark.parametrize(
    ("text", "rules", "role", "expected"),
    [
        ("нужен окулист", "", "офтальмолог", "офтальмолог"),
        ("запишите к сердечно-сосудистому хирургу", "хирург", "сердечно-сосудистый хирург", "сердечно-сосудистый хирург"),
        ("к мануальному терапевту", "терапевт", "мануальный терапевт", "мануальный терапевт"),
    ],
)
def test_llm_role_replaces_rules_specialty(monkeypatch, text, rules, role, expected):
    _before, after, _calls = _apply(monkeypatch, text, _picks(role), rules=rules)
    assert after.entities.get("specialty") == expected
    assert after.label == "APPOINTMENT"


def test_same_role_as_rules_changes_nothing(monkeypatch):
    before, after, _calls = _apply(monkeypatch, "запишите к неврологу", _picks("невролог"), rules="невролог")
    assert after is before


def test_stop_list_refusal_is_lifted_for_a_role_in_catalog(monkeypatch):
    flags = {"unsupported_catalog", "unsupported_specialist"}
    _before, after, _calls = _apply(monkeypatch, "есть ли окулист", _picks("офтальмолог"), label="DOCTOR_INFO", flags=flags)
    assert not (set(after.flags) & flags) and after.entities.get("specialty") == "офтальмолог"


@pytest.mark.parametrize("text", ["есть ли у вас трихолог", "нужен сомнолог", "нужен психиатр"])
def test_absent_role_is_honest_no_with_operator_offer(monkeypatch, text):
    _before, after, _calls = _apply(monkeypatch, text, _picks(None, "absent"), label="DOCTOR_INFO")
    assert "role_absent" in after.flags
    reply = build_unsupported_catalog_response(Evidence(items={"unsupported_catalog": {"kind": "role_absent"}}))
    assert reply is not None and "такого специалиста" in reply.text.lower()


def test_child_qualifier_goes_to_operator(monkeypatch):
    _before, after, _calls = _apply(monkeypatch, "запишите к детскому эндокринологу", _picks(None, "qualifier"), rules="эндокринолог")
    assert "role_qualifier" in after.flags and after.entities.get("specialty") == "эндокринолог"


def test_price_of_reception_is_not_acted_on(monkeypatch):
    before, after, calls = _apply(monkeypatch, "сколько стоит приём кардиолога", _picks("кардиолог"), label="PRICE")
    assert after is before and not calls


@pytest.mark.parametrize("answer", [_down, _picks(None, "not_role"), _picks(None)])
def test_failure_procedure_and_none_keep_the_decision(monkeypatch, answer):
    before, after, _calls = _apply(monkeypatch, "запишите на узи щитовидки", answer)
    assert after is before


def test_empty_catalog_never_says_no(monkeypatch):
    before, after, _calls = _apply(
        monkeypatch, "нужен психиатр", _picks(None, "absent"), label="DOCTOR_INFO", catalog=build_role_catalog([], in_region=_here)
    )
    assert after is before


def test_reproductologist_is_found_by_llm_from_catalog(monkeypatch):
    catalog = build_role_catalog([_doctor((1, "Врач-терапевт")), _doctor((9, "Врач-репродуктолог"))], in_region=_here)
    _before, after, _calls = _apply(monkeypatch, "есть ли у вас врач по бесплодию", _picks("репродуктолог"), label="DOCTOR_INFO", catalog=catalog)
    assert after.entities.get("specialty") == "репродуктолог"
