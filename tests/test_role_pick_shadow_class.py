"""«Слово → роль» по каталогу МИС: решение `role_pick.decide` (ход (г), L-17, этап 2; STEP-2026-10-21-G-ROLE-PICK-SHADOW).

Инвариант класса (решения владельца 20.10 (г) 1, 2, 6): роль решает LLM по каталогу ролей из среза; точный слой и
специальность правил — только кандидаты (первыми в списке вариантов), «похожий, но другой врач» не подставляется
кодом; где точный слой нашёл одну роль, поля хода пишут согласие LLM с ним (`role_llm_agree`); уточнение пациента
(«детский») — исход `qualifier`, процедура — `not_role`; сбой LLM — вид сбоя, без согласия. LLM — подделка настоящего типа.
Тень (`roles_catalog_llm = shadow`) и путь `off` сняты 29.10 (STEP-2026-10-29-SWITCH-REMOVAL); тесты тени удалены.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from messengers_router import choice, llm_runtime, role_pick
from messengers_router.services._role_catalog import build_role_catalog
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_role_catalog_class import _doctor, _here

_MARK = "о каком враче (специальности приёма) пишет пациент"
_CATALOG = build_role_catalog(
    [
        _doctor((1, "Врач-терапевт")),
        _doctor((2, "Врач-хирург")),
        _doctor((3, "Врач-сердечно-сосудистый хирург")),
        _doctor((4, "Врач-мануальный терапевт")),
        _doctor((5, "Врач-офтальмолог")),
        _doctor((6, "Врач гинеколог-маммолог")),
        _doctor((7, "Врач-невролог")),
    ],
    in_region=_here,
)


def _picks(role: str | None, outcome: str | None = None):
    """Подделка LLM: выбирает вариант с названием `role` (по номеру из промпта) и/или исход."""

    def answer(prompt: str) -> str:
        numbers = dict((name.strip(), int(n)) for n, name in re.findall(r"^(\d+)\. (.+)$", prompt, flags=re.M))
        out: dict = {"picks": [numbers[role]] if role else []}
        if outcome:
            out["outcome"] = outcome
        return json.dumps(out)

    return answer


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


def _decide(monkeypatch, text: str, answer, *, label: str = "APPOINTMENT", rules: str = "", catalog=_CATALOG) -> dict:
    fake = _LLM({_MARK: answer})
    with monkeypatch.context() as patch:
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
        choice._CACHE.clear()
        return asyncio.run(role_pick.decide(label, text, rules, catalog)) or {}


def test_exact_layer_hit_is_checked_by_llm(monkeypatch):
    f = _decide(monkeypatch, "запишите к сердечно-сосудистому хирургу", _picks("сердечно-сосудистый хирург"), rules="хирург")
    assert (f["role_layer"], f["role_exact"], f["role_llm"]) == ("exact", "сердечно-сосудистый хирург", "сердечно-сосудистый хирург")
    assert f["role_llm_agree"] is True
    assert f["role_rules_agree"] is False  # правила свели к «хирургу» — чужой врач (SURGEON-PRICE-FOR-SPECIALIST)


def test_colloquial_rules_name_is_not_counted_as_disagreement(monkeypatch):
    f = _decide(monkeypatch, "нужен окулист", _picks("офтальмолог"), rules="окулист")
    assert "role_rules_agree" not in f and f["role_rules_unmapped"] is True


def test_exact_layer_disagreement_is_counted(monkeypatch):
    f = _decide(monkeypatch, "к мануальному терапевту", _picks("терапевт"))
    assert f["role_exact"] == "мануальный терапевт" and f["role_llm_agree"] is False


def test_miss_of_exact_layer_goes_to_llm(monkeypatch):
    f = _decide(monkeypatch, "нужен окулист", _picks("офтальмолог"))
    assert (f["role_layer"], f["role_exact"], f["role_llm"]) == ("miss", None, "офтальмолог")
    assert "role_llm_agree" not in f


def test_partial_role_is_first_candidate_not_answer(monkeypatch):
    seen: list[str] = []

    def answer(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"picks": []})

    f = _decide(monkeypatch, "есть ли маммолог", answer, label="DOCTOR_INFO")
    assert f["role_layer"] == "partial" and f["role_exact"] is None and f["role_llm"] == "none"
    assert re.search(r"^1\. гинеколог-маммолог$", seen[0], flags=re.M), "часть составной роли — первым кандидатом"


@pytest.mark.parametrize("outcome", ["qualifier", "not_role"])
def test_outcomes_are_recorded(monkeypatch, outcome):
    f = _decide(monkeypatch, "к детскому неврологу", _picks(None, outcome))
    assert f["role_llm"] == outcome


def test_llm_failure_is_not_agreement(monkeypatch):
    f = _decide(monkeypatch, "к неврологу", _down)
    assert f["role_llm"] == "unavailable" and "role_llm_agree" not in f


def test_price_of_reception_is_a_doctor_turn(monkeypatch):
    f = _decide(monkeypatch, "сколько стоит приём сердечно-сосудистого хирурга", _picks("сердечно-сосудистый хирург"), label="PRICE")
    assert f["role_layer"] == "exact" and f["role_llm_agree"] is True


def test_not_a_doctor_turn_is_not_decided(monkeypatch):
    assert _decide(monkeypatch, "сколько стоит ферритин", _down, label="PRICE") == {}


def test_empty_catalog_is_not_decided(monkeypatch):
    assert _decide(monkeypatch, "к неврологу", _down, catalog=build_role_catalog([], in_region=_here)) == {"role_layer": "no_catalog"}
