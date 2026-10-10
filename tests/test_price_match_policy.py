"""Политика силы совпадения для цены (L-13, решение владельца 07.10 — вариант B) — через эндпоинт.

Выключатель `[MESSENGER_ROUTER] price_match_policy = off | shadow | on` (по умолчанию shadow).

Инварианты класса:
- сильное совпадение по данным (дословное название, синоним МИС, код) — только эти строки, без LLM;
- слабое совпадение, а LLM не ответила внятно: цены слабой строки в ответе нет — до 5 названий,
  покрывающих все слова пациента, без цен; таких нет — честный перевод на оператора;
- LLM не нашла услугу и во втором круге — «не нашёл», без чужой цены;
- политика заменяет только слабое предложение правил: правила ничего не предложили — их ответ;
- off и shadow не меняют ни одного ответа; on меняет только объявленные диалоги (`POLICY_PLANNED`) —
  эталон пути цены при недоступной LLM, ответы под on — `tests/golden/price_turns_policy_on.json`
  (перезапись: `GOLDEN_UPDATE=1`, дифф — в коммит с разбором);
- тень пишет решение в телеметрию хода (`price_policy`), ответ прежний.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.test_price_turns_golden import DIALOGS, FIXTURE, _golden, run_dialog, synonym_fork  # noqa: F401 — фикстура для synonym_fork-диалогов

GOLDEN_POLICY = Path(__file__).resolve().parent / "golden" / "price_turns_policy_on.json"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
_PRICE_PROMPT_MARK = "ответить пациенту на вопрос о цене"

# Диалоги, которые под on (LLM недоступна) отвечают иначе, — с причиной; разбор диффа 11.10.
POLICY_PLANNED: dict[str, str] = {
    "one_gastroscopy": "слабое совпадение «Гастрин» без LLM — перевод вместо чужой цены",
    "family_glucose": "дословное название «Глюкоза» — только эта строка, без мочи и синовиальной жидкости",
    "name_with_commas": "дословное название с уточнением в скобках — ровно эта строка",
    "name_with_plus": "вместо мусорного списка с ценами — одно название целиком, без цены",
    "one_blood_draw": "вместо списка чужих анализов с ценами — названия без цен",
    "one_ttg": "вариант B: ТТГ — слабое совпадение, названия без цен",
    "list_over_stale_context": "вариант B на первом ходе («ттг»); корзина — L-15",
    "one_oak_words": "вариант B: «общий анализ крови» не дословное название — названия без цен",
    "family_vitamin_d_all": "вариант B: семейство витамина D — названия без цен",
    "one_neurologist": "вариант B: пять вариантов приёма невролога — названия без цен",
    "name_with_and": "вариант B: одна похожая услуга — название без цены",
    "one_oak_abbr": "алиас кода — не данные: в гейте нет синонимов МИС, названия «ОАК (…)» без цен",
    "synonym_fork_oak": "синоним МИС «оак» — ровно его услуги из справочника клиники, без хвоста семейства",
}


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


def _policy(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_MATCH_POLICY", mode, raising=False)


def _load() -> dict:
    return json.loads(GOLDEN_POLICY.read_text(encoding="utf-8")) if GOLDEN_POLICY.exists() else {}


def _store(case: str, record) -> None:
    data = _load()
    if record is None:
        data.pop(case, None)
    else:
        data[case] = record
    GOLDEN_POLICY.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_price_path_without_llm_changes_only_planned_dialogs(case, mode, mini_prod_world, request, monkeypatch):
    _policy(monkeypatch, mode)
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog(DIALOGS[case], f"policy-{case}-{mode}")
    base = _golden()[case]
    if UPDATE and mode == "on":
        _store(case, None if record == base else record)
        return
    expected = _load().get(case, base) if mode == "on" else base
    assert record == expected


def test_policy_changes_only_declared_dialogs():
    assert set(_load()) == set(POLICY_PLANNED)


class _PriceLLM:
    """LLM отвечает только на выбор строки прайса (форма модуля или прежняя); остальное — «недоступна»."""

    def __init__(self, picks_by_round: list[list[int]], reason: str = "other_service", terms: list[str] | None = None) -> None:
        self.rounds, self.reason, self.terms, self.price_calls = list(picks_by_round), reason, terms or [], 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _PRICE_PROMPT_MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.price_calls += 1
        picks = self.rounds.pop(0) if self.rounds else []
        terms = [] if picks else self.terms
        if system is None:
            answer = {"match": picks, "search_terms": terms}
        else:
            answer = {"picks": picks, "reason": "none" if picks else self.reason, "search_terms": terms}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _turn(question: str, session: str, mini_prod_world) -> dict:
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        [turn] = run_dialog([question], session)
        turn["telemetry"] = world.take_telemetry()
    return turn


def test_weak_match_without_llm_never_shows_its_price(mini_prod_world, monkeypatch):
    _policy(monkeypatch, "on")
    turn = _turn("сколько стоит гастроскопия", "policy-gastro", mini_prod_world)
    assert turn["handoff"] is True
    assert "руб" not in turn["text"] and "Гастрин" not in turn["text"]


def test_weak_match_without_llm_lists_names_covering_all_words_without_prices(mini_prod_world, monkeypatch):
    _policy(monkeypatch, "on")
    turn = _turn("сколько стоит прием невролога", "policy-neuro", mini_prod_world)
    names = [line[2:] for line in turn["text"].splitlines() if line.startswith("— ")]
    assert turn["handoff"] is False and "руб" not in turn["text"]
    assert 1 < len(names) <= 5 and all("невролог" in name.lower() for name in names)


@pytest.mark.parametrize("choose_mode", ["off", "on"])
def test_model_found_nothing_twice_is_not_found_without_a_foreign_price(choose_mode, mini_prod_world, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    _policy(monkeypatch, "on")
    monkeypatch.setattr(cfg, "MR_CHOOSE_PRICE", choose_mode, raising=False)
    llm = _PriceLLM([[], []], terms=["ФГДС"])
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    if choose_mode == "off":
        from messengers_router.services import _price_select_llm as PS

        async def old_wire(prompt, **_kwargs):
            return json.dumps({"match": [], "search_terms": ["ФГДС"]})

        monkeypatch.setattr(PS, "generate_text", old_wire)
    turn = _turn("сколько стоит гастроскопия", f"policy-nf-{choose_mode}", mini_prod_world)
    assert turn["text"].startswith("Не нашёл актуальную стоимость для «гастроскопия»")
    assert "750" not in turn["text"]


def test_strong_match_is_shown_without_llm(mini_prod_world, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    _policy(monkeypatch, "on")
    monkeypatch.setattr(cfg, "MR_CHOOSE_PRICE", "on", raising=False)
    llm = _PriceLLM([[1]])
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    turn = _turn("сколько стоит глюкоза", "policy-strong", mini_prod_world)
    assert "Глюкоза — 190" in turn["text"] and "в моче" not in turn["text"]
    assert llm.price_calls == 0


def test_shadow_writes_the_decision_and_keeps_the_answer(mini_prod_world, monkeypatch):
    _policy(monkeypatch, "shadow")
    turn = _turn("сколько стоит гастроскопия", "policy-shadow", mini_prod_world)
    assert turn["text"] == _golden()["one_gastroscopy"][0]["text"]
    assert turn["telemetry"]["price_policy"] == "handoff"


def test_too_general_or_empty_rules_proposal_keeps_the_rules_answer(monkeypatch, tmp_path):
    import asyncio

    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _price_select_llm as PS

    _policy(monkeypatch, "on")
    monkeypatch.setattr(cfg, "MR_CHOOSE_PRICE", "on", raising=False)
    monkeypatch.setattr(PS, "_journal_path", lambda: tmp_path / "journal.jsonl")
    gastrin = {"serviceName": "Гастрин", "serviceHomecode": "1001", "cost": 750}
    rows = [gastrin, {"serviceName": "Гастрин-17 стимулированный", "serviceHomecode": "1002", "cost": 1950}]

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceLLM([[]], reason="too_general"))
    assert asyncio.run(PS.select_price_rows("сколько стоит гастроскопия", rows, candidates=[gastrin])) is None

    # Правила ничего не предложили, LLM недоступна — чужой цены не было, их ответ остаётся.
    class _Down:
        calls = 0

        async def generate(self, **_kwargs):
            _Down.calls += 1
            raise ConnectionError("ollama down")

    from messengers_router import choice

    choice._CACHE.clear()  # тот же промпт — иначе ответ первого вызова пришёл бы из кэша выбора
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Down())
    assert asyncio.run(PS.select_price_rows("сколько стоит гастроскопия", rows)) is None
    assert _Down.calls == 1
