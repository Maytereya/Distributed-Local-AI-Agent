"""Выбор строки прайса через модуль `choose()` (ход (в), клиент 1) — через обработчик эндпоинта.

Выключатель `[MESSENGER_ROUTER] choose_price = off | legacy | on` (по умолчанию off): `legacy` —
прежний промпт через модуль (доказательство переноса на записанных ответах прод-LLM), `on` — роли,
JSON Schema, коды причин. Политика сбоя прежняя: сбой и мусор — ответ правил (её меняет L-13).

Инварианты класса:
- перенос не меняет поведения там, где LLM недоступна: эталон пути цены (45 диалогов, мини-прод,
  LLM недоступна) одинаков при off, legacy и on;
- смысл, а не буквы: «гастроскопия» — ФГДС, найденная вторым кругом по термину модели, а не
  «Гастрин» по похожим буквам; цены «Гастрина» в ответе нет;
- ответ модели — недоверенный ввод: номер вне списка → ответ правил, строк «от модели» нет;
- выбор записан в телеметрию хода: задача, статус, число вариантов.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.test_price_turns_golden import DIALOGS, FIXTURE, _golden, run_dialog, synonym_fork  # noqa: F401 — фикстура для synonym_fork-диалогов

_PRICE_PROMPT_MARK = "ответить пациенту на вопрос о цене"


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    """Мир стенда на фикстуре «мини-прод» — тот же, что у эталона пути цены."""

    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


def _set_mode(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_CHOOSE_PRICE", mode, raising=False)


class _PriceOnlyLLM:
    """LLM отвечает только на выбор строки прайса; остальные места — «недоступна», как в эталоне.

    `decide(rows_by_name)` → (номера, термины) в номерах списка, который видит модель.
    """

    def __init__(self, decide) -> None:
        self.decide = decide
        self.price_calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _PRICE_PROMPT_MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.price_calls += 1
        rows = {}
        for line in seen.splitlines():
            head, dot, rest = line.partition(". ")
            if dot and head.isdigit() and " — " in rest:
                rows[rest.rsplit(" — ", 1)[0]] = int(head)
        picks, terms = self.decide(rows)
        if system is None:
            answer = {"match": picks, "search_terms": terms}
        else:
            answer = {"picks": picks, "reason": "none" if picks else "other_service", "search_terms": terms}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _fgds_by_term(rows: dict[str, int]):
    if "Эзофагогастродуоденоскопия (ФГДС)" in rows:
        return [rows["Эзофагогастродуоденоскопия (ФГДС)"]], []
    return [], ["ФГДС"]


@pytest.mark.parametrize("mode", ["legacy", "on"])
@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_price_path_without_llm_is_the_golden_master_in_every_mode(case, mode, mini_prod_world, request, monkeypatch):
    _set_mode(monkeypatch, mode)
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog(DIALOGS[case], f"choose-{case}-{mode}")
    assert record == _golden()[case]


@pytest.mark.parametrize("mode", ["legacy", "on"])
def test_gastroscopy_is_fgds_by_model_term_not_gastrin_by_letters(mode, mini_prod_world, monkeypatch):
    _set_mode(monkeypatch, mode)
    llm = _PriceOnlyLLM(_fgds_by_term)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        [turn] = run_dialog(["сколько стоит гастроскопия"], f"choose-fgds-{mode}")

    assert "ФГДС" in turn["text"]
    assert "Гастрин" not in turn["text"] and "750" not in turn["text"]
    assert llm.price_calls == 2  # первый круг — «нет», второй — по термину модели


@pytest.mark.parametrize("mode", ["legacy", "on"])
def test_out_of_range_pick_keeps_rules_answer(mode, mini_prod_world, monkeypatch):
    _set_mode(monkeypatch, mode)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceOnlyLLM(lambda rows: ([999], [])))
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog(DIALOGS["one_gastroscopy"], f"choose-invalid-{mode}")

    assert record == _golden()["one_gastroscopy"]


def test_choice_is_written_to_turn_telemetry(mini_prod_world, monkeypatch):
    _set_mode(monkeypatch, "on")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceOnlyLLM(_fgds_by_term))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:  # мир стенда сам собирает строки телеметрии
        run_dialog(["сколько стоит гастроскопия"], "choose-telemetry")
        row = world.take_telemetry()

    choices = row["choose"]
    assert [(c["task"], c["status"]) for c in choices] == [("price_service", "none_fit"), ("price_service", "picked")]
    assert choices[0]["reason"] == "other_service" and all(c["options"] >= 1 for c in choices)
    assert [c["site"] for c in row["llm"] if c["site"].startswith("choose_")] == ["choose_price_service"] * 2
