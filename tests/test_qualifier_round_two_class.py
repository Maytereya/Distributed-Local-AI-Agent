"""Ответ LLM «условия нет» — окончательный у класса A и у `p2_llm_judge` (BUG-2026-10-16-QUALIFIER-ROUND-TWO-BUDGET).

П2 (решение владельца): уточнение, которого клиника не оказывает («по ОМС», «по полису», «бесплатно»), — честное «не
нашёл», а не цена «почти той» услуги. Его судит выбор строки (`select_price_rows(judge_qualifier=True)`, правило 3 промпта
`choose_price_service.txt`): пустой выбор с кодом `qualifier_absent`. Но с поисковыми терминами пустой выбор вёл во второй
круг (поиск по терминам LLM): у класса A (`parse_price_guard = llm`) разбор хода — решение 1, выбор строки — 2, второй круг
упирался в лимит решений хода (`budget`) → `unavailable` → ответ правил («Уточните»), а не «не нашёл». У `p2_llm_judge`
второй круг тратил второй вызов LLM и искал «почти то» — то, что П2 и запрещает. Путь сильной фразы исправлен раньше
(`one_round`, BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS); здесь — то же для обоих оставшихся клиентов, в одном месте.

Инварианты класса (через обработчик эндпоинта, мини-прод, прод-набор 16.10, оракул LLM):
- `qualifier_absent` с терминами — «Не нашёл» за ОДИН вызов выбора строки, без `budget` (класс A; судья П2 под
  разбор хода — единственный режим; второй ход «а сколько стоит по омс?» после цены);
- прочие коды (`other_service`) — второй круг как раньше (у судьи П2 — второй вызов; у класса A под `turn_parse = on` —
  третье решение хода; до 19.10 упирался в лимит хода 2);
- сбой LLM выбора строки — ответ как раньше;
- реплики без условия — ответ и число вызовов как раньше.
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.spans_oracle import keep_llm_breaker_closed
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _ROWS_MARKS, _Oracle

# Прод 16.10 поверх набора деплоя: разбор хода, выбор строки промптом модуля (коды причин), L-13, класс A, судья П2.
_PROD_1016 = {
    "MR_PRICE_MATCH_POLICY": "on",
    "MR_STRONG_MATCH": "data",
    "MR_STRONG_PHRASE_GUARD": "llm",
}
# Выбор строки на пути ответа о цене: обычный промпт и промпт суда условия (`choose_price_service_qualifier_general`; на проде —
# единственный путь суда с 22.10, выключатель `price_candidates_by_meaning` снят; промпты `*_general` — с 31.10 единственные на
# пути цены, выключатель `price_select_rules` снят).
_SITES = ("choose_price_service_general", "choose_price_service_qualifier_general")

# Класс A: разбор (LLM) выбирает одну СЛАБУЮ фразу вместе с условием (правило 3 промпта разбора) — её судит выбор строки.
_CLASS_A = {
    "Сколько стоит анализ крови на концентрацию ферритина по ОМС": "анализ крови на концентрацию ферритина по ОМС",
    "сколько стоит анализ крови на концентрацию ферритина бесплатно": "анализ крови на концентрацию ферритина бесплатно",
    "анализ крови на концентрацию ферритина по полису сколько стоит": "анализ крови на концентрацию ферритина по полису",
}
# Судья П2: резолвер по реплике отказывает (П2), название — из сущности NLU / контекста.
_P2 = "сколько стоит узи щитовидной железы по омс"
# Два хода: услуга из контекста — сильная по данным МИС (её помнит сессия), второй ход — условие без названия.
_TWO_TURNS = ("сколько стоит ферритин", "а сколько стоит по омс?")


def _with_condition(question: str) -> bool:
    """Реплика пациента с условием, которого нет в прайсе (так «видит» её оракул — правило 3 промпта)."""

    return any(word in question.lower() for word in ("омс", "полис", "бесплатно"))


class _Judge(_Oracle):
    """Разбор — фразы `phrases`; выбор строки на реплике с условием — пусто с кодом `reason` и терминами второго круга;
    на прочих — строки `rows` (как `_Oracle`); `rows_down` — выбор строки недоступен везде."""

    def __init__(self, phrases: tuple[str, ...] = (), reason: str = "qualifier_absent", rows: tuple[str, ...] = ("—",), *, rows_down: bool = False) -> None:
        super().__init__(phrases, rows)
        self.reason = reason
        self.rows_down = rows_down

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if any(mark in seen for mark in _ROWS_MARKS):
            if self.rows_down:
                raise ConnectionError("тест: выбор строки недоступен")
            question = re.search(r"пациента: «(.*?)»", prompt, re.S)
            if question and _with_condition(question.group(1)):
                self.asked.append("rows")
                answer = {"picks": [], "reason": self.reason, "search_terms": ["ферритин сыворотки", "узи щитовидной железы"]}
                return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        return await super().generate(prompt=prompt, system=system, **kwargs)


def _run(turns, world, monkeypatch, *, llm: _Oracle, session: str, **switches) -> tuple[list[dict], list[dict]]:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    for key, value in {**_PROD_1016, **switches}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(world, "day") as frozen:
        record = run_dialog(list(turns), session)
        rows = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return record, rows


def _calls(row: dict, outcome: str = "ok") -> int:
    """Вызовы выбора строки в ходе с исходом `outcome` (без тени)."""

    return sum(1 for call in row.get("llm") or [] if call.get("site") in _SITES and call.get("outcome") == outcome and not call.get("shadow"))


def _priced(text: str) -> bool:
    return bool(re.search(r"\d\s*руб", text))


def _sid(*parts: str) -> str:
    return "qr2-" + "-".join(re.sub(r"\W+", "_", part)[-14:] for part in parts)


def test_precondition_the_class_a_phrase_is_weak_and_judged(mini_prod_world, monkeypatch):  # noqa: F811
    """Предусловие класса A на мини-прайсе: фраза разбора с условием — слабая (строк МИС нет), гард П2 по всей реплике
    отказывает — её судит выбор строки (`price_guard: llm`), а не путь сильной фразы."""

    from agent_logic_2.nayka_api import api_price

    from messengers_router.services import _prices_helpers as H
    from messengers_router.services import _turn_phrases as P

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        for reply, phrase in _CLASS_A.items():
            options = [option.raw for option in P.phrase_candidates(reply, facet=None, catalog=rows).options]
            assert phrase in options and not P.phrase_rows(phrase, rows), (reply, options)
            assert H.query_has_unsatisfiable_qualifier(reply, rows), reply
    reply, phrase = next(iter(_CLASS_A.items()))
    _record, [row] = _run([reply], mini_prod_world, monkeypatch, llm=_Judge((phrase,)), session=_sid("pre"))
    assert row["tu_parse_status"] == "picked" and row.get("price_guard") == "llm"
    assert "strong_phrase_guard" not in row and "p2_llm_judge" not in row


# --------------------------------------------------------------------------- класс A (`parse_price_guard = llm`)


@pytest.mark.parametrize("reply", _CLASS_A)
def test_class_a_condition_absent_is_not_found_in_one_round(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """«Условия нет» — «Не нашёл» сразу: второй круг не идёт, лимит хода не задет (до фикса: `budget` → «Уточните»)."""

    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, llm=_Judge((_CLASS_A[reply],)), session=_sid("a-qa", reply))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert _calls(row) == 1 and _calls(row, "budget") == 0, row.get("llm")
    assert row.get("price_policy") == "not_found"


@pytest.mark.parametrize("reply", _CLASS_A)
def test_class_a_other_reason_keeps_the_second_round(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """`other_service` с терминами — второй круг как раньше: разбор — решение 1, выбор строки — 2, второй круг — 3.
    До 19.10 (лимит хода 2 решения) второй круг здесь всегда упирался в лимит (`budget`) — на проде 14–19.10 так
    отрезано 7 из 73 выборов строки цены; с лимитом 3 он выполняется (STEP-2026-10-19-LLM-TURN-BUDGET)."""

    phrase = _CLASS_A[reply]
    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, llm=_Judge((phrase,), "other_service"), session=_sid("a-os", reply))
    assert _calls(row) == 2 and _calls(row, "budget") == 0, row.get("llm")
    assert not _priced(turn["text"]), turn["text"]  # оракул и во втором круге не выбирает строк — цены нет


@pytest.mark.parametrize("reply", _CLASS_A)
def test_class_a_rows_down_answers_as_the_rules_guard(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM выбора строки — как раньше: ответ правил с гардом П2, цены нет (до 17.10 — равен ответу под
    `parse_price_guard = rules`; выключатель снят)."""

    phrase = _CLASS_A[reply]
    [down], _rows = _run([reply], mini_prod_world, monkeypatch, llm=_Judge((phrase,), rows_down=True), session=_sid("a-d", reply))
    assert not _priced(down["text"]), down["text"]


def test_class_a_without_a_condition_does_not_change(mini_prod_world, monkeypatch):  # noqa: F811
    """Вне класса: слабая фраза без условия — выбор строки находит услугу, один вызов, цена (как с 15.10)."""

    reply, phrase = "Сколько стоит анализ крови на концентрацию ферритина", "анализ крови на концентрацию ферритина"
    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, llm=_Judge((phrase,), rows=("Ферритин",)), session=_sid("a-out"))
    assert "Ферритин — 530" in turn["text"], turn["text"]
    assert _calls(row) == 1 and row.get("price_guard") == "llm"


# --------------------------------------------------------------------------- судья П2 (единственный путь с 17.10)


def test_p2_judge_condition_absent_is_not_found_in_one_round(mini_prod_world, monkeypatch):  # noqa: F811
    """«Условия нет» — «Не нашёл» одним вызовом (до фикса — второй круг по терминам: лишний вызов LLM, поиск «почти того»)."""

    [turn], [row] = _run([_P2], mini_prod_world, monkeypatch, llm=_Judge(), session=_sid("p2-qa", "on"))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert _calls(row) == 1 and _calls(row, "budget") == 0, row.get("llm")
    assert row["p2_llm_judge"] == "not_found"


def test_p2_judge_other_reason_keeps_the_second_round(mini_prod_world, monkeypatch):  # noqa: F811
    """`other_service` с терминами — второй круг как раньше; пусто и в нём — «не нашёл» (`none_fit` с кандидатами)."""

    [turn], [row] = _run([_P2], mini_prod_world, monkeypatch, llm=_Judge(reason="other_service"), session=_sid("p2-os", "on"))
    assert _calls(row) == 2 and _calls(row, "budget") == 0, row.get("llm")
    assert not _priced(turn["text"]) and row["p2_llm_judge"] == "not_found", turn["text"]


def test_p2_judge_rows_down_answers_as_off(mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM выбора строки — как раньше: ответ без судьи (название из сущности судье не отдаётся), телеметрия
    `fallback`."""

    from messengers_router.services import prices

    [on], [row] = _run([_P2], mini_prod_world, monkeypatch, llm=_Judge(rows_down=True), session=_sid("p2-d", "on"))
    with monkeypatch.context() as mp:
        mp.setattr(prices, "_p2_name_from_entity", lambda *_args, **_kwargs: False)
        [off], _rows = _run([_P2], mini_prod_world, monkeypatch, llm=_Judge(rows_down=True), session=_sid("p2-doff", "on"))
    assert on["text"] == off["text"], (on["text"], off["text"])
    assert row["p2_llm_judge"] == "fallback"


@pytest.mark.xfail(
    strict=True,
    reason="BUG-2026-10-27-OMS-FOLLOWUP-PRICED: под разметкой turn_spans (прод с 24.10) условие без названия во втором ходе "
    "(«а сколько стоит по омс?» после цены ферритина) — разбор none_fit, судья П2 не спрашивается, платная цена услуги контекста",
)
def test_two_turns_condition_after_a_price(mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода: цена ферритина → «а сколько стоит по омс?» (услуга из контекста, судья П2): «Не нашёл» одним вызовом,
    без `budget`; первый ответ — цена."""

    # Путь прода (`turn_spans_llm = on`): разметка первого хода называет «ферритин», второго — ни одной услуги (условие без
    # названия — не услуга, правило 7 промпта `turn_spans`); предохранитель не размыкается от мест, на которые подделка не
    # отвечает. Дефект прода (BUG-2026-10-27-OMS-FOLLOWUP-PRICED): второй ход — «Розничная цена: Ферритин — 530 руб.» без судьи П2.
    keep_llm_breaker_closed(monkeypatch)
    oracle = _Judge(("ферритин",), rows=("Ферритин",))
    (first, second), rows = _run(_TWO_TURNS, mini_prod_world, monkeypatch, llm=oracle, session=_sid("two", "on"))
    assert "Ферритин — 530" in first["text"], first["text"]
    assert not _priced(second["text"]) and "Не нашёл" in second["text"], second["text"]
    assert rows[-1]["p2_llm_judge"] == "not_found"
    assert _calls(rows[-1]) == 1 and _calls(rows[-1], "budget") == 0, rows[-1].get("llm")
