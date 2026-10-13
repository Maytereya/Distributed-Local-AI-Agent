"""Политика силы совпадения для цены (L-13, решение владельца 07.10 — вариант B) — через эндпоинт.

Выключатель `[MESSENGER_ROUTER] price_match_policy = off | shadow | on` (по умолчанию shadow).

Инварианты класса:
- сильное совпадение по данным (дословное название, синоним МИС, код) — только эти строки, без LLM;
- слабое совпадение, а LLM не ответила внятно: ОДНА слабая строка не показывается с ценой — до 5
  названий с самым различающим словом пациента, без цен; таких нет — честный перевод; СПИСОК правил
  (≥2 строк) остаётся списком (решение владельца 11.10 по замеру: строгий вариант B прятал и списки
  с нужной строкой — форму 22.07); «одна» и «список» — по строкам, которые увидит пациент (`shown`),
  а не по числу кандидатов правил: у приёма рендер пакета показывает одну строку из нескольких
  кандидатов — она без цены (BUG-2026-10-13-L13-SHOWN-SHADOWED);
- LLM не нашла услугу и во втором круге — «не нашёл», без чужой цены;
- политика заменяет только слабое предложение правил: правила ничего не предложили — их ответ;
- off и shadow не меняют ни одного ответа; on меняет только объявленные диалоги (`POLICY_PLANNED`) —
  эталон пути цены при недоступной LLM, ответы под on — `tests/golden/price_turns_policy_on.json`
  (перезапись: `GOLDEN_UPDATE=1`, дифф — в коммит с разбором);
- тень пишет решение в телеметрию хода (`price_policy`), ответ прежний;
- контекст = показанное: после ответа политики без цены в состоянии нет услуги — ни строки, которую
  пациент не видел или не выбирал, ни его фразы (её следующий ход заземлил бы по буквам в ту же чужую
  строку); следующий короткий вопрос («сколько стоит?») цены не называет — уточнение
  (BUG-2026-10-13-L13-CONTEXT-LEAK, решение владельца 13.10, вариант (а)).
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
    "family_glucose": "дословное название «Глюкоза» — только эта строка, без мочи и синовиальной жидкости",
    "name_with_commas": "дословное название с уточнением в скобках — ровно эта строка",
    "synonym_fork_oak": "синоним МИС «оак» — ровно его услуги из справочника клиники, без хвоста семейства",
    "name_with_and": "одна слабая строка без LLM — её название без цены, выбирает пациент; услуги в контексте нет (13.10, вариант (а))",
    "one_neurologist": "приём: пациент увидел бы одну слабую строку из пяти кандидатов — названия без цены; услуги в контексте нет (13.10)",
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


def test_single_weak_row_without_llm_is_named_without_its_price(mini_prod_world, monkeypatch):
    _policy(monkeypatch, "on")
    turn = _turn("сколько стоит экспресс тест определения сифилиса антитела igg igm к", "policy-single", mini_prod_world)
    assert "руб" not in turn["text"] and "520" not in turn["text"]
    assert "«Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema pallidum)»" in turn["text"]


def test_weak_list_without_llm_stays_the_rules_list(mini_prod_world, monkeypatch):
    # Сужение варианта B (решение 11.10): список остаётся — при живой LLM его решает выбор модели
    # (`test_choose_price.py::test_gastroscopy_is_fgds_by_model_term_not_gastrin_by_letters`).
    _policy(monkeypatch, "on")
    turn = _turn("сколько стоит гастроскопия", "policy-gastro", mini_prod_world)
    assert turn["text"] == _golden()["one_gastroscopy"][0]["text"]
    assert turn["telemetry"]["price_policy"] == "rules_list"


def test_one_shown_row_of_several_candidates_is_named_without_its_price(mini_prod_world, monkeypatch):
    # Приём: правила собрали пять строк невролога, рендер пакета покажет одну — её цена без LLM не
    # звучит, хотя кандидатов больше одного (BUG-2026-10-13-L13-SHOWN-SHADOWED).
    _policy(monkeypatch, "on")
    turn = _turn("сколько стоит прием невролога", "policy-one-of-many", mini_prod_world)
    assert turn["plan"] == ["PRICE", "service_bundle_info"]
    assert "руб" not in turn["text"] and "2 700" not in turn["text"]
    assert "— Прием (осмотр, консультация) врача- невролога\n" in turn["text"]
    assert turn["telemetry"]["price_policy"] == "names"
    assert not turn["handoff"]


def _weak_rows(count: int) -> list[dict]:
    return [
        {"serviceName": f"Прием (осмотр, консультация) врача-невролога, программа {chr(1040 + i)}", "serviceHomecode": f"77{i:02d}", "cost": 2000 + i}
        for i in range(count)
    ]


@pytest.mark.parametrize(
    ("mode", "shown_count", "decision", "label"),
    [
        ("on", 1, "unavailable", "names"),  # пациент увидел бы одну строку из двенадцати — её цена не звучит
        ("shadow", 1, None, "names"),  # тень: ответ прежний, решение — в телеметрии
        ("on", 3, None, "rules_list"),  # список — остаётся списком правил
        ("on", None, None, "rules_list"),  # `shown` не передан — правила покажут всех кандидатов
    ],
)
def test_policy_decides_by_rows_the_patient_sees_not_by_candidate_count(mode, shown_count, decision, label, monkeypatch, tmp_path):
    import asyncio

    from messengers_router import turn_telemetry
    from messengers_router.services import _price_select_llm as PS

    _policy(monkeypatch, mode)
    monkeypatch.setattr(PS, "_journal_path", lambda: tmp_path / "journal.jsonl")
    rows = _weak_rows(12)  # кандидатов правил больше, чем первых в списке для LLM (`_PROPOSAL_FIRST`)
    extra = {} if shown_count is None else {"shown": rows[:shown_count]}

    async def run():
        token = turn_telemetry.begin("test", session_id="policy-shown")
        selection = await PS.select_price_rows("сколько стоит прием невролога", rows, candidates=rows, **extra)
        return selection, turn_telemetry.finish(token, handoff=False, chars=0)

    selection, row = asyncio.run(run())
    assert row["price_policy"] == label
    if decision is None:
        assert selection is None
    else:
        assert selection.status == decision and selection.rows == ()
        assert selection.names


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
    turn = _turn("сколько стоит экспресс тест определения сифилиса антитела igg igm к", "policy-shadow", mini_prod_world)
    assert turn["text"] == _golden()["name_with_and"][0]["text"]
    assert turn["telemetry"]["price_policy"] == "names"


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


# --- Контекст = показанное: два хода (BUG-2026-10-13-L13-CONTEXT-LEAK) ----------------------------
#
# Обязательный класс плана с 13.10 («Слепое пятно плана», мастер-хэндофф §4 (б)): после ответа без
# цены следующий короткий вопрос не берёт услугу, которую пациент не видел и не выбирал. Первый ход —
# ответ политики под on: «не нашёл» (LLM выбора цены отвечает «не та услуга» в обоих кругах, остальные
# места LLM недоступны) и названия без цен (вариант B, LLM недоступна). Мини-прод: правила находят на
# «гастроскопию» анализы «Гастрин» (750 / 1 950 руб.), на «анализ ттг» — две строки ТТГ.
#
# Решение владельца 13.10, вариант (а): после такого ответа услуга из контекста снимается. Фраза
# пациента в контексте (`233d22e`) не спасала: следующий ход заземлял её заново по буквам (каталожная
# инъекция роутера, `catalog_service_exact`: «гастроскопия» → «Гастрин», «анализ ттг» → «ТТГ (TSH)…»),
# и «сколько стоит?» называл чужую цену. Теперь короткий вопрос получает уточнение.

_ANSWER_WITHOUT_PRICE = {
    "not_found": ("сколько стоит гастроскопия", ("Гастрин", "Гастрин-17 стимулированный")),
    "names_without_prices": ("сколько стоит анализ ттг", ("ТТГ (TSH) тиреотропный гормон", "Антитела к рецепторам ТТГ")),
}
_SHORT_FOLLOWUPS = ("сколько стоит?", "а цена?", "и сколько?")
_PRICE_MARK = " руб."  # цена в ответе — «750 руб.»; без пробела ловилось бы «Трубина» из подсказки


def _two_turns(scenario: str, followup: str, mini_prod_world, monkeypatch, tmp_path) -> list[dict]:
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _price_select_llm as PS

    _policy(monkeypatch, "on")
    monkeypatch.setattr(PS, "_journal_path", lambda: tmp_path / "journal.jsonl")
    if scenario == "not_found":
        monkeypatch.setattr(cfg, "MR_CHOOSE_PRICE", "on", raising=False)
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceLLM([[]] * 4, terms=["ФГДС"]))
    question, _unseen = _ANSWER_WITHOUT_PRICE[scenario]
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        return run_dialog([question, followup], f"policy-two-turns-{scenario}-{followup}")


@pytest.mark.parametrize("followup", _SHORT_FOLLOWUPS)
@pytest.mark.parametrize("scenario", sorted(_ANSWER_WITHOUT_PRICE))
def test_context_after_an_answer_without_price_is_what_the_patient_was_shown(scenario, followup, mini_prod_world, monkeypatch, tmp_path):
    first, second = _two_turns(scenario, followup, mini_prod_world, monkeypatch, tmp_path)

    assert _PRICE_MARK not in first["text"], first["text"]  # первый ход — ответ политики без цены
    assert first["service_name_effective"]  # ответ назвал фразу пациента
    # Контекст = показанное: пациент не увидел цены и не выбрал услугу — в состоянии услуги нет: ни
    # строки прайса, которую он не видел («Гастрин»), ни фразы, которую следующий ход заземлил бы по
    # буквам в ту же строку.
    assert first["service_name"] is None
    # Второй ход не переносит в контекст чужую строку словами пациента без услуги.
    assert second["service_name"] is None


@pytest.mark.parametrize("followup", _SHORT_FOLLOWUPS)
@pytest.mark.parametrize("scenario", sorted(_ANSWER_WITHOUT_PRICE))
def test_short_question_after_an_answer_without_price_names_no_price(scenario, followup, mini_prod_world, monkeypatch, tmp_path):
    _first, second = _two_turns(scenario, followup, mini_prod_world, monkeypatch, tmp_path)
    _question, unseen = _ANSWER_WITHOUT_PRICE[scenario]

    # Пациент не видел цены и не выбрал услугу: цена во втором ответе — чужая или выбранная за него.
    assert _PRICE_MARK not in second["text"], second["text"]
    assert not any(name in second["text"] for name in unseen if scenario == "not_found"), second["text"]
    assert not second["handoff"]  # уточнение, а не перевод: короткий вопрос — не повод звать оператора


def test_policy_handoff_leaves_no_service_in_context(mini_prod_world, monkeypatch):
    """Перевод на оператора — третья форма ответа политики (LLM недоступна, названий целым словом нет):
    до `build_*` он не доходит, контекст снимает общий сброс после перевода (`clear_on_handoff` в
    `patient_routing_stream`). Тот же инвариант: следующий короткий вопрос цены не называет. Без сброса
    (мутация) «сколько стоит?» отвечал «Ферритин — 530 руб.» — услугой, которую пациент не видел."""

    _policy(monkeypatch, "on")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog(["сколько стоит ферритинн", "сколько стоит?"], "policy-handoff-two-turns")
        first_turn = list(world.telemetry)[0]
    assert first_turn["price_policy"] == "handoff"  # первый ход — именно перевод политики
    assert first["handoff"] and first["service_name"] is None
    assert second["service_name"] is None
    assert _PRICE_MARK not in second["text"], second["text"]
