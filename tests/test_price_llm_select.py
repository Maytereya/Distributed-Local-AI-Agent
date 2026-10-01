"""LLM выбирает строки прайса, которые являются ровно той услугой из вопроса (01.10).

Класс дефектов «цена не той услуги» (журнал, SURGEON-PRICE-FOR-SPECIALIST): правила
подбирали строку прайса по похожим словам — «приём эндокринолога» → «Прием … хирурга
(эндокринологическое отделение)», «гастроскопия» → анализ «Гастрин», «приём
пластического хирурга» → обычный хирург. Решение владельца 30.09: решает LLM, правила
только собирают кандидатов.

Инварианты класса:
  1) LLM выбирает только из реальных строк прайса — выдумать позицию не может;
  2) верный выбор правил LLM лишь подтверждает — ответ не меняется;
  3) неверный выбор правил заменяется строками, которые LLM признала той услугой;
  4) «не нашла» от LLM не стирает ответ правил — заменяет только положительный выбор;
  5) сбой, таймаут, мусор, режим strict, kill-switch → ответ правил без изменений;
  6) особые варианты (cito, капиллярная кровь, к.м.н., на дому…), которых пациент не
     просил, не показываются, если есть базовый;
  7) термин, которым LLM нашла услугу во втором проходе, пишется в журнал для клиники —
     синонимы заносит клиника, не мы.
LLM в тестах замокана; живое поведение — свип по прайсу и remote eval.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from messengers_router.services import _price_select_llm as PS
from messengers_router.services import Services
from messengers_router.services import core as svc_mod

_SAMARA = "г. Самара, пр. Ленина, 5"

_ENDO_SURGEON = {"serviceName": "Прием (осмотр, консультация) врача-хирурга (эндокринологическое отделение)", "serviceHomecode": "15.2.1.2", "cost": 4500}
_ENDO_THERAPIST = {"serviceName": "Прием (осмотр, консультация) врача-терапевта (эндокринологическое отделение)", "serviceHomecode": "13.1.3.15", "cost": 3950}
_ENDO = {"serviceName": "Приём эндокринолога", "serviceHomecode": "17.1.14", "cost": 3000}
_GASTRIN = {"serviceName": "Гастрин", "serviceHomecode": "1001", "cost": 750}
_GASTRIN17 = {"serviceName": "Гастрин-17 стимулированный", "serviceHomecode": "1002", "cost": 1950}
_FGDS = {"serviceName": "Эзофагогастродуоденоскопия (ФГДС)", "serviceHomecode": "22.2.1", "cost": 5100}
_HCG = {"serviceName": "ХГЧ (общий)", "serviceHomecode": "2001", "cost": 420}
_HCG_CITO = {"serviceName": "Cito ХГЧ (общий)", "serviceHomecode": "2002", "cost": 740}
_FERRITIN = {"serviceName": "Ферритин", "serviceHomecode": "3001", "cost": 530}
_THERAPIST = {"serviceName": "Прием (осмотр, консультация) врача-терапевта", "serviceHomecode": "13.1.1", "cost": 2700}
_ENDO_CENTER = {"serviceName": "Прием эндокринолога Центра здоровья", "serviceHomecode": "17.1.22", "cost": 4500}

RETAIL = [_ENDO_SURGEON, _ENDO_THERAPIST, _ENDO, _GASTRIN, _GASTRIN17, _FGDS, _HCG, _HCG_CITO, _FERRITIN, _THERAPIST, _ENDO_CENTER]


def run(coro):
    return asyncio.run(coro)


def _rows_in_prompt(prompt: str) -> dict[str, int]:
    """Номера строк прайса, как их увидела LLM: «12. Название — 500 руб.»."""
    out: dict[str, int] = {}
    for num, name in re.findall(r"^(\d+)\. (.+?) — \d+ руб\.$", prompt, flags=re.M):
        out[name] = int(num)
    return out


def _fake_llm(monkeypatch, decide, calls: list | None = None):
    """decide(prompt, rows_by_name) -> dict ответа LLM (будет сериализован в JSON)."""

    async def fake_generate_text(prompt: str, **kwargs):
        if calls is not None:
            calls.append(prompt)
        reply = decide(prompt, _rows_in_prompt(prompt))
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)

    monkeypatch.setattr(PS, "generate_text", fake_generate_text)


def _pick(*names: str):
    """Решительная LLM: названное — та услуга, всё остальное в списке — другая."""

    def decide(prompt, rows):
        match = [rows[n] for n in names if n in rows]
        return {"match": match, "other": [num for num in rows.values() if num not in match], "search_terms": []}

    return decide


def _pick_quietly(*names: str):
    """LLM выбрала часть вариантов, но ни одну строку не назвала другой услугой."""

    def decide(prompt, rows):
        return {"match": [rows[n] for n in names if n in rows], "other": [], "search_terms": []}

    return decide


@pytest.fixture(autouse=True)
def _enabled_and_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(PS, "_ENABLED", True)
    monkeypatch.setattr(PS, "_journal_path", lambda: tmp_path / "price_synonym_suggestions.jsonl")
    yield


# --- сам выбор ----------------------------------------------------------------


def test_llm_picks_the_asked_service_among_lexical_lookalikes(monkeypatch):
    _fake_llm(monkeypatch, _pick("Приём эндокринолога"))

    sel = run(PS.select_price_rows("сколько стоит приём эндокринолога", RETAIL))

    assert [r["serviceName"] for r in sel.rows] == ["Приём эндокринолога"]


def test_llm_cannot_invent_a_row(monkeypatch):
    # Номер вне списка и строка, которой нет в прайсе, — мусор: ответ правил не трогаем.
    _fake_llm(monkeypatch, lambda prompt, rows: {"match": [999], "search_terms": []})
    assert run(PS.select_price_rows("сколько стоит приём эндокринолога", RETAIL)) is None


@pytest.mark.parametrize("reply", ["не json", "[]", '{"match": "3"}', ""])
def test_garbage_reply_keeps_rules_answer(monkeypatch, reply):
    _fake_llm(monkeypatch, lambda prompt, rows: reply)
    assert run(PS.select_price_rows("сколько стоит приём эндокринолога", RETAIL)) is None


def test_llm_failure_keeps_rules_answer(monkeypatch):
    async def boom(prompt: str, **kwargs):
        raise TimeoutError("llm busy")

    monkeypatch.setattr(PS, "generate_text", boom)
    assert run(PS.select_price_rows("сколько стоит приём эндокринолога", RETAIL)) is None


@pytest.mark.parametrize("mode", ["strict"])
def test_strict_mode_does_not_call_llm(monkeypatch, mode):
    calls: list = []
    _fake_llm(monkeypatch, _pick("Приём эндокринолога"), calls)
    assert run(PS.select_price_rows("сколько стоит приём эндокринолога", RETAIL, runtime_llm_mode=mode)) is None
    assert calls == []


def test_kill_switch_does_not_call_llm(monkeypatch):
    calls: list = []
    monkeypatch.setattr(PS, "_ENABLED", False)
    _fake_llm(monkeypatch, _pick("Приём эндокринолога"), calls)
    assert run(PS.select_price_rows("сколько стоит приём эндокринолога", RETAIL)) is None
    assert calls == []


@pytest.mark.parametrize("turn", ["2", "да", "а сколько стоит?", "ок"])
def test_turn_without_own_service_words_does_not_call_llm(monkeypatch, turn):
    # Ответ цифрой, «да», «а сколько стоит?» — услуга в контексте, его ведут правила.
    calls: list = []
    _fake_llm(monkeypatch, _pick("Ферритин"), calls)
    assert run(PS.select_price_rows(turn, RETAIL, hints=["ферритин"])) is None
    assert calls == []


def test_exact_price_name_does_not_call_llm(monkeypatch):
    # «ферритин» = «Ферритин» дословно — правила не ошибаются, LLM не нужна (и не тратит секунду).
    calls: list = []
    _fake_llm(monkeypatch, _pick("Ферритин"), calls)
    assert run(PS.select_price_rows("сколько стоит ферритин", RETAIL)) is None
    assert calls == []


def test_unsatisfiable_qualifier_stays_with_rules(monkeypatch):
    # «по ОМС»: правила честно отвечают «не нашёл» (решение владельца, П2) — LLM не зовём.
    calls: list = []
    _fake_llm(monkeypatch, _pick("Прием (осмотр, консультация) врача-терапевта"), calls)
    rows = RETAIL + [{"serviceName": f"Услуга {i}", "cost": 100} for i in range(200)]
    assert run(PS.select_price_rows("Сколько стоит приём терапевта по ОМС", rows)) is None
    assert calls == []


def test_base_variant_wins_over_unrequested_cito(monkeypatch):
    _fake_llm(monkeypatch, _pick("ХГЧ (общий)", "Cito ХГЧ (общий)"))
    sel = run(PS.select_price_rows("сколько стоит хгч", RETAIL))
    assert [r["serviceName"] for r in sel.rows] == ["ХГЧ (общий)"]


def test_requested_cito_is_kept(monkeypatch):
    _fake_llm(monkeypatch, _pick("ХГЧ (общий)", "Cito ХГЧ (общий)"))
    sel = run(PS.select_price_rows("сколько стоит хгч срочно", RETAIL))
    assert {r["serviceName"] for r in sel.rows} == {"ХГЧ (общий)", "Cito ХГЧ (общий)"}


def test_second_pass_finds_service_by_medical_term_and_journals_it(monkeypatch, tmp_path):
    calls: list = []

    def decide(prompt, rows):
        if "Эзофагогастродуоденоскопия (ФГДС)" in rows:
            return {"match": [rows["Эзофагогастродуоденоскопия (ФГДС)"]], "other": [], "search_terms": []}
        return {"match": [], "other": list(rows.values()), "search_terms": ["ФГДС"]}

    _fake_llm(monkeypatch, decide, calls)
    sel = run(PS.select_price_rows("сколько стоит гастроскопия", [_GASTRIN, _GASTRIN17, _FGDS] + [_FERRITIN]))

    assert [r["serviceName"] for r in sel.rows] == ["Эзофагогастродуоденоскопия (ФГДС)"]
    assert len(calls) == 2
    entries = [json.loads(line) for line in (tmp_path / "price_synonym_suggestions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["phrase"] == "гастроскопия"
    assert entries[-1]["terms"] == ["ФГДС"]
    assert entries[-1]["services"] == [{"name": "Эзофагогастродуоденоскопия (ФГДС)", "homecode": "22.2.1"}]


def test_nothing_found_and_nothing_rejected_keeps_rules_answer(monkeypatch):
    _fake_llm(monkeypatch, lambda prompt, rows: {"match": [], "other": [], "search_terms": ["массаж общий"]})
    sel = run(PS.select_price_rows("сколько стоит массаж", RETAIL))
    assert PS.reconcile_with_proposal([_THERAPIST], sel) is None


def test_bot_guess_widens_candidates_but_never_reaches_the_prompt(monkeypatch):
    # Свип 01.10: догадка правил по этой же реплике («биопсия» → «Биопсия вульвы»),
    # поданная LLM как «контекст», сжимала общий вопрос до одной позиции.
    seen: list = []
    _fake_llm(monkeypatch, _pick("Гастрин", "Гастрин-17 стимулированный"), seen)
    run(PS.select_price_rows("сколько стоит анализ на гастрин", RETAIL, context="Гастрин-17 стимулированный"))
    prompt = seen[-1]
    assert "Сообщение пациента: «сколько стоит анализ на гастрин»" in prompt
    assert prompt.count("Гастрин-17 стимулированный") == 1  # только как строка прайса
    assert "ранее" not in prompt and "контекст" not in prompt.lower()


# --- сверка с ответом правил -----------------------------------------------------


def _sel(*rows):
    return PS.PriceSelection(rows=tuple(rows))


def test_reconcile_keeps_rules_answer_the_llm_confirmed():
    assert PS.reconcile_with_proposal([_ENDO], _sel(_ENDO)) is None
    assert PS.reconcile_with_proposal([_HCG], _sel(_HCG, _FERRITIN)) is None


def test_reconcile_replaces_a_wrong_pick():
    assert PS.reconcile_with_proposal([_ENDO_SURGEON], _sel(_ENDO)) == [_ENDO]


def test_reconcile_keeps_only_the_asked_service_from_a_list():
    assert PS.reconcile_with_proposal([_GASTRIN, _FGDS, _GASTRIN17], _sel(_FGDS)) == [_FGDS]


def test_reconcile_adds_variants_the_rules_missed_to_a_list():
    assert PS.reconcile_with_proposal([_HCG, _GASTRIN], _sel(_HCG, _HCG_CITO)) == [_HCG, _HCG_CITO]


def test_reconcile_does_not_turn_a_confirmed_single_answer_into_a_list():
    # Подтверждённый приём остаётся одной строкой с врачами, как было.
    assert PS.reconcile_with_proposal([_THERAPIST], _sel(_THERAPIST, _ENDO_THERAPIST)) is None


def test_reconcile_llm_found_nothing_keeps_rules_answer():
    # «Не нашла» от LLM слабее, чем найденное правилами: ответ не стираем.
    assert PS.reconcile_with_proposal([_ENDO_SURGEON], _sel()) is None


def test_reconcile_without_llm_choice_keeps_rules_answer():
    assert PS.reconcile_with_proposal([_ENDO_SURGEON], None) is None


def test_reconcile_fills_an_empty_rules_answer():
    assert PS.reconcile_with_proposal([], _sel(_FGDS)) == [_FGDS]


# --- в инструментах цены -----------------------------------------------------------


def _services(monkeypatch, retail, doctors=(), doctor_prices=()):
    svc = Services()

    async def fake_regions():
        return [{"id": 1, "addressForSite": _SAMARA, "city": "Самара"}]

    async def fake_doctors():
        return [dict(d) for d in doctors]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _region_id: [dict(r) for r in retail])
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", lambda: [dict(r) for r in doctor_prices])
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", lambda *args, **kwargs: [])
    return svc


_HYBRID = {"__runtime_llm_mode": "hybrid", "__price_question": True}


def test_service_bundle_shows_endocrinologist_and_his_doctors_not_the_surgeon(monkeypatch):
    _fake_llm(monkeypatch, _pick("Приём эндокринолога"))
    endo_doctor = {"id": 7, "fio": "Эндокринолог Анна", "ord": 1, "regions": [_SAMARA], "main_units": ["Врач-эндокринолог"], "units": ["Врач-эндокринолог"]}
    svc = _services(
        monkeypatch,
        RETAIL,
        doctors=[endo_doctor],
        doctor_prices=[{"doctorId": 7, "fio": "Эндокринолог Анна", "serviceName": "Приём эндокринолога", "serviceHomecode": "17.1.14", "cost": 3000}],
    )

    payload = run(svc.service_bundle_info("сколько стоит приём эндокринолога", {"service_name": _ENDO_SURGEON["serviceName"], **_HYBRID}))

    assert payload["retail_prices"][0]["serviceName"] == "Приём эндокринолога"
    assert [d["fio"] for d in payload["doctors"]] == ["Эндокринолог Анна"]


def test_price_info_replaces_wrong_family_with_the_service_found_by_term(monkeypatch):
    def decide(prompt, rows):
        if "Эзофагогастродуоденоскопия (ФГДС)" in rows:
            return {"match": [rows["Эзофагогастродуоденоскопия (ФГДС)"]], "other": [], "search_terms": []}
        return {"match": [], "other": list(rows.values()), "search_terms": ["ФГДС"]}

    _fake_llm(monkeypatch, decide)
    svc = _services(monkeypatch, RETAIL)

    payload = run(svc.price_info("сколько стоит гастроскопия", dict(_HYBRID)))

    names = [r["serviceName"] for r in payload.get("prices") or payload.get("family_variants") or []]
    assert names == ["Эзофагогастродуоденоскопия (ФГДС)"]


def test_price_info_shows_several_llm_rows_as_variants(monkeypatch):
    # Правила выбрали чужую строку, LLM — два варианта той услуги: показываем оба.
    _fake_llm(monkeypatch, _pick("Приём эндокринолога", "Прием эндокринолога Центра здоровья"))
    svc = _services(monkeypatch, RETAIL)

    payload = run(svc.price_info("сколько стоит приём эндокринолога", dict(_HYBRID)))

    assert payload["service_kind"] == "family_query"
    assert {r["serviceName"] for r in payload["family_variants"]} == {"Приём эндокринолога", "Прием эндокринолога Центра здоровья"}


def test_price_info_keeps_rules_answer_the_llm_confirmed(monkeypatch):
    _fake_llm(monkeypatch, _pick("Гастрин", "Гастрин-17 стимулированный"))
    svc = _services(monkeypatch, RETAIL)
    monkeypatch.setattr(PS, "_ENABLED", False)
    rules_only = run(svc.price_info("сколько стоит анализ на гастрин", dict(_HYBRID)))
    monkeypatch.setattr(PS, "_ENABLED", True)

    assert run(svc.price_info("сколько стоит анализ на гастрин", dict(_HYBRID))) == rules_only


def test_tools_answer_as_before_when_llm_is_down(monkeypatch):
    async def boom(prompt: str, **kwargs):
        raise ConnectionError("llm down")

    monkeypatch.setattr(PS, "generate_text", boom)
    svc = _services(monkeypatch, RETAIL)
    for call in (
        lambda: svc.price_info("сколько стоит приём эндокринолога", dict(_HYBRID)),
        lambda: svc.service_bundle_info("сколько стоит приём эндокринолога", {"service_name": "прием эндокринолог", **_HYBRID}),
    ):
        monkeypatch.setattr(PS, "_ENABLED", False)
        rules_only = run(call())
        monkeypatch.setattr(PS, "_ENABLED", True)
        assert run(call()) == rules_only


def test_prompt_bundle_and_host_copies_match():
    # Промпты в ДВУХ местах (CLAUDE.md): прод читает app_data/prompts/mr_*, и копия
    # без правки молча оставила бы старую версию.
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bundle = (root / "messengers_router" / "prompts" / "price_service_select.txt").read_text(encoding="utf-8")
    host = (root / "app_data" / "prompts" / "mr_price_service_select.txt").read_text(encoding="utf-8")
    assert bundle == host


def test_rules_answer_is_always_among_the_candidates(monkeypatch):
    # Свип 01.10: «оак» — поиск по словам находил только строки со словом «ОАК»,
    # а ответ правил («Общий анализ крови» 390/550) LLM даже не видела и выбрала
    # урезанный «ОАК (без лейкоцитарной формулы и СОЭ)» за 250.
    full = {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "5001", "cost": 390}
    reduced = {"serviceName": "ОАК (без лейкоцитарной формулы и СОЭ)", "serviceHomecode": "5002", "cost": 250}
    seen: list = []
    _fake_llm(monkeypatch, _pick("Общий анализ крови (Le, Er, Hb, СОЭ)"), seen)

    sel = run(PS.select_price_rows("сколько стоит оак", [reduced, full, _FERRITIN], candidates=[full]))

    assert "1. Общий анализ крови (Le, Er, Hb, СОЭ)" in seen[-1]
    assert PS.reconcile_with_proposal([full], sel) is None


def test_comprehensive_service_is_not_mistaken_for_a_package(monkeypatch):
    # «(комплексное)» — основная услуга, не пакет: фильтр вариантов её не трогает.
    full = {"serviceName": "Ультразвуковое исследование органов брюшной полости (комплексное)", "serviceHomecode": "6001", "cost": 2800}
    liver = {"serviceName": "Ультразвуковое исследование печени и желчного пузыря", "serviceHomecode": "6002", "cost": 1800}
    _fake_llm(monkeypatch, _pick(full["serviceName"], liver["serviceName"]))

    sel = run(PS.select_price_rows("сколько стоит узи брюшной полости", [full, liver]))

    assert [r["serviceName"] for r in sel.rows] == [full["serviceName"], liver["serviceName"]]


def test_price_shown_in_passing_does_not_call_llm(monkeypatch):
    # Шаг записи и карточка врача зовут price_info попутно: там правила, без LLM, —
    # иначе каждый ход записи стал бы на секунду дольше.
    calls: list = []
    _fake_llm(monkeypatch, _pick("Приём эндокринолога"), calls)
    svc = _services(monkeypatch, RETAIL)

    run(svc.price_info("хочу записаться к эндокринологу", {"__runtime_llm_mode": "hybrid"}))

    assert calls == []


def test_planner_marks_only_price_questions():
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import RouteDecision, SessionState
    from messengers_router.planner import build_plan

    state = SessionState(session_id="t")
    state.last_entities = {"service_name": "прием эндокринолог"}
    price_plan = build_plan(RouteDecision(label="PRICE", confidence=0.9), state, "сколько стоит приём эндокринолога", MemoryStore())
    assert all(step.input["entities"].get("__price_question") for step in price_plan.steps)

    state.last_entities = {}
    appointment_plan = build_plan(RouteDecision(label="APPOINTMENT", confidence=0.9), state, "хочу записаться к эндокринологу", MemoryStore())
    assert not any(step.input["entities"].get("__price_question") for step in appointment_plan.steps)


def test_per_call_options_override_general_generation_settings(monkeypatch):
    # Выбор строки прайса просит temperature 0 поверх общих настроек (там 0.2),
    # остальные параметры генерации — общие.
    from messengers_router import llm_runtime

    seen: dict = {}

    class FakeClient:
        async def generate(self, **kwargs):
            seen.update(kwargs)
            return {"response": '{"match": []}'}

    from ollama import Options

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", FakeClient())
    # Настоящий тип: общие настройки — `ollama.Options`, не dict (01.10 TypeError на
    # живой LLM при dict-слиянии поймал только свип, не этот тест с dict-подделкой).
    monkeypatch.setattr(llm_runtime.ollama_settings, "options_set", lambda: Options(temperature=0.2, top_p=0.9))

    run(llm_runtime.generate_text("промпт", timeout_s=5, queue_timeout_ms=1000, fmt="json", options={"temperature": 0}))
    assert seen["options"].model_dump(exclude_none=True) == {"temperature": 0, "top_p": 0.9}

    run(llm_runtime.generate_text("промпт", timeout_s=5, queue_timeout_ms=1000))
    assert seen["options"].model_dump(exclude_none=True) == {"temperature": 0.2, "top_p": 0.9}
