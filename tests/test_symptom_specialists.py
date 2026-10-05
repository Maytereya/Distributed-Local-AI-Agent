"""Медвопрос: к каким врачам клиники обратиться с жалобой (05.10, решение владельца).

«У меня болит голова 3 дня» получало «Не совсем понял» или «По техническим вопросам
обратитесь к администратору». Теперь бот не ставит диагноз, но называет специальности
самарских врачей из МИС, к которым с такой жалобой обычно идут, и предлагает записать.

Инварианты:
- специальности — только роли приёма самарских врачей из среза МИС, без повторов;
- выбор LLM разбирается строго: номера из списка (не больше трёх), «СРОЧНО» — шаблон
  скорой, всё остальное и сбой — прежний шаблон медвопроса;
- совет никогда не ставит диагноз и всегда напоминает про скорую.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router import renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.services import Services
from messengers_router.services import _symptom_specialists as S
from messengers_router.services._regions import _normalize_region_text

_REAL_ADVISE = S.advise  # до подмены фикстурой conftest: для тестов самого совета
_ROOT = Path(__file__).resolve().parents[1]
_SPECIALTIES = ["терапевт", "невролог", "кардиолог", "оториноларинголог", "педиатр"]


def _doctor(region: str, *units: str) -> dict:
    return {"regions": [region], "unit_links": [{"company_unit_name": u} for u in units]}


# --- список специальностей ------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "shown"),
    [
        ("Врач-кардиолог", "кардиолог"),
        ("Врач невролог", "невролог"),
        ("Ревматолог (центр)", "ревматолог"),
        ("Врач-сердечно-сосудистый хирург, флеболог", "сердечно-сосудистый хирург, флеболог"),
    ],
)
def test_specialty_is_shown_without_doctor_prefix(unit, shown):
    assert S.specialty_display_name(unit) == shown


def test_only_samara_doctors_give_specialties():
    doctors = [
        _doctor("пр.Ленина, 5", "Врач невролог", "Врач-ревматолог"),
        _doctor("г. Пенза, ул. Калинина, 22А", "Врач-нефролог"),
        _doctor("ул. Победы, 83", "Ревматолог (центр)", "Врач невролог"),
    ]
    tokens = {_normalize_region_text("пр.Ленина, 5"), _normalize_region_text("ул. Победы, 83")}

    assert S.samara_specialties(doctors, tokens) == ["невролог", "ревматолог"]


def test_without_branch_list_other_cities_are_still_excluded():
    doctors = [_doctor("пр.Ленина, 5", "Врач терапевт"), _doctor("г. Ульяновск, ул. Рябикова, 60", "Врач-нефролог")]

    assert S.samara_specialties(doctors, set()) == ["терапевт"]


# --- выбор LLM -------------------------------------------------------------------


def _llm_answers(monkeypatch, answer) -> list[str]:
    prompts: list[str] = []

    async def fake_generate_text(prompt, **_kwargs):
        prompts.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(S, "generate_text", fake_generate_text)
    return prompts


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("2, 1", S.SpecialistAdvice(specialties=("невролог", "терапевт"))),
        ("Номера: 3.", S.SpecialistAdvice(specialties=("кардиолог",))),
        ("2, 2, 4, 1, 3", S.SpecialistAdvice(specialties=("невролог", "оториноларинголог", "терапевт"))),
        ("СРОЧНО", S.SpecialistAdvice(urgent=True)),
        ("НЕТ", None),
        ("0, 99", None),
        ("не знаю", None),
        ("", None),
        (TimeoutError(), None),
    ],
    ids=["two", "one_with_words", "dedup_cap_three", "urgent", "no", "out_of_range", "unclear", "empty", "llm_failure"],
)
def test_llm_choice_is_parsed_strictly(monkeypatch, answer, expected):
    prompts = _llm_answers(monkeypatch, answer)

    assert asyncio.run(S.pick_specialists("у меня болит голова 3 дня", _SPECIALTIES)) == expected
    assert "«у меня болит голова 3 дня»" in prompts[0]
    assert "2. невролог" in prompts[0]


def test_kill_switch_and_empty_catalog_skip_llm(monkeypatch):
    prompts = _llm_answers(monkeypatch, "1")

    assert asyncio.run(S.pick_specialists("болит голова", [])) is None
    monkeypatch.setattr(S._cfg, "MR_LLM_SYMPTOM_SPECIALISTS", False)
    assert asyncio.run(S.pick_specialists("болит голова", _SPECIALTIES)) is None
    assert prompts == []


class _FakeServices:
    def __init__(self, doctors, tokens, fail=False):
        self._doctors, self._tokens, self._fail = doctors, tokens, fail

    async def _ensure_doctors_cache_loaded(self):
        if self._fail:
            raise ConnectionError("МИС недоступна")
        return self._doctors

    async def _samara_region_tokens(self):
        return self._tokens


def test_advice_offers_only_samara_specialties(monkeypatch):
    seen: list[list[str]] = []

    async def fake_pick(_text, specialties):
        seen.append(list(specialties))
        return S.SpecialistAdvice(specialties=tuple(specialties[:1]))

    monkeypatch.setattr(S, "pick_specialists", fake_pick)
    services = _FakeServices(
        [_doctor("пр.Ленина, 5", "Врач терапевт"), _doctor("г. Пенза, ул. Калинина, 22А", "Врач-нефролог")],
        {_normalize_region_text("пр.Ленина, 5")},
    )

    assert asyncio.run(_REAL_ADVISE("болит голова", services)) == S.SpecialistAdvice(specialties=("терапевт",))
    assert seen == [["терапевт"]]


def test_advice_failures_fall_back_to_template():
    assert asyncio.run(_REAL_ADVISE("болит голова", None)) is None
    assert asyncio.run(_REAL_ADVISE("болит голова", _FakeServices([], set(), fail=True))) is None


def test_prompt_copies_are_identical():
    for key in ("symptom_specialists", "classifier_patient"):
        bundle = (_ROOT / "messengers_router" / "prompts" / f"{key}.txt").read_text(encoding="utf-8")
        host = (_ROOT / "app_data" / "prompts" / f"mr_{key}.txt").read_text(encoding="utf-8")
        assert bundle == host, key


# --- ответ пациенту ----------------------------------------------------------------


def test_advice_names_doctors_without_diagnosis():
    answer = renderer.render_medical_advice(S.SpecialistAdvice(specialties=("невролог", "терапевт")))

    assert "невролог, терапевт" in answer.text
    assert "поставить диагноз и назначить лечение в чате я не могу" in answer.text.lower()
    assert "103" in answer.text
    assert "специалистам нашей клиники" in answer.text
    assert answer.handoff is False  # дальше пациент выбирает врача — запись у бота


def test_single_specialty_reads_naturally():
    answer = renderer.render_medical_advice(S.SpecialistAdvice(specialties=("терапевт",)))

    assert "к специалисту нашей клиники: терапевт." in answer.text


def test_urgent_and_missing_advice_keep_their_templates():
    assert renderer.render_medical_advice(S.SpecialistAdvice(urgent=True)) == renderer.render_urgent()
    template = renderer.render_medical_advice()
    assert renderer.render_medical_advice(None) == template
    assert template.handoff is True


# --- сквозной путь: метка от LLM → совет -------------------------------------------


def _route_with_advice(monkeypatch, text: str) -> tuple[str, bool, list]:
    """Полный путь пациента; NLU даёт медвопрос от LLM, совет записывает, с чем его звали."""

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="MEDICAL_ADVICE", confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    calls: list = []

    async def fake_advise(said, services):
        calls.append((said, services))
        return S.SpecialistAdvice(specialties=("невролог", "терапевт")) if services is not None else None

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )
    monkeypatch.setattr(S, "advise", fake_advise)

    async def go():
        state, memory = SessionState(session_id="symptom-advice"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    answer, handoff = asyncio.run(go())
    return answer, handoff, calls


@pytest.mark.parametrize(
    "text",
    ["у меня болит голова 3 дня", "что со мной? болит горло и температура"],
    ids=["label_from_llm", "label_from_rule_short_circuit"],
)
def test_complaint_gets_clinic_doctors_whoever_labels_it(monkeypatch, text):
    # Медвопрос по правилу («что со мной») отвечается раньше NLU, по короткому пути; 05.10
    # там не передавались сервисы, и совет молча откатывался на общий шаблон.
    answer, handoff, calls = _route_with_advice(monkeypatch, text)

    assert [said for said, _ in calls] == [text]
    assert all(services is not None for _, services in calls)
    assert "невролог, терапевт" in answer
    assert "По техническим вопросам" not in answer
    assert handoff is False
