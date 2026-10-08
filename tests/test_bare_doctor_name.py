"""Голая фамилия врача первым сообщением — карточка врача, а не «нет информации» (08.10).

Живой пациент 08.10: «Бородина» → «Извините, но у меня нет информации о враче с фамилией
Бородина…». Проба прода: LLM-классификатор ставит голой фамилии OTHER (0,9), сущность врача
для OTHER выбрасывается, свободная генерация сочиняет отказ — а врач в справочнике МИС есть и
со слотами. С контекстом («хочу к Бородиной», после кнопки) — карточка и расписание.

Инвариант класса: сообщение только из ФИО врача справочника (фамилия, со строчной, с именем,
отчеством или инициалами) даёт карточку этого врача. Совпадение — точное, по словам ФИО одного
врача: нечёткий `resolve_doctor_name` разбирает «хорошо» как «Хорошун».
"""

import asyncio
import json
from pathlib import Path

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.services import Services

_APIDATA = Path(__file__).resolve().parents[1] / "agent_logic_2" / "nayka_api" / "apidata"


def _doctor_fios() -> list[str]:
    snapshots = sorted(_APIDATA.glob("doctors_*.jsonl"))
    if not snapshots:
        pytest.skip("нет локального среза врачей (CLAUDE.md, ловушки)")
    with snapshots[-1].open(encoding="utf-8") as fh:
        return sorted({str(json.loads(line).get("fio") or "").strip() for line in fh} - {""})


@pytest.fixture
def services_llm_says_other(monkeypatch):
    # Как на проде: голую фамилию LLM относит к OTHER, сущность врача не даёт.
    async def nlu_other(*, user_text, state, services, memory, runtime_options):
        return RouteDecision(label="OTHER", confidence=0.9, source="llm_primary", flags={"rule_none"}), {}

    monkeypatch.setattr(router_mod, "_resolve_nlu_decision_before_doctor_guard", nlu_other)
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    services.doctors_schedule_week = _no_schedule  # расписание МИС в тестах недоступно — карточки хватит
    return services


async def _no_schedule(*args, **kwargs):
    return {}


def _decisions(services: Services, texts: list[str]) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []
    original = router_mod._verify_decision_and_prefetch_catalog

    async def spy(**kwargs):
        decision, prefetch = await original(**kwargs)
        seen.append((decision.label, dict(decision.entities)))
        return decision, prefetch

    router_mod._verify_decision_and_prefetch_catalog = spy
    try:
        async def run():
            for i, text in enumerate(texts):
                state = SessionState(session_id=f"bare-doctor-{i}")
                async for _ in router_mod.patient_routing_stream(text, state, services, MemoryStore()):
                    pass

        asyncio.run(run())
    finally:
        router_mod._verify_decision_and_prefetch_catalog = original
    return seen


def test_every_catalog_surname_alone_finds_its_doctor(services_llm_says_other):
    surnames = sorted({fio.split()[0] for fio in _doctor_fios()})
    decisions = _decisions(services_llm_says_other, surnames)
    missed = [s for s, (label, ent) in zip(surnames, decisions) if label == "OTHER" or ent.get("doctor_name") != s]
    assert not missed, missed


def test_name_forms_find_the_doctor(services_llm_says_other):
    fio = next(f for f in _doctor_fios() if len(f.split()) == 3)
    surname, name, patronymic = fio.split()
    texts = [surname.lower(), f"{surname} {name}", fio, f"{surname} {name[0]}.{patronymic[0]}.", f"{surname}?"]
    decisions = _decisions(services_llm_says_other, texts)
    assert [ent.get("doctor_name") for _, ent in decisions] == [surname] * len(texts), list(zip(texts, decisions))


@pytest.mark.parametrize("text", ["хорошо", "спасибо", "привет", "болит животов", "на мостовой", "Иванов Иван"])
def test_ordinary_words_are_not_a_doctor(services_llm_says_other, text):
    # «хорошо» нечётко похоже на фамилию врача; «животов», «мостовой» — фамилии врачей и обычные
    # слова (разбор переписки 08.10); «Иванов Иван» — фамилия есть, имени у этого врача нет.
    [(label, entities)] = _decisions(services_llm_says_other, [text])
    assert label == "OTHER" and not entities.get("doctor_name"), (label, entities)


@pytest.mark.parametrize("action", ["cancel", "reschedule"])
def test_surname_inside_cancel_or_reschedule_is_not_a_new_doctor_card(services_llm_says_other, action):
    # Разбор переписки 08.10: 4 из 13 голых фамилий шли после «отменить / проверить запись» —
    # там решение владельца 08.10 — оператор, а не карточка врача.
    surname = _doctor_fios()[0].split()[0]
    seen: list = []
    original = router_mod._verify_decision_and_prefetch_catalog

    async def spy(**kwargs):
        decision, prefetch = await original(**kwargs)
        seen.append(decision.label)
        return decision, prefetch

    router_mod._verify_decision_and_prefetch_catalog = spy
    try:
        state = SessionState(session_id=f"bare-doctor-{action}")
        state.last_entities["appointment_action"] = action

        async def run():
            async for _ in router_mod.patient_routing_stream(surname, state, services_llm_says_other, MemoryStore()):
                pass

        asyncio.run(run())
    finally:
        router_mod._verify_decision_and_prefetch_catalog = original
    assert "DOCTOR_SCHEDULE" not in seen, seen
