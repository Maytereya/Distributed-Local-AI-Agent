"""Врачи по услуге без списков слов (DESIGN-2026-10-23-NO-STOPWORDS, этап 2, выключатель `doctors_by_parse`).

Инварианты класса (решения владельца 22.10; принципы 1–2 `docs/ARCHITECTURE_PRINCIPLES.md`):
- один смысл в разных словоформах — один набор врачей («цистоскопию», «цистоскопия», «цистоскопии»);
- что в реплике услуга — решает разбор LLM (`_turn_spans`), какая это услуга — данные МИС (название, синоним, код, то же название
  в другой словоформе) или выбор LLM из услуг с теми же словами; врачей услуги называет связь «врач ↔ код» в МИС;
- без LLM услугу не угадывать: сбой разбора, сбой выбора, «ни одна не подходит» — «уточните услугу или специальность» с оффером;
- услуга есть, врачей по ней в данных нет (или срез недоступен) — оффер оператора, «врачей нет» не утверждаем;
- врач другого города не появляется; роль приёма, названная словами МИС, сужает;
- путь `on` поиска врачей не читает пяти списков файлов врачей (`_SERVICE_FILTER_STOPWORDS`, `_PROCEDURE_GENERIC_TOKENS`,
  `_SERVICE_CODE_STOPWORDS`; `_CATALOG_SERVICE_STOPWORDS`, `_CATALOG_SERVICE_TRAILING_TIME_RE`) ни сам, ни через функции, которые
  их читают (AST + ловушки). Два последних читает подстановка каталога (`match_catalog_service`) — она под выключателем не меняется.
Журнал: STEP-2026-10-24-DOCTORS-BY-PARSE.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router import evidence_keys as ek
from messengers_router import llm_runtime
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.response_builder import (
    DOCTORS_BY_SERVICE_NOT_FOUND_OFFER,
    DOCTORS_SERVICE_CLARIFY_OFFER,
    build_doctor_info_response,
)
from messengers_router.services import Services
from messengers_router.services import _doctors_by_parse as P
from messengers_router.services import _doctors_helpers as H
from messengers_router.services import doctors as D

_ROOT = Path(__file__).resolve().parents[1]
_LISTS = (
    "_CATALOG_SERVICE_STOPWORDS",
    "_SERVICE_FILTER_STOPWORDS",
    "_PROCEDURE_GENERIC_TOKENS",
    "_CATALOG_SERVICE_TRAILING_TIME_RE",
    "_SERVICE_CODE_STOPWORDS",
)
_HERE, _AWAY = "пр.Ленина, 5", "г. Сызрань, ул. Кирова, 1"
_SURNAMES = {1: "Кузнечиков", 2: "Муравьёв", 3: "Стрекозин", 4: "Жуков", 5: "Сверчков", 6: "Комаров"}
_UNITS = {"Врач-уролог": 17, "Врач-хирург": 40, "Врач-эндоскопист": 30, "Врач-оториноларинголог": 55}


def _doctor(doctor_id: int, unit: str, text: str, region: str = _HERE) -> dict:
    return {
        "id": doctor_id,
        "fio": f"{_SURNAMES[doctor_id]} Тестовый Тестович",  # ПДн — выдуманные
        "specialization": text,
        "regions": [region],
        "units": [unit],
        "main_units": [unit],
        "region_ids": [101 if region == _HERE else 909],
        "unit_links": [{"company_unit_id": _UNITS[unit], "company_unit_name": unit, "main": True, "specialization": text}],
        "main_specializations": [text],
    }


_DOCTORS = [
    _doctor(1, "Врач-уролог", "Приём взрослых пациентов"),
    _doctor(2, "Врач-уролог", "Приём взрослых пациентов"),  # услуг в МИС не связан ни с одной
    _doctor(3, "Врач-хирург", "Амбулаторная хирургия"),
    _doctor(4, "Врач-эндоскопист", "• Колоноскопия\n• ФГДС"),
    _doctor(5, "Врач-оториноларинголог", "Заболевания уха, горла, носа"),
    _doctor(6, "Врач-уролог", "Приём взрослых пациентов", region=_AWAY),
]
_CYSTO, _COLO_SED, _COLO_DIAG = "Цистоскопия", "Колоноскопия под седацией", "Колоноскопия диагностическая"
_NOSE, _ECG = "Эндоскопия полости носа и носоглотки с видеозаписью", "Электрокардиограмма"
_PRICE = [
    {"serviceName": _CYSTO, "serviceHomecode": "4.2.1", "cost": 5000},
    {"serviceName": _COLO_SED, "serviceHomecode": "5.1.2", "cost": 9000},
    {"serviceName": _COLO_DIAG, "serviceHomecode": "5.1.1", "cost": 7000},
    {"serviceName": _NOSE, "serviceHomecode": "7.3.1", "cost": 3000},
    {"serviceName": _ECG, "serviceHomecode": "9.9.9", "cost": 600},  # услуга есть, врача в МИС у неё нет
]


def _dp(doctor_id: int, name: str, code: str, unit: int, branch: str = _HERE) -> dict:
    return {"doctorId": doctor_id, "serviceName": name, "serviceHomecode": code, "companyUnitId": unit, "regionName": branch}


_DOCTOR_PRICES = [
    _dp(1, _CYSTO, "4.2.1", 17),
    _dp(3, _CYSTO, "4.2.1", 40),  # цистоскопию делает и хирург — другая роль приёма
    _dp(6, _CYSTO, "4.2.1", 17, branch=_AWAY),  # врач другого города
    _dp(4, _COLO_SED, "5.1.2", 30),
    _dp(4, _COLO_DIAG, "5.1.1", 30),
    _dp(5, _NOSE, "7.3.1", 55),
]


class _LLM:
    """Разметка слов (`turn_spans`) — номер первого слова, начинающегося с основы; выбор (`choose_doctors_service`) — заданный."""

    def __init__(self, mark: str = "", picks: list[int] | None = None, fail: str = "") -> None:
        self.mark, self.picks, self.fail = mark, picks or [], fail
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> tuple[str, dict]:
        self.calls.append(kw)
        if self.fail and kw["site"] == self.fail:
            raise TimeoutError("LLM недоступна")
        if kw["site"] == "turn_spans":
            words = kw["user"].split("Слова сообщения с номерами:")[-1].split()
            hits = [int(w.split(":", 1)[0]) for w in words if ":" in w and self.mark and w.split(":", 1)[1].lower().startswith(self.mark)]
            return json.dumps({"phrases": [[hits[0], hits[-1]]] if hits else []}), {}
        return json.dumps({"picks": self.picks}), {}

    def options(self) -> list[str]:
        seen = [kw for kw in self.calls if kw["site"] == "choose_doctors_service"]
        return [line.split(". ", 1)[1] for line in seen[-1]["user"].splitlines() if line[:1].isdigit()] if seen else []


@pytest.fixture
def services(monkeypatch):
    svc = Services()
    svc.ensure_background_refresh_started = lambda: None

    async def doctors():
        return [dict(d) for d in _DOCTORS]

    async def tokens():
        return {"пр ленина 5", "ленина"}

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", doctors)
    monkeypatch.setattr(svc, "_samara_region_tokens", tokens)
    monkeypatch.setattr(api_price, "load_doctor_prices", lambda *a, **k: list(_DOCTOR_PRICES))
    monkeypatch.setattr(api_price, "load_price_by_region", lambda *a, **k: [dict(r) for r in _PRICE])
    monkeypatch.setattr(D._cfg, "MR_DOCTORS_BY_PARSE", "on", raising=False)
    return svc


def _ask(services, monkeypatch, query: str, llm: _LLM, entities: dict | None = None) -> tuple[list[int], dict]:
    monkeypatch.setattr(llm_runtime, "generate_structured", llm)
    payload = asyncio.run(services.doctors_info(query, dict(entities or {})))
    return sorted(d["id"] for d in payload["doctors"]), payload


@pytest.mark.parametrize(
    ("reply", "stem", "expected"),
    [
        ("Кто делает цистоскопию?", "цистоскоп", [1, 3]),
        ("кто делает цистоскопия", "цистоскоп", [1, 3]),
        ("Какие врачи делают цистоскопии?", "цистоскоп", [1, 3]),
        ("Цистоскопию у вас кто проводит?", "цистоскоп", [1, 3]),
        ("Кто делает колоноскопию?", "колоноскоп", [4]),
        ("какой врач делает колоноскопии", "колоноскоп", [4]),
    ],
)
def test_wordforms_give_one_doctor_set(services, monkeypatch, reply, stem, expected) -> None:
    llm = _LLM(mark=stem, picks=[1, 2])
    ids, payload = _ask(services, monkeypatch, reply, llm)

    assert ids == expected  # врач другого города (6) по коду не появляется; уролог 2 услугу не делает
    assert payload["doctors_source"] == "parse_data"


def test_weak_phrase_service_is_chosen_by_llm_from_mis_names_only(services, monkeypatch) -> None:
    llm = _LLM(mark="колоноскоп", picks=[1, 2])
    _ask(services, monkeypatch, "Кто делает колоноскопию?", llm)

    # Названия «колоноскопия» в прайсе нет — LLM выбирает из услуг МИС с теми же словами; чужих (эндоскопия носа) в вариантах нет.
    assert sorted(llm.options()) == sorted([_COLO_DIAG, _COLO_SED])


def test_pool_sees_compound_names_and_one_extra_word_in_long_phrases() -> None:
    """Пул — генератор вариантов, не ответ: «колоноскопию» видит «Фиброколоноскопию», «удаляет серную пробку» — «Удаление
    серной пробки»; во фразе из двух слов нужны оба («эндоскопию желудка» — не «Эндоскопия полости носа»)."""

    from messengers_router.services._service_doctors import build_service_doctor_index

    names = ["Тотальная фиброколоноскопия (ФКС)", "Удаление серной пробки", _NOSE, _CYSTO, "Временная пластмассовая коронка",
             "Пневмомассаж барабанной перепонки", "Массаж предстательной железы (1 сеанс)"]
    index = build_service_doctor_index([_dp(1, name, str(k), 17) for k, name in enumerate(names)], in_region=lambda _row: True)

    def pool(phrase: str) -> list[str]:
        return [item.name for item in P.candidate_services((phrase,), index)]

    assert pool("колоноскопию") == ["Тотальная фиброколоноскопия (ФКС)"]
    assert pool("удаляет серную пробку") == ["Удаление серной пробки"]
    assert pool("эндоскопию желудка") == []
    assert pool("узи") == []  # короткая основа — только целое слово: «узи» не внутри «эндоскопии»
    assert pool("массаж") == ["Массаж предстательной железы (1 сеанс)"]  # «масс» — с начала слова, не внутри «пластмассовая»


def test_letters_alone_do_not_name_the_service(services, monkeypatch) -> None:
    """«Эндоскопия» есть только в «Эндоскопии полости носа» (свип L-06 21.10) — решает LLM: не узнала — «уточните», не ЛОР."""

    ids, payload = _ask(services, monkeypatch, "Кто делает эндоскопию?", _LLM(mark="эндоскоп", picks=[]))
    assert ids == [] and payload["service_clarify"] is True

    ids, _ = _ask(services, monkeypatch, "Кто делает эндоскопию носа?", _LLM(mark="эндоскоп", picks=[1]))
    assert ids == [5]


def test_procedure_role_of_mis_is_an_option_for_a_general_procedure(services, monkeypatch) -> None:
    """«Эндоскопия» желудка в МИС зовётся без этого слова — без роли пул только ЛОР (свип 24.10); роль «эндоскопист» — вариант
    из каталога ролей МИС, выбирает её LLM, а не правило «процедура → роль»."""

    llm = _LLM(mark="эндоскоп", picks=[2])
    ids, payload = _ask(services, monkeypatch, "Кто делает эндоскопию?", llm)

    assert llm.options() == [_NOSE, "Все услуги специалиста «эндоскопист»"]
    assert ids == [4] and payload["doctors_source"] == "parse_data"


@pytest.mark.parametrize("fail", ["turn_spans", "choose_doctors_service"])
def test_llm_failure_asks_to_clarify_instead_of_guessing(services, monkeypatch, fail) -> None:
    ids, payload = _ask(services, monkeypatch, "Кто делает колоноскопию?", _LLM(mark="колоноскоп", picks=[1], fail=fail))

    assert ids == [] and payload["service_clarify"] is True and payload["service_filter_emptied"] is False
    state, memory = SessionState(session_id="dbp-clarify"), MemoryStore()
    env = build_doctor_info_response("DOCTOR_INFO", Evidence(items={ek.DOCTORS_INFO: payload}), state, memory)
    assert env is not None and env.text == DOCTORS_SERVICE_CLARIFY_OFFER and env.handoff is False
    assert state.last_entities.get("_operator_offer_pending") is True


def test_named_role_answers_when_the_service_is_not_understood(services, monkeypatch) -> None:
    """Услугу не понять (сбой разбора), но пациент назвал роль — врачи роли, как на вопрос о роли; не «уточните» и не угадывание услуги."""

    llm = _LLM(mark="цистоскоп", fail="turn_spans")
    ids, payload = _ask(services, monkeypatch, "Кто из урологов делает цистоскопию?", llm, {"specialty": "уролог"})
    assert ids == [1, 2] and payload["doctors_source"] == "parse_role" and payload["service_clarify"] is False


def test_service_without_doctors_in_mis_offers_operator(services, monkeypatch) -> None:
    ids, payload = _ask(services, monkeypatch, "Кто делает электрокардиограмму?", _LLM(mark="электрокардиограм"))

    assert ids == [] and payload["service_filter_emptied"] is True  # «врачей нет» по пустым данным не утверждаем
    env = build_doctor_info_response("DOCTOR_INFO", Evidence(items={ek.DOCTORS_INFO: payload}), SessionState(session_id="dbp-ecg"), MemoryStore())
    assert env is not None and env.text == DOCTORS_BY_SERVICE_NOT_FOUND_OFFER.format(service="электрокардиограмму")


def test_doctor_slice_unavailable_offers_operator(services, monkeypatch) -> None:
    def broken(*_a, **_k):
        raise OSError("срез цен врачей недоступен")

    monkeypatch.setattr(api_price, "load_doctor_prices", broken)
    ids, payload = _ask(services, monkeypatch, "Кто делает цистоскопию?", _LLM(mark="цистоскоп"))
    assert ids == [] and payload["service_filter_emptied"] is True and payload["doctors_source"] == "parse_no_slice"


def test_role_named_in_mis_words_narrows_service_doctors(services, monkeypatch) -> None:
    ids, payload = _ask(services, monkeypatch, "Уролог делает цистоскопию?", _LLM(mark="цистоскоп"), {"specialty": "уролог"})
    assert ids == [1] and payload["doctors_source"] == "parse_data_role"


def test_service_from_context_when_reply_names_none(services, monkeypatch) -> None:
    llm = _LLM(mark="")  # разбор: в реплике услуги нет
    ids, payload = _ask(services, monkeypatch, "Напишите врачей по этой услуге", llm, {"service_name": _CYSTO})
    assert ids == [1, 3] and payload["doctors_source"] == "parse_data"

    ids, payload = _ask(services, monkeypatch, "Напишите врачей по этой услуге", _LLM(mark=""))
    assert ids == [] and payload["service_clarify"] is True  # ни услуги, ни роли — «уточните»


def test_named_doctor_is_answered_by_name_without_parse(services, monkeypatch) -> None:
    llm = _LLM(mark="цистоскоп")
    ids, _ = _ask(services, monkeypatch, "Кузнечиков делает цистоскопию?", llm, {"doctor_name": "Кузнечиков"})
    assert ids == [1] and llm.calls == []


def test_off_keeps_the_previous_path(services, monkeypatch) -> None:
    monkeypatch.setattr(D._cfg, "MR_DOCTORS_BY_PARSE", "off", raising=False)

    async def forbidden(*_a, **_k):
        raise AssertionError("off не ходит в путь разбора")

    monkeypatch.setattr(P, "doctors_info", forbidden)
    llm = _LLM(mark="цистоскоп")
    _ask(services, monkeypatch, "Кто делает цистоскопию?", llm, {"service_name": "цистоскопия"})
    assert all(kw["site"] != "turn_spans" for kw in llm.calls)


class _Tripwire:
    """Список-ловушка: любое чтение — ошибка теста."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"путь on прочитал список ({name})")

    def __contains__(self, item: Any) -> bool:
        raise AssertionError("путь on прочитал список (in)")

    def __iter__(self):
        raise AssertionError("путь on прочитал список (iter)")

    def __or__(self, other: Any) -> Any:
        raise AssertionError("путь on прочитал список (|)")


def test_on_path_reads_none_of_the_five_lists_at_runtime(services, monkeypatch) -> None:
    for module in (H, D):
        for name in _LISTS:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, _Tripwire())
    for reply, mark, picks in (("Кто делает цистоскопию?", "цистоскоп", []), ("Кто делает колоноскопию?", "колоноскоп", [1, 2])):
        ids, _ = _ask(services, monkeypatch, reply, _LLM(mark=mark, picks=picks))
        assert ids
    for reply in ("Кто делает эндоскопию?", "Напишите врачей по этой услуге"):  # «уточните» — тоже без списков
        ids, payload = _ask(services, monkeypatch, reply, _LLM(mark="эндоскоп"))
        assert ids == [] and payload["service_clarify"] is True


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def _readers() -> set[str]:
    """Функции врачей, которые читают пять списков — сами или через другие такие функции (замыкание по вызовам)."""

    funcs: dict[str, set[str]] = {}
    for rel in ("messengers_router/services/_doctors_helpers.py", "messengers_router/services/doctors.py"):
        for node in ast.parse((_ROOT / rel).read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs[node.name] = _names(node)
    readers = set(_LISTS)
    grew = True
    while grew:
        grew = False
        for name, used in funcs.items():
            if name not in readers and used & readers:
                readers.add(name)
                grew = True
    return readers


def test_on_module_does_not_reference_lists_or_their_readers() -> None:
    tree = ast.parse((_ROOT / "messengers_router/services/_doctors_by_parse.py").read_text(encoding="utf-8"))
    readers = _readers()
    assert {"_doctor_matches_service", "_service_catalog_query_candidates", "_service_name_tokens"} <= readers  # замыкание видит их
    assert not (_names(tree) & readers)


def test_prompt_lives_in_both_locations() -> None:
    bot = (_ROOT / "messengers_router/prompts/choose_doctors_service.txt").read_text(encoding="utf-8")
    prod = (_ROOT / "app_data/prompts/mr_choose_doctors_service.txt").read_text(encoding="utf-8")
    assert bot == prod and "<<DATA>>" in bot and "<<OPTIONS>>" in bot
