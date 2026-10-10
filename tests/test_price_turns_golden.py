"""Характеризация пути цены через эндпоинт — golden master перед ходом (б), шаг 1 (10.10).

Ход (б) переносит решение «какие услуги назвал пациент и о каком их свойстве спросил» в одно
место (`услуги[]` + `свойство`, L-14). Поверхность регрессий большая: 35 мест меняют решение
NLU после классификатора, у 13 из них нет ни записи журнала, ни теста (`~/Dev/handoffs/dla/
10_inventory_b.md`). Поэтому до переноса поведение пути цены фиксируется как есть — вместе с
известными дефектами (список сводится к одной услуге, срок готовности получает цену): golden
master ловит ЛЮБОЕ изменение, а намеренное меняется только перезаписью эталона с разбором
диффа.

Ход идёт через обработчик эндпоинта `messenger_generate_once` (precheck, кнопки, анти-повтор —
то, что обходит `route_patient_message`) в мире стенда на фикстуре «мини-прод» (404 строки
настоящего прайса, решение 08.10): день, LLM недоступна, сеть закрыта. На каждый ход
фиксируется: текст ответа, перевод на оператора, метка решения, инструменты плана,
`service_name_effective` инструмента, `last_entities.service_name` и висящий вопрос.

Чего здесь нет: цены у конкретного врача — в мини-прод нет врачей (ФИО в репозиторий не кладём);
живая LLM — это стенд (`evalbench`, режим `cached`).

Перезапись эталона после НАМЕРЕННОЙ правки: `GOLDEN_UPDATE=1 pytest tests/test_price_turns_golden.py`,
затем дифф `tests/golden/price_turns.json` — в коммит вместе с правкой и с объяснением.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from tests.test_button_actions import GATEWAY_V1_TITLES

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_prod" / "snapshots"
GOLDEN = Path(__file__).resolve().parent / "golden" / "price_turns.json"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"


def _button(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


# Ход — реплика или (кнопка, подпись). Каждый диалог — новая сессия.
DIALOGS: dict[str, list[str | tuple[str, str]]] = {
    # --- одна услуга
    "one_oak_words": ["сколько стоит общий анализ крови"],
    "one_oak_abbr": ["сколько стоит оак"],
    "one_ttg": ["сколько стоит ТТГ"],
    "one_ferritin_bare": ["ферритин"],
    "one_endocrinologist": ["сколько стоит приём эндокринолога"],
    "one_neurologist": ["сколько стоит прием невролога"],
    "one_gastroscopy": ["сколько стоит гастроскопия"],
    "one_pochem_bare": ["почем"],
    "one_pochem_named": ["почем узи брюшной полости"],
    "one_blood_draw": ["сколько стоит сдача крови"],
    "one_lipid_bare": ["липидограмма"],
    "one_crp_bare": ["С-реактивный белок"],
    # --- семейство, «все», синоним с несколькими услугами
    "family_glucose": ["сколько стоит глюкоза"],
    "family_vitamin_d_all": ["сколько стоит витамин D", "покажи все"],
    "synonym_fork_oak": ["сколько стоит оак"],
    # --- уточнение, которого в прайсе нет
    "qualifier_oms": ["сколько стоит приём терапевта по ОМС"],
    # --- УЗИ
    "uzi_generic": ["сколько стоит узи"],
    "uzi_compound": ["сколько стоит узи почек и щитовидной железы"],
    # --- висящий вопрос цены и ответ городом
    "pending_city_then_service": ["сколько стоит анализ", "в Самаре", "ферритин"],
    # --- после цены: подготовка и «где сдать»
    "after_price_prepare": ["сколько стоит ферритин", "а как подготовиться?"],
    "after_price_where": ["сколько стоит ферритин", "а где сдать?"],
    # --- подбор анализов и категория
    "category_checkup": ["чекап"],
    "assist_goal": ["какие анализы сдать для профилактики"],
    # --- список: текстом и ответом на кнопку (BUG-2026-10-05-PRICE-LIST-COLLAPSES-TO-ONE)
    "list_text_three": ["сколько стоит ферритин, глюкоза, холестерин"],
    "list_text_oak_oam": ["сколько стоит оак, оам"],
    "list_text_ttg_t4": ["сколько стоит ттг, т4 свободный"],
    "list_text_and": ["сколько стоит ферритин и глюкоза"],
    "list_text_no_commas": ["ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА"],
    "list_bare_two": ["ферритин, глюкоза"],
    "list_button_three": [_button("menu.price.test"), "ферритин, глюкоза, холестерин"],
    "list_button_oak_oam": [_button("menu.tests.price"), "оак, оам"],
    "list_button_ttg_t4": [_button("menu.price.test"), "ттг, т4 свободный"],
    # --- что остаётся после списка
    "after_list_prepare": ["сколько стоит ферритин, глюкоза", "а как подготовиться?"],
    "after_list_where": ["сколько стоит ферритин, глюкоза", "где сдать?"],
    "list_over_stale_context": ["сколько стоит ТТГ", "а ферритин, глюкоза?", "а как подготовиться?"],
    # --- кнопка цены и одна услуга / специальность
    "button_one_test": [_button("menu.price.test"), "ферритин"],
    "button_doctor_specialty": [_button("menu.price.doctor"), "невролог"],
    # --- срок готовности (BUG-2026-10-05-READINESS-AS-PRICE)
    "readiness_generic": ["сколько готовятся анализы"],
    "readiness_named": ["сколько готовится ферритин"],
    "readiness_when": ["когда будет готов анализ на ферритин"],
    # --- «нужна ли запись» про названную услугу
    "walkin_named": ["а на ферритин нужна запись?"],
    # --- название прайса с разделителем внутри — одна услуга, не корзина
    "name_with_plus": ["сколько стоит Глюкозотолерантный тест (глюкоза+ C-пептид)"],
    "name_with_commas": ["сколько стоит Общий анализ крови (Le, Er, Hb, СОЭ)"],
    "name_with_and": ["сколько стоит Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema pallidum)"],
}

# Синоним МИС на три услуги («оак» в живом словаре клиники). Гейт держит синонимы пустыми
# (`_hermetic_mis_synonyms`), поэтому словарь этого диалога задан явно.
_SYNONYM_FORK = {
    "оак": (
        "Общий анализ крови (Le, Er, Hb, СОЭ)",
        "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
        "Cito Общий анализ крови (Le, Er,Hb)",
    ),
}


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory) -> Path:
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


def _effective_service(evidence_items: dict[str, Any]) -> str | None:
    for payload in (evidence_items or {}).values():
        if isinstance(payload, dict):
            used = payload.get("entities_used")
            if isinstance(used, dict) and str(used.get("service_name_effective") or "").strip():
                return str(used["service_name_effective"])
    return None


def _turn_record(body: dict[str, Any], state_entities: dict[str, Any]) -> dict[str, Any]:
    dbg = (body.get("state_update") or {}).get("debug") or {}
    plan = dbg.get("plan") or {}
    pending = state_entities.get("_pending") if isinstance(state_entities.get("_pending"), dict) else None
    return {
        "text": body.get("text", ""),
        "handoff": bool(body.get("handoff")),
        "label": (dbg.get("decision") or {}).get("label"),
        "plan": [plan.get("label")] + [s.get("tool") for s in plan.get("steps") or []] if plan else None,
        "service_name_effective": _effective_service((dbg.get("evidence") or {}).get("items") or {}),
        "service_name": state_entities.get("service_name"),
        "pending": {"label": pending.get("label"), "missing": pending.get("missing")} if pending else None,
    }


def run_dialog(turns: list[str | tuple[str, str]], session_id: str) -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта в одной сессии и одном цикле событий."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True, button_id=button_id)
            resp = await messenger_generate_once(req, memory=memory, services=services)
            body = json.loads(resp.body)
            out.append(_turn_record(body, dict(memory.get(session_id).last_entities or {})))
        return out

    return asyncio.run(run())


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}


def _write_golden(case: str, record: list[dict[str, Any]]) -> None:
    data = _golden()
    data[case] = record
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


@pytest.fixture
def synonym_fork(monkeypatch):
    from messengers_router.services import _biomaterial as _bio

    hermetic = _bio._vocabularies

    def _with_fork():
        vocab = hermetic()
        vocab.synonym_candidates = dict(_SYNONYM_FORK)
        vocab.synonym_to_service = {}
        return vocab

    monkeypatch.setattr(_bio, "_vocabularies", _with_fork)


@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_price_path_matches_golden(case, mini_prod_world, request):
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        record = run_dialog(DIALOGS[case], f"golden-{case}")
        attempts = list(world.net.attempts)
    assert attempts == []
    for turn in record:
        assert turn["text"].strip(), (case, turn)
    if UPDATE:
        _write_golden(case, record)
        return
    expected = _golden().get(case)
    assert expected is not None, f"нет эталона для {case}: GOLDEN_UPDATE=1"
    assert record == expected


def test_golden_has_no_orphans():
    """Эталон не держит диалогов, которых больше нет в наборе (переименование не теряет кейс молча)."""

    assert set(_golden()) == set(DIALOGS)
