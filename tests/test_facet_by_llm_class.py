"""Тип вопроса хода решает LLM — одно решение на ход (деплой 22, шаг 2, STEP-2026-10-30-FACET-BY-LLM).

Было: `TurnUnderstanding.facet` ставили пять детекторов по буквам (`price_turn_facet`), обзор названия звал их напрямую, П1 спрашивал
LLM своим вопросом, а переклейки после цены и подготовки (`flow_price_followup_prepare/address`, `flow_prepare_followup_override`)
решали тип вопроса своими детекторами. Буквы ошибаются в обе стороны: «за какое время делают глюкозу?» — не срок (ответ ценой),
«сколько стоит подготовка к колоноскопии» — подготовка (памятка вместо цены), «где сдать ферритин» и «а без записи можно?» после
цены — ни одно.

Инвариант класса под `facet_by_llm = on`:
- тип вопроса — решение LLM (задача `turn_facet`), один вызов на ход: П1 и другие читатели берут его, а не спрашивают заново;
- срок — ответ сроком, не ценой; цена — ценой, даже если буквы видят «подготовку»; после цены «как подготовиться» — памятка,
  «где / без записи» — адреса; после подготовки голое название — подготовка; голое название — обзор, вопрос о цене — не обзор;
- сбой LLM — как было (детекторы), `tu_facet_src = fail`; ход нажатия кнопки без слов — LLM не зовётся;
- off — как было: вопрос `turn_facet` не задаётся, поля `tu_facet_src` нет.
Через обработчик эндпоинта; LLM — подделка клиента Ollama, решение проходит настоящий `choose()` (тип — `Choice`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура
from tests.test_price_turns_golden import FIXTURE as _MINI_FIXTURE
from tests.test_service_answer_class import _REGIONS

_FACET_MARK = "что именно пациент хочет узнать об анализе, исследовании, приёме врача"
_P1_MARK = "Ты помогаешь чат-боту медицинской клиники понять, что именно пациент хочет узнать об анализе или услуге, которую назвал"
_WALKIN_MARK = "хочет ли пациент сдать анализы или сделать ЭКГ"
_OVERVIEW_MARK = "прислал ли пациент только название услуги"
# Номера вариантов задачи `turn_facet` (`turn_understanding.FACET_OPTIONS`).
_AFTER_PRICE = {"_last_label": "PRICE", "service_name": "Глюкоза"}
_AFTER_PREPARE = {"_last_label": "PREPARE", "service_name": "Глюкоза"}
_FACET_NO = {"price": 1, "availability": 2, "where": 3, "booking": 4, "readiness": 5, "prepare": 6, "bare": 7}


_ECG_ROW = {"serviceId": 990101, "priceUnitId": 291, "serviceHomecode": "50.1", "serviceName": "ЭКГ", "deadline": " ", "cost": 800.0, "promotion": 0}


@pytest.fixture(scope="module")
def ecg_world(tmp_path_factory) -> Path:
    """Мини-прайс прода + строка ЭКГ (в выдержке её нет): «можно ли сделать ЭКГ без записи» — услуга каталога."""

    source = tmp_path_factory.mktemp("facet_src")
    for sub in ("price_by_region", "price_units"):
        (source / sub).mkdir()
        for path in (_MINI_FIXTURE / sub).iterdir():
            text = path.read_text(encoding="utf-8")
            if sub == "price_by_region":
                text = text.rstrip("\n") + "\n" + json.dumps(_ECG_ROW, ensure_ascii=False) + "\n"
            (source / sub / path.name).write_text(text, encoding="utf-8")
    return ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("facet_world") / "world")


def _classifier(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _facet(value: str | Callable[[str], str]) -> str | Callable[[str], str]:
    if callable(value):
        return value
    return json.dumps({"picks": [] if value == "none" else [_FACET_NO[value]]})


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


def _dialog(
    world_dir,
    monkeypatch,
    turns: list[tuple[str, str, Any]],
    *,
    switch: str,
    cfg_extra: dict[str, Any] | None = None,
    extra_answers: dict[str, Any] | None = None,
    seed: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], _LLM]:
    """Диалог через эндпоинт: на каждом ходе (реплика, метка классификатора, ответ задачи `turn_facet`); `seed` — состояние
    сессии до первого хода (прошлый ответ: тема и услуга)."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    now: dict[str, Any] = {}
    answers: dict[str, Any] = {
        _CLASSIFIER_MARK: lambda _p: _classifier(now["label"]),
        _FACET_MARK: lambda p: (now["facet"](p) if callable(now["facet"]) else now["facet"]),
        **(extra_answers or {}),
    }
    fake = _LLM(answers)
    session = f"facet-{switch}-{turns[0][0]}"

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_FACET_BY_LLM", switch, raising=False)
        for key, value in (cfg_extra or {}).items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)

        async def regions(_self):
            return list(_REGIONS)

        patch.setattr(Services, "_ensure_regions_loaded", regions)  # ответ адресами — по синтетике живого /regions
        choice._CACHE.clear()

        async def run() -> list[dict[str, Any]]:
            memory, services = MemoryStore(), Services()
            memory.get(session).last_entities.update(seed or {})
            out = []
            for text, label, facet in turns:
                now.update(label=label, facet=_facet(facet))
                before = len(fake.calls)
                req = MessengerGenerateRequest(session_id=session, text=text, debug=True)
                body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
                dbg = (body.get("state_update") or {}).get("debug") or {}
                tel = world.take_telemetry() or {}
                out.append(
                    {
                        "text": str(body.get("text") or ""),
                        "telemetry": tel,
                        "flags": (dbg.get("decision") or {}).get("flags") or [],
                        "calls": fake.calls[before:],
                    }
                )
            return out

        return asyncio.run(run()), fake


def _relabels(turn: dict[str, Any]) -> list[str]:
    return list(turn["telemetry"].get("tu_relabel_src") or [])


# --- off — как было ------------------------------------------------------------------------------------------------------

_OFF_DIALOGS = [
    [("когда будет готов анализ на глюкозу", "PRICE", "readiness")],
    [("а как к нему подготовиться?", "OTHER", "prepare")],
    [("есть ли у вас анализ на ферритин", "TEST_ASSIST", "availability")],
]


@pytest.mark.parametrize("turns", _OFF_DIALOGS, ids=lambda t: t[-1][0])
def test_off_never_asks_the_turn_facet(turns, mini_prod_world, monkeypatch):  # noqa: F811
    out, fake = _dialog(
        mini_prod_world, monkeypatch, turns, switch="off", seed=_AFTER_PRICE, cfg_extra={"MR_TOPIC_BY_LLM": "rule_test_assist"}
    )
    assert _FACET_MARK not in fake.calls
    assert all("tu_facet_src" not in t["telemetry"] for t in out), [t["telemetry"] for t in out]


# --- on: один вопрос на ход, тип вопроса — от LLM -------------------------------------------------------------------------


def test_readiness_the_letters_miss_is_answered_as_readiness(mini_prod_world, monkeypatch):  # noqa: F811
    # Буквы: «за какое время делают» — не срок (`price_turn_facet` → None) — ответ ценой. LLM: срок.
    out, _ = _dialog(mini_prod_world, monkeypatch, [("за какое время делают глюкозу?", "PRICE", "readiness")], switch="on")
    tel = out[0]["telemetry"]
    assert tel.get("tu_facet") == "readiness" and tel.get("tu_facet_src") == "llm", tel
    assert {"readiness_from_mis", "readiness_to_operator"} & set(tel.get("tu_acted") or []), tel
    assert "руб" not in out[0]["text"], out[0]["text"]


def test_readiness_question_is_not_a_price(mini_prod_world, monkeypatch):  # noqa: F811
    out, fake = _dialog(mini_prod_world, monkeypatch, [("когда будет готов анализ на глюкозу", "PRICE", "readiness")], switch="on")
    assert out[0]["calls"].count(_FACET_MARK) == 1
    assert "руб" not in out[0]["text"], out[0]["text"]


def test_price_wins_over_the_prepare_letters(mini_prod_world, monkeypatch):  # noqa: F811
    # Буквы видят «подготовку» в «сколько стоит подготовка…»: после цены переклейка увела бы в памятку. LLM: стоимость.
    turns = [("сколько стоит подготовка к нему?", "PRICE", "price")]
    off, _ = _dialog(mini_prod_world, monkeypatch, turns, switch="off", seed=_AFTER_PRICE)
    on, _ = _dialog(mini_prod_world, monkeypatch, turns, switch="on", seed=_AFTER_PRICE)
    assert any(s.startswith("flow_price_followup_prepare:") for s in _relabels(off[0])), off[0]["telemetry"]
    assert not any(s.startswith("flow_price_followup_prepare:") for s in _relabels(on[0])), on[0]["telemetry"]
    assert on[0]["telemetry"].get("tu_facet") == "price", on[0]["telemetry"]


def test_prepare_after_price_goes_to_the_memo(mini_prod_world, monkeypatch):  # noqa: F811
    turns = [("а что нельзя делать перед ним?", "OTHER", "prepare")]
    out, _ = _dialog(mini_prod_world, monkeypatch, turns, switch="on", seed=_AFTER_PRICE)
    tel = out[0]["telemetry"]
    assert "flow_price_followup_prepare:OTHER>PREPARE" in _relabels(out[0]), tel
    assert "followup_prepare_turn_facet" in (tel.get("tu_acted") or []), tel


@pytest.mark.parametrize("reply,facet", [("а где его сдать?", "where"), ("а без записи можно?", "booking")])
def test_where_or_booking_after_price_goes_to_addresses(reply, facet, mini_prod_world, monkeypatch):  # noqa: F811
    # «а без записи можно?» буквы не видят вовсе (ни регэксп «где/адрес/сдать», ни детектор): ответ был бы не о записи.
    walkin = json.dumps({"picks": [1], "outcome": "none"})  # LLM по правилу клиники: анализы — без записи
    out, _ = _dialog(
        mini_prod_world, monkeypatch, [(reply, "OTHER", facet)], switch="on", seed=_AFTER_PRICE,
        cfg_extra={"MR_WALKIN_BY_LLM": "on"}, extra_answers={_WALKIN_MARK: walkin},
    )
    tel = out[0]["telemetry"]
    assert tel.get("tu_facet") == facet, tel
    assert any(s.startswith("flow_price_followup_address:") for s in _relabels(out[0])), tel
    assert tel.get("tu_label_final") == "ADDRESS", tel


def test_walkin_class_still_decides_by_appointment_service(mini_prod_world, monkeypatch):  # noqa: F811
    # «Где» спросили, но услуга — по записи (LLM `walkin`: booking): адресов «без записи» нет — как было с подтверждением LLM.
    walkin = json.dumps({"picks": [], "outcome": "booking"})
    out, _ = _dialog(
        mini_prod_world, monkeypatch, [("а где его сдать?", "OTHER", "where")], switch="on", seed=_AFTER_PRICE,
        cfg_extra={"MR_WALKIN_BY_LLM": "on"}, extra_answers={_WALKIN_MARK: walkin},
    )
    assert not any(s.startswith("flow_price_followup_address:") for s in _relabels(out[0])), out[0]["telemetry"]


def test_bare_name_after_prepare_stays_in_prepare(mini_prod_world, monkeypatch):  # noqa: F811
    out, _ = _dialog(mini_prod_world, monkeypatch, [("а ТТГ", "OTHER", "bare")], switch="on", seed=_AFTER_PREPARE)
    assert "flow_prepare_followup_override:OTHER>PREPARE" in _relabels(out[0]), out[0]["telemetry"]


def test_price_question_after_prepare_leaves_prepare(mini_prod_world, monkeypatch):  # noqa: F811
    # «а по деньгам?» буквы цены не видят — короткая реплика после подготовки осталась бы в памятке.
    turns = [("а по деньгам?", "OTHER", "price")]
    off, _ = _dialog(mini_prod_world, monkeypatch, turns, switch="off", seed=_AFTER_PREPARE)
    on, _ = _dialog(mini_prod_world, monkeypatch, turns, switch="on", seed=_AFTER_PREPARE)
    assert "flow_prepare_followup_override:OTHER>PREPARE" in _relabels(off[0]), off[0]["telemetry"]
    assert not any(s.startswith("flow_prepare_followup_override:") for s in _relabels(on[0])), on[0]["telemetry"]


@pytest.mark.parametrize("facet,shown", [("availability", "TEST_ASSIST"), ("where", "ADDRESS"), ("booking", "ADDRESS")])
def test_p1_reads_the_turn_decision_and_does_not_ask_again(facet, shown, mini_prod_world, monkeypatch):  # noqa: F811
    reply = "есть ли у вас анализ на ферритин"  # каталог находит услугу (П1); тип вопроса — ответ подделки LLM
    out, fake = _dialog(
        mini_prod_world, monkeypatch, [(reply, "TEST_ASSIST", facet)], switch="on", cfg_extra={"MR_TOPIC_BY_LLM": "rule_test_assist"}
    )
    tel = out[0]["telemetry"]
    assert out[0]["calls"].count(_FACET_MARK) == 1, out[0]["calls"]
    assert _P1_MARK not in fake.calls, fake.calls  # прежний вопрос П1 не задаётся: решение хода одно
    assert tel.get("tu_p1_facet") == ("where_when" if facet in {"where", "booking"} else facet), tel
    assert tel.get("tu_label_final") == shown, tel


def _ecg_turn(world, monkeypatch, label: str) -> dict[str, Any]:
    walkin = json.dumps({"picks": [2], "outcome": "none"})  # LLM по правилу клиники: без записи — ЭКГ
    out, _ = _dialog(
        world,
        monkeypatch,
        [("можно ли сделать ЭКГ без записи", label, "booking")],
        switch="on",
        cfg_extra={"MR_TOPIC_BY_LLM": "rule_nonbookable_walkin", "MR_WALKIN_BY_LLM": "on"},
        extra_answers={_WALKIN_MARK: walkin},
    )
    return out[0]


def test_ecg_without_booking_question_is_a_booking_question(ecg_world, monkeypatch):
    # «Запись нужна ли» — свой тип вопроса (в задаче П1 он был частью «где / когда»).
    tel = _ecg_turn(ecg_world, monkeypatch, "TEST_ASSIST")["telemetry"]
    assert tel.get("tu_facet") == "booking" and tel.get("tu_facet_src") == "llm", tel


def test_ecg_without_booking_on_address_topic_gets_the_walkin_answer(ecg_world, monkeypatch):
    out = _ecg_turn(ecg_world, monkeypatch, "ADDRESS")
    assert "без записи" in out["text"] and "руб" not in out["text"], out["text"]


@pytest.mark.xfail(
    strict=True,
    reason="известный дефект и базы: тема «подбор анализов» на вопросе «без записи» — воронка цели; П1 ждёт услугу каталога, "
    "а каталог на walk-in не ищет, `policy_nonbookable_walkin` при `topic_by_llm` молчит. Читатель TU.facet = booking — шаг 4 "
    "(тип вопроса без лишнего вызова на каждом ходе)",
)
def test_ecg_without_booking_on_test_assist_topic_gets_addresses(ecg_world, monkeypatch):
    out = _ecg_turn(ecg_world, monkeypatch, "TEST_ASSIST")
    assert out["telemetry"].get("tu_label_final") == "ADDRESS", out["telemetry"]


def test_list_named_without_a_question_stays_a_price(mini_prod_world, monkeypatch):  # noqa: F811
    # Список без вопроса: LLM «просто назвал» — для П1 это цена списка, как None у правил (иначе — воронка подбора).
    out, _ = _dialog(mini_prod_world, monkeypatch, [("ферритин, глюкоза, холестерин", "TEST_ASSIST", "bare")], switch="on")
    tel = out[0]["telemetry"]
    assert tel.get("tu_facet") == "bare", tel
    if (tel.get("tu_list") or 0) >= 2:
        assert tel.get("tu_label_final") == "PRICE", tel


# --- сбой и ход без слов -------------------------------------------------------------------------------------------------


def test_llm_failure_is_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    turns = [("когда будет готов анализ на глюкозу", "PRICE", _down)]
    out, _ = _dialog(mini_prod_world, monkeypatch, turns, switch="on")
    tel = out[0]["telemetry"]
    assert tel.get("tu_facet_src") == "fail" and tel.get("tu_facet") == "readiness", tel  # детекторы, как было
    assert tel.get("tu_facet_fail"), tel
    assert {"readiness_from_mis", "readiness_to_operator"} & set(tel.get("tu_acted") or []), tel


def test_prepare_followup_failure_falls_back_to_detectors(mini_prod_world, monkeypatch):  # noqa: F811
    out, _ = _dialog(mini_prod_world, monkeypatch, [("а как к нему подготовиться?", "TEST_ASSIST", _down)], switch="on", seed=_AFTER_PRICE)
    assert "flow_price_followup_prepare:TEST_ASSIST>PREPARE" in _relabels(out[0]), out[0]["telemetry"]
    assert out[0]["telemetry"].get("tu_facet_src") == "fail", out[0]["telemetry"]


def test_turn_without_patient_words_does_not_ask(monkeypatch):
    from messengers_router import turn_understanding as tu
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_FACET_BY_LLM", "on", raising=False)
    fake = _LLM({_FACET_MARK: _facet("price")})
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)

    async def run() -> tuple[Any, Any]:
        tu.begin_turn()
        return await tu.turn_facet(""), await tu.llm_facet("")

    decided, by_turn = asyncio.run(run())
    assert decided.src == "rules" and by_turn is None
    assert _FACET_MARK not in fake.calls


def test_one_decision_per_turn_for_every_reader(monkeypatch):
    from messengers_router import turn_understanding as tu
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_FACET_BY_LLM", "on", raising=False)
    fake = _LLM({_FACET_MARK: _facet("where")})
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    choice._CACHE.clear()

    async def run() -> list[str | None]:
        tu.begin_turn()
        first = tu.start_facet("где сдать ферритин")
        return [await tu.llm_facet("где сдать ферритин"), await tu.llm_facet("где сдать ферритин"), (await first).value]

    assert asyncio.run(run()) == ["where", "where", "where"]
    assert fake.calls.count(_FACET_MARK) == 1


# --- обзор голого названия ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reply,facet,asked", [("ТТГ", "bare", True), ("ТТГ во что обойдётся?", "price", False)])
def test_overview_reads_the_turn_type(reply, facet, asked, mini_prod_world, monkeypatch):  # noqa: F811
    # «во что обойдётся» буквы цены не видят — обзор голого названия спрашивал бы LLM о вопросе о цене.
    overview = json.dumps({"picks": [], "outcome": "none"})
    out, fake = _dialog(
        mini_prod_world,
        monkeypatch,
        [(reply, "ADDRESS", facet)],
        switch="on",
        cfg_extra={"MR_BARE_NAME_OVERVIEW": "on"},
        extra_answers={_OVERVIEW_MARK: overview},
    )
    assert out[0]["telemetry"].get("tu_facet") == facet, out[0]["telemetry"]
    assert (_OVERVIEW_MARK in fake.calls) is asked, fake.calls


def _facet_under_budget(monkeypatch, *, calls: int, seconds: float, spent: int) -> tuple[Any, str | None, dict[str, Any], int]:
    from messengers_router import turn_telemetry
    from messengers_router import turn_understanding as tu
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_FACET_BY_LLM", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_LLM_DECISION_TURN_CALLS", calls, raising=False)
    monkeypatch.setattr(cfg, "MR_LLM_DECISION_TURN_SECONDS", seconds, raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)
    monkeypatch.setattr(turn_telemetry, "enabled", lambda: True)
    fake = _LLM({_FACET_MARK: _facet("price")})
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    choice._CACHE.clear()

    async def run() -> tuple[Any, str | None, dict[str, Any], int]:
        token = llm_runtime.begin_turn_budget()
        turn_telemetry.begin("test", session_id="t-budget")
        try:
            tu.begin_turn()
            llm_runtime._TURN_BUDGET.get().calls = spent  # решения хода, уже принятые до вопроса о типе
            text = "когда будет готов анализ на глюкозу"
            decided, by_turn = await tu.turn_facet(text), await tu.llm_facet(text)
            return decided, by_turn, dict(turn_telemetry._TURN.get().fields), llm_runtime.turn_decisions_spent()
        finally:
            llm_runtime.end_turn_budget(token)

    return asyncio.run(run())


def test_turn_type_does_not_spend_the_turn_decision_count(monkeypatch):
    # Тип вопроса — понимание хода, как вопрос рамки кнопки и ответ на висящий вопрос: число решений хода не тратит и не
    # упирается в него — иначе на ходе цены (классификатор, разметка, выбор строки) он отнимал место у проверки строки С3
    # (стенд 30.10, лимит 4: отказов `budget` 9 → 41, почти все — `service_answer_exact`).
    decided, by_turn, fields, spent = _facet_under_budget(monkeypatch, calls=1, seconds=25, spent=1)
    assert (decided.value, decided.src, by_turn) == ("price", "llm", "price"), fields
    assert spent == 1


def test_turn_seconds_limit_is_a_failure_with_rules_type(monkeypatch):
    # Секунды хода — общие с решениями: исчерпаны — тип вопроса правил, источник `fail`, причина `budget`; читатели — как было.
    decided, by_turn, fields, _ = _facet_under_budget(monkeypatch, calls=4, seconds=0.5, spent=0)
    assert (decided.value, decided.src) == ("readiness", "fail") and by_turn is None
    assert fields.get("tu_facet_fail") == "budget" and fields.get("tu_facet_src") == "fail", fields
