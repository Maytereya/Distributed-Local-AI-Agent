"""Рамку кнопки проверяет одна точка — таблица политик темы (Н-2, STEP-2026-10-30-FRAME-POLICY, консилиум 30.10).

Дефект (карта «Понимания» 30.10, П-2): тему, которую держит кнопка, после NLU переклеивали правила по буквам — их видели только
точечные флаги `frame_firm` (правило до NLU) и `topic_firm` (П1, walk-in). Д3: «Как подготовиться» → «а сколько стоит ТТГ?» →
судья ответа (LLM) «другая тема» → PRICE → `flow_prepare_followup_override` по `_last_label = PREPARE` снова PREPARE: LLM сказала
«другое», буквы вернули обратно.

Инвариант класса: под `frame_policy = on` строка таблицы `in_frame = keeps` не меняет решение LLM о рамке кнопки этого хода — не
уводит из темы, которую держит кнопка, и не возвращает в тему кнопки реплику, которую LLM признала другой темой; след
`tu_relabel_frame_kept`. Строки `overrides` (перевод, существующая и идущая запись, решения LLM того же хода) рамка не держит.
off — как было; ход без кнопки и вне рамки — как было (встречный свип: переклейки, верные вне рамки, не изменились). Через
обработчик эндпоинта, LLM — подделка настоящего типа; ключи кнопок — как у деплоя 21.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import button_flow, choice, llm_runtime, topic_policy, turn_telemetry
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_JUDGE_MARK = "пациент спрашивает о другом, но в теме нажатой кнопки"  # судья ответа, вариант Б (`button_frame_check = judge`)
_FREE_MARK = "Бот попросил назвать что-то своими словами"  # судья без кнопки — встречные «ход без кнопки»
_FRAME_MARK = "остаётся ли сообщение в теме нажатой кнопки"  # вопрос рамки листа без своего вопроса
# Ключи кнопок деплоя 21 (`release` = d6139b02).
_DEPLOY21 = {
    "MR_PENDING_BUTTONS": "on",
    "MR_CHOOSE_CACHE_TTL_S": 0,
    "MR_BUTTON_ANSWER_TOPIC": "frame",
    "MR_BUTTON_FRAME_CHECK": "judge",
    "MR_BUTTON_REPLY_JUDGE": "on",
    "MR_BUTTON_TOPIC_FIRM": "on",
    "MR_BUTTON_LEAF_FRAME": "menu.tests.booking_needed",
}


def _label(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _outcome(outcome: str) -> str:
    return json.dumps({"picks": [], "outcome": outcome})


def _dialog(
    world_dir,
    monkeypatch,
    turns: list[Any],
    *,
    policy: str,
    classifier: str = "PRICE",
    judge: str = "new_topic",
    frame: str | None = None,
    switches: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"frame-policy-{policy}-{classifier}-{judge}-{frame}-{json.dumps(turns, ensure_ascii=False)}-{switches}"
    answers: dict[str, Any] = {
        _JUDGE_MARK: _outcome(judge),
        _FREE_MARK: _outcome(judge),
        _CLASSIFIER_MARK: _label(classifier),
        "Сообщение пациента": json.dumps({"picks": []}),
    }
    if frame is not None:
        answers = {_FRAME_MARK: _outcome(frame), **answers}
    answers = {**(extra or {}), **answers}

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "handoff": bool(body.get("handoff")),
                    "label": decision.get("label"),
                    "telemetry": world.take_telemetry() or {},
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {**_DEPLOY21, **(switches or {}), "MR_FRAME_POLICY": policy}.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def _rows(tel: dict, key: str = "tu_relabel_src") -> list[str]:
    return [str(step).split(":", 1)[0] for step in tel.get(key) or []]


# --- перечень строк: решение по каждой — явное (поле без умолчания), `overrides` — ровно этот список -----------------------

_OVERRIDES = {
    # перевод на оператора
    "topic_handoff", "catalog_unavailable",
    # существующая запись
    "appointment_change_verb", "existing_appointment_operator_handoff", "existing_appointment_operator_offer",
    # идущая запись — путь к необратимому действию
    "flow_pending_answer", "flow_datetime_appointment_override", "flow_appointment_override", "flow_appointment_guard_override",
    "flow_datetime_appointment_prelock",
    # решение LLM по словам этого хода, не буквы (список разметки — как С6 28.10; ДМС / ОМС; подготовка против записи)
    "understanding_list_is_price", "insurance_topic", "prepare_over_booking_llm",
    # нужна ли запись на названную услугу — каталог МИС (рамка листа «Нужно ли записываться» так и задумана)
    "walkin_named_bookable_service",
    # шаг ответа в той же теме, не смена темы: уточнение вместо чужой цены, срок из МИС, выбор инструмента
    "catalog_confirm_requested", "readiness_reply", "plan_doctor_list",
}


def test_every_row_decides_the_frame_explicitly():
    assert {row.in_frame for row in topic_policy._ROWS} <= {topic_policy.KEEPS, topic_policy.OVERRIDES}
    assert {row.id for row in topic_policy._ROWS if row.in_frame == topic_policy.OVERRIDES} == _OVERRIDES
    with pytest.raises(TypeError):  # без `in_frame` строки не бывает
        topic_policy.Relabel("x", "layer", None, "site", "why")  # type: ignore[call-arg]


# --- одна проверка: `frame_kept` ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "policy", "before", "after", "kept"),
    [
        (button_flow.Frame("PREPARE", held=False), "flow_prepare_followup_override", "ADDRESS", "PREPARE", True),  # Д3
        (button_flow.Frame("TEST_ASSIST", held=True), "policy_nonbookable_walkin", "TEST_ASSIST", "ADDRESS", True),  # П-2
        (button_flow.Frame("TEST_ASSIST", held=True), "existing_appointment_operator_offer", "TEST_ASSIST", "OTHER", False),
        (button_flow.Frame("PREPARE", held=False), "flow_price_followup_address", "PRICE", "ADDRESS", False),  # не в тему кнопки
        (button_flow.Frame("PRICE", held=True), "verified_doctor_override", "OTHER", "DOCTOR_SCHEDULE", False),  # уже вне рамки
        (button_flow.Frame("PRICE", held=True), "flow_price_city_reply_override", "PRICE", "PRICE", False),  # метка та же
        (None, "flow_prepare_followup_override", "ADDRESS", "PREPARE", False),  # рамки нет
    ],
)
def test_frame_kept_is_one_check(frame, policy, before, after, kept, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    token = turn_telemetry.begin("test", session_id="frame-kept")
    monkeypatch.setattr(button_flow, "turn_frame", lambda: frame)
    monkeypatch.setattr(cfg, "MR_FRAME_POLICY", "on", raising=False)
    assert topic_policy.frame_kept(policy, before, after) is kept
    monkeypatch.setattr(cfg, "MR_FRAME_POLICY", "off", raising=False)
    assert topic_policy.frame_kept(policy, before, after) is False  # off — как было
    row = turn_telemetry.finish(token, handoff=False, chars=0) or {}
    assert row.get("tu_relabel_frame_kept") == ([f"{policy}:{before}>{after}"] if kept else None), row


# --- Д3: судья (LLM) сказал «другая тема» — буквы не возвращают в тему кнопки -----------------------------------------------

# Вопрос о другом сразу после нажатия «Как подготовиться к анализу»: правило короткой реплики после подготовки
# (`flow_prepare_followup_override`, `_last_label = PREPARE`) ловит его по буквам.
_D3 = [("а где сдать?", "ADDRESS"), ("а запишите меня", "APPOINTMENT"), ("расписание терапевта", "DOCTOR_SCHEDULE")]
_PREPARE = "menu.tests.prepare"


@pytest.mark.parametrize(("reply", "topic"), _D3)
def test_off_letters_return_the_new_topic_to_the_button_as_before(reply, topic, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), reply], policy="off", classifier=topic)[1]
    tel = turn["telemetry"]
    assert tel.get("tu_pending_outcome") == "new_topic" and tel.get("tu_label_nlu") == topic, tel
    assert "flow_prepare_followup_override" in _rows(tel) and turn["label"] == "PREPARE", turn


@pytest.mark.parametrize(("reply", "topic"), _D3)
def test_on_the_llm_new_topic_stands(reply, topic, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), reply], policy="on", classifier=topic)[1]
    tel = turn["telemetry"]
    assert "flow_prepare_followup_override" not in _rows(tel), tel
    assert _rows(tel, "tu_relabel_frame_kept") == ["flow_prepare_followup_override"], tel
    assert turn["label"] != "PREPARE", turn
    assert "К какому анализу или исследованию нужна подготовка" not in turn["text"], turn["text"]


@pytest.mark.parametrize(("reply", "topic"), _D3)
def test_rules_new_topic_is_not_a_frame(reply, topic, mini_prod_world, monkeypatch):  # noqa: F811
    """«Другая тема» без LLM (сбой судьи) — решения LLM о рамке нет: правило по буквам — как было."""

    turn = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), reply], policy="on", classifier=topic, judge="broken")[1]
    assert "tu_relabel_frame_kept" not in turn["telemetry"], turn["telemetry"]


# --- П-2: тема, которую держит кнопка, переклейки по буквам не уводят — та же проверка, что `topic_firm` ---------------------

_WALKIN_MARK = "хочет ли пациент сдать анализы или сделать ЭКГ"
_SUGGEST = "menu.tests.suggest"
# «Подобрать анализы»: детектор «без записи» (адреса) и П1 по совпадению с каталогом (цена) — STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM.
_HELD = [("подберите анализы, где сдать кровь ребенку", "policy_nonbookable_walkin"), ("анализ на ферритин", "test_assist_catalog_hit_is_price")]


def _held(world_dir, monkeypatch, reply: str, *, policy: str, firm: str) -> dict[str, Any]:
    switches = {"MR_BUTTON_TOPIC_FIRM": firm, "MR_WALKIN_BY_LLM": "on"}
    walkin = {_WALKIN_MARK: json.dumps({"picks": [1], "outcome": "none"})}  # «анализы — без записи»
    turns = [_press(_SUGGEST), reply]
    return _dialog(world_dir, monkeypatch, turns, policy=policy, classifier="TEST_ASSIST", judge="answer", switches=switches, extra=walkin)[1]


@pytest.mark.parametrize(("reply", "row"), _HELD)
def test_off_without_topic_firm_letters_take_the_button_topic_as_before(reply, row, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _held(mini_prod_world, monkeypatch, reply, policy="off", firm="off")
    assert turn["telemetry"].get("nlu_topic_by_llm") == "button_answer", turn["telemetry"]
    assert row in _rows(turn["telemetry"]) and turn["label"] != "TEST_ASSIST", turn


@pytest.mark.parametrize(("reply", "row"), _HELD)
def test_on_keeps_the_button_topic_like_topic_firm(reply, row, mini_prod_world, monkeypatch):  # noqa: F811
    policy = _held(mini_prod_world, monkeypatch, reply, policy="on", firm="off")
    firm = _held(mini_prod_world, monkeypatch, reply, policy="off", firm="on")
    assert _rows(policy["telemetry"], "tu_relabel_frame_kept") == [row], policy["telemetry"]
    assert policy["label"] == firm["label"] == "TEST_ASSIST", (policy, firm)
    assert policy["text"] == firm["text"]


# --- `overrides` рамка не держит: перевод, существующая запись, срочное и медвопрос — раньше таблицы -------------------------

_ALWAYS = [
    "хочу отменить свою запись на завтра",
    "переведите на оператора",
    "у меня сильная боль в груди и задыхаюсь",
    "что со мной, болит голова третий день",
]


@pytest.mark.parametrize("reply", _ALWAYS)
@pytest.mark.parametrize("judge", ["answer", "domain", "new_topic"])
def test_frame_never_holds_handoff_booking_or_safety(reply, judge, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), reply], policy="off", classifier="OTHER", judge=judge)[1]
    on = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), reply], policy="on", classifier="OTHER", judge=judge)[1]
    assert (on["text"], on["label"], on["handoff"]) == (off["text"], off["label"], off["handoff"])
    assert "tu_relabel_frame_kept" not in on["telemetry"], on["telemetry"]


_LEAF = "menu.tests.booking_needed"


@pytest.mark.parametrize("reply", ["Записаться на хгч", "Записаться на узи брюшной полости", "нужна ли запись к неврологу?"])
def test_leaf_frame_lets_the_catalog_decide_booking(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Рамка листа «Нужно ли записываться» держит тему ADDRESS, но нужна ли запись на названную услугу, решает каталог МИС
    (`walkin_named_bookable_service` — `overrides`): ход под on — как под off."""

    off = _dialog(mini_prod_world, monkeypatch, [_press(_LEAF), reply], policy="off", classifier="APPOINTMENT", frame="same")[1]
    on = _dialog(mini_prod_world, monkeypatch, [_press(_LEAF), reply], policy="on", classifier="APPOINTMENT", frame="same")[1]
    assert on["telemetry"].get("button_frame") == "same", on["telemetry"]
    assert (on["text"], on["label"]) == (off["text"], off["label"])
    assert on["telemetry"].get("tu_relabel_src") == off["telemetry"].get("tu_relabel_src")


# --- встречный свип: вне рамки переклейки — как были -----------------------------------------------------------------------

# Ход без кнопки (набранный вопрос о подготовке, затем вопрос о другом): правило короткой реплики работает, как задумано и как было.
_TYPED = [
    ["как подготовиться к узи брюшной полости", "а где сдать?"],
    ["как подготовиться к узи брюшной полости", "вульвоскопия"],
    ["сколько стоит ферритин", "а как к нему подготовиться?"],
    ["сколько стоит ферритин", "Самара"],
    ["есть ли у вас анализ на ферритин"],
    ["мне нужно к лору"],
    ["хочу записаться к кардиологу"],
    ["спасибо большое"],
]


@pytest.mark.parametrize("turns", _TYPED, ids=lambda t: " / ".join(t))
def test_relabels_outside_the_frame_are_untouched(turns, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, turns, policy="off", classifier="TEST_ASSIST")
    on = _dialog(mini_prod_world, monkeypatch, turns, policy="on", classifier="TEST_ASSIST")
    for a, b in zip(off, on):
        assert (b["text"], b["label"], b["handoff"]) == (a["text"], a["label"], a["handoff"])
        assert b["telemetry"].get("tu_relabel_src") == a["telemetry"].get("tu_relabel_src")
        assert "tu_relabel_frame_kept" not in b["telemetry"], b["telemetry"]


@pytest.mark.parametrize(("reply", "topic"), _D3)
def test_button_reply_after_the_frame_turn_is_untouched(reply, topic, mini_prod_world, monkeypatch):  # noqa: F811
    """Рамка — один ход: ответ на вопрос кнопки («ферритин») закрывает её, следующий ход — как было."""

    turns = [_press(_PREPARE), "ферритин", reply]
    off = _dialog(mini_prod_world, monkeypatch, turns, policy="off", classifier=topic, judge="answer")
    on = _dialog(mini_prod_world, monkeypatch, turns, policy="on", classifier=topic, judge="answer")
    assert (on[2]["text"], on[2]["label"]) == (off[2]["text"], off[2]["label"])
    assert "tu_relabel_frame_kept" not in on[2]["telemetry"], on[2]["telemetry"]
