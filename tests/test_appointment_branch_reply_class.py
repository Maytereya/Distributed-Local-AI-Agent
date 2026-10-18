"""Класс 5 («бот знает, чего ждёт»): ответ на вопрос бота «Какой филиал вам удобен?» — не смена темы.

BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH, найден стендом сбоев 17.10: «хочу записаться к урологу» → бот перечисляет
филиалы и спрашивает «Какой филиал вам удобен? Или написать список врачей?» → «в любой филиал» (или «какие адреса?»)
→ «Сейчас идет оформление записи. Отменить этот процесс и перейти к новому вопросу?».

Подбаг A — ИСПРАВЛЕН 18.10 по решению владельца: вопрос о филиале пишет `_expects` (`appointment_branch`, варианты —
адреса, «список врачей», «любой филиал»); предпроверка записи (`appointment_flow_guard._answers_branch_question`) его
читает. Ответ разбирает LLM (`pending_reply`, тип `appointment_branch` в `pending_reply_types` под `on`): «другая тема» —
вопрос об отмене, выбран адрес — филиал, «любой филиал» — передаётся оператору как есть, «список врачей» — список,
иное — запись идёт дальше. LLM не решала — запасной путь правил (сужение): ответ о филиалах, адресах, врачах или
показанный вариант — не смена темы; «где удобнее, на Ленина или на Победы?» правила не узнают — его разбирает LLM.

Подбаг B того же класса — ИСПРАВЛЕН 18.10: в записи на УСЛУГУ даже дословно названный филиал давал «Пока не понял, к
какому врачу или на какую услугу вас записать» (`entity_grounder._is_active_booking_target`). Инвариант: в активной
записи цель-услуга переживает ход-ответ на шаг записи так же, как специальность.

Мир — гейт (локальные срезы врачей и филиалов: в мини-проде филиалов нет); LLM — недоступна или оракул ответа на
висящий вопрос (`_BranchOracle`, отвечает только на промпт `choose_pending_reply`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from ollama import GenerateResponse

from messengers_router import llm_runtime
from messengers_router.policies import appointment_text_topic_switch_confirm

_BRANCH_QUESTION = "Какой филиал вам удобен?"
_UNKNOWN_TARGET = "Пока не понял, к какому врачу или на какую услугу вас записать"
_WISH_QUESTION = "Когда вам удобно прийти?"
_PENDING_MARK = "понять, что пациент ответил на вопрос"
_ANY_BRANCH = "любой филиал"

# Цель записи: специальность и услуга (у услуги своя ветка «цель не подтверждена» в планировщике — B′).
OPENINGS = {
    "specialty": "хочу записаться к урологу",
    "service": "запишите на Ультразвуковое исследование щитовидной железы",
}

# Ответы на вопрос «какой филиал» — о филиалах, адресах и врачах из того же вопроса — и что о них скажет LLM:
# («pick», вариант) или (исход, "").
BRANCH_REPLIES: dict[str, tuple[str, tuple[str, str]]] = {
    "any_branch": ("в любой филиал", ("pick", _ANY_BRANCH)),
    "no_matter": ("без разницы какой филиал", ("pick", _ANY_BRANCH)),
    "which_addresses": ("какие адреса?", ("none", "")),
    "closer": ("а какой филиал ближе к центру?", ("none", "")),
    "compare": ("где удобнее, на Ленина или на Победы?", ("none", "")),
    # Слова адреса — не «услуга»: раньше заземление делало из «Победы» услугу («Похоже, вы имели в виду …?», свип 18.10).
    "hours": ("до скольки работает филиал на Победы?", ("none", "")),
}
# Без LLM правила узнают ответ о филиале по своим детекторам; «compare» — только LLM.
RULES_REPLIES = ("any_branch", "no_matter", "which_addresses", "closer", "hours")


class _BranchOracle:
    """LLM отвечает только на разбор ответа на висящий вопрос (`choose_pending_reply`); прочие места — «недоступна».

    `answers`: реплика → («pick», текст варианта) | (исход, "")."""

    def __init__(self, answers: dict[str, tuple[str, str]]) -> None:
        self.answers = answers
        self.calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _PENDING_MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.calls += 1
        reply = next((line[len("Ответ пациента: «") : -1] for line in seen.splitlines() if line.startswith("Ответ пациента: «")), "")
        kind, value = self.answers.get(reply, ("none", ""))
        picks: list[int] = []
        if kind == "pick":
            for line in seen.splitlines():
                head, dot, rest = line.partition(". ")
                if dot and head.isdigit() and rest.strip() == value:
                    picks = [int(head)]
        answer = {"picks": picks, "outcome": "none" if kind == "pick" else kind}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


@pytest.fixture
def branch_llm(monkeypatch):
    """`pending_reply = on`, тип `appointment_branch` исполняется; LLM — оракул (ставит тест)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PENDING_REPLY", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_PENDING_REPLY_TYPES", "appointment_branch", raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)

    def install(answers: dict[str, tuple[str, str]]) -> _BranchOracle:
        oracle = _BranchOracle(answers)
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
        return oracle

    return install


def _dialog(turns: tuple[str, ...], session_id: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            entities = dict(memory.get(session_id).last_entities or {})
            out.append({"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff")), "entities": entities})
        return out

    return asyncio.run(run())


def _assert_booking_goes_on(first: dict[str, Any], second: dict[str, Any]) -> None:
    assert _BRANCH_QUESTION in first["text"], first["text"]
    assert second["text"] != appointment_text_topic_switch_confirm(), second["text"]
    assert not second["entities"].get("appointment_topic_switch_pending"), second["entities"]
    # Запись продолжается с той же целью: не «не понял, куда записать», не перевод.
    assert _UNKNOWN_TARGET not in second["text"], second["text"]
    assert not second["handoff"], second["text"]
    assert second["entities"].get("service_name") == first["entities"].get("service_name"), second["entities"]
    assert second["entities"].get("appointment_flow_active"), second["entities"]


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_opening_asks_for_a_branch_and_expects_the_answer(opening):
    first = _dialog((OPENINGS[opening],), f"branch-reply-open-{opening}")[0]
    assert _BRANCH_QUESTION in first["text"], first["text"]
    options = first["entities"].get("appointment_branch_options")
    assert options, first["entities"]
    expects = first["entities"].get("_expects") or {}
    assert expects.get("type") == "appointment_branch" and expects.get("kind") == "pick", expects
    assert expects["options"][: len(options)] == options and expects["options"][-1] == _ANY_BRANCH, expects


@pytest.mark.parametrize("reply", sorted(BRANCH_REPLIES))
@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_branch_reply_read_by_llm_is_not_a_topic_switch(opening, reply, branch_llm):
    text, answer = BRANCH_REPLIES[reply]
    oracle = branch_llm({text: answer})
    first, second = _dialog((OPENINGS[opening], text), f"branch-reply-llm-{opening}-{reply}")
    _assert_booking_goes_on(first, second)
    assert "имели в виду" not in second["text"], second["text"]
    assert oracle.calls == 1


@pytest.mark.parametrize("reply", RULES_REPLIES)
@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_branch_reply_without_llm_is_not_a_topic_switch(opening, reply):
    first, second = _dialog((OPENINGS[opening], BRANCH_REPLIES[reply][0]), f"branch-reply-down-{opening}-{reply}")
    _assert_booking_goes_on(first, second)
    assert "имели в виду" not in second["text"], second["text"]


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_any_branch_goes_to_the_operator_as_is(opening, branch_llm):
    branch_llm({"в любой филиал": ("pick", _ANY_BRANCH)})
    first, second, third = _dialog((OPENINGS[opening], "в любой филиал", "в субботу утром"), f"branch-any-{opening}")
    assert _WISH_QUESTION in second["text"], second["text"]
    assert second["entities"].get("branch_name") == _ANY_BRANCH, second["entities"]
    assert third["handoff"] and _ANY_BRANCH in third["text"] and "«в субботу утром»" in third["text"], third["text"]


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_llm_new_topic_still_asks_to_leave_the_booking(opening, branch_llm):
    branch_llm({"а сколько стоит мрт?": ("new_topic", "")})
    _, second = _dialog((OPENINGS[opening], "а сколько стоит мрт?"), f"branch-llm-new-topic-{opening}")
    assert second["text"] == appointment_text_topic_switch_confirm(), second["text"]


def test_llm_picked_address_is_the_branch(branch_llm):
    first = _dialog((OPENINGS["specialty"],), "branch-llm-pick-probe")[0]
    branch = first["entities"]["appointment_branch_options"][1]
    branch_llm({"давайте во второй": ("pick", branch)})
    _, second = _dialog((OPENINGS["specialty"], "давайте во второй"), "branch-llm-pick")
    assert _WISH_QUESTION in second["text"] and branch in second["text"], second["text"]


# Записи на услугу: цель — каноническое название каталога (у специальности — «приём к …»). Мир гейта без LLM: разговорное
# «узи щитовидки» с 18.10 — подтверждение или названия на выбор (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS), поэтому
# здесь, где проверяется ответ на вопрос о филиале, услуга названа дословно (сильное совпадение, без LLM).
SERVICE_OPENINGS = (
    "запишите на Ультразвуковое исследование щитовидной железы",
    "запишите на Ультразвуковое исследование почек и надпочечников",
)
# Ответы на шаг записи, которые новую цель не называют.
SLOT_ANSWERS = ("завтра после 17", "Иванов Иван")


def _named_branch_dialogs(opening: str) -> dict[str, list[dict[str, Any]]]:
    """Каждый показанный филиал, названный дословно, — второй ход; третий — пожелание по времени (сессия на филиал)."""

    first = _dialog((opening,), f"branch-named-probe-{opening}")[0]
    options = first["entities"].get("appointment_branch_options") or []
    assert options, first["text"]
    return {
        option: _dialog((opening, option, "в субботу утром"), f"branch-named-{opening}-{i}")
        for i, option in enumerate(options)
    }


@pytest.mark.parametrize("opening", (OPENINGS["specialty"], *SERVICE_OPENINGS))
def test_named_branch_moves_to_the_time_wish_and_the_operator(opening):
    for option, (first, second, third) in _named_branch_dialogs(opening).items():
        assert _WISH_QUESTION in second["text"], (option, second["text"])
        assert second["entities"].get("service_name") == first["entities"].get("service_name"), (option, second)
        assert third["handoff"], (option, third["text"])
        assert option in third["text"] and "«в субботу утром»" in third["text"], (option, third["text"])


@pytest.mark.parametrize("answer", SLOT_ANSWERS)
@pytest.mark.parametrize("opening", SERVICE_OPENINGS)
def test_slot_answer_keeps_the_service_target(opening, answer):
    # Паритет со специальностью: ответ без новой цели — запись продолжается с той же услугой, не «не понял, куда».
    first, second = _dialog((opening, answer), f"branch-slot-{opening}-{answer}")
    assert _UNKNOWN_TARGET not in second["text"], second["text"]
    assert not second["handoff"], second["text"]
    assert second["entities"].get("service_name") == first["entities"].get("service_name"), second["entities"]
    assert second["entities"].get("appointment_flow_active"), second["entities"]


@pytest.mark.parametrize("opening", (OPENINGS["specialty"], *SERVICE_OPENINGS))
def test_named_new_service_replaces_the_target(opening):
    # Встречный случай: правило держит только цель, которую реплика не меняет; новая услуга сменяет цель.
    first, second = _dialog((opening, "лучше на узи сердца"), f"branch-new-service-{opening}")
    assert second["entities"].get("service_name") != first["entities"].get("service_name"), second["entities"]
    assert "сердца" in second["text"], second["text"]


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_new_topic_still_asks_to_leave_the_booking(opening):
    _, second = _dialog((OPENINGS[opening], "какие у вас сейчас акции?"), f"branch-new-topic-{opening}")
    assert second["text"] == appointment_text_topic_switch_confirm(), second["text"]
