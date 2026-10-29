"""Харнесс контракта «понимание хода → поведение» (ход (б), деплой 22; консилиум 30.10, Q6: эталон — понимание, а не текст).

Тест задаёт, что LLM «поняла» на каждом ходе (`Understood`), — харнесс отвечает за неё на её местах, конвейер идёт через обработчик
эндпоинта (`messenger_generate_once`, как у шлюза) в мире стенда `d21_world` («мини-прод» + выдержка врачей) под конфигом деплоя 21 (`D21_CFG`, прод с 29.10), и
тест проверяет поведение: тему, ответ, перевод, что записано в состояние. Так проверяется «читатель понял одинаково», а не «текст тот
же»: перепись текста ответа контракт не ломает, чтение понимания мимо таблицы — ломает.

Подделка LLM — настоящего типа на настоящей границе (`llm_runtime._OLLAMA_CLIENT`, ответ `ollama.GenerateResponse`), память
«подделки в тестах — настоящего типа»: решения задач выбора задаются `choice.Choice` и проходят настоящий разбор `choice._judge`
(исход, которого нет у задачи, — INVALID, как у живой модели); задача узнаётся по своей инструкции (роль `system` = промпт
`choose_<задача>` до `<<DATA>>`), а не по случайным словам промпта. Разметка слов `turn_spans` — оракул `spans_oracle`; классификатор —
метка темы. Место, для которого понимания не задано, — «LLM недоступна» (как у лежащей модели), предохранитель не размыкается
(`keep_llm_breaker_closed`).

Поля понимания, которых ещё нет у LLM-производителя (`TurnUnderstanding.facet`, `names_service` — их пока решают правила), задаются
подменой на выходе производителя (`Understood.tu`: поля `dataclasses.replace` поверх построенного объекта). Поля шага 0 штурмана
(`intent`, `frame`, `expects`) лягут сюда же: когда они появятся у `TurnUnderstanding`, тест задаёт их тем же `tu={...}`, без правки
харнесса; до того `tu` с неизвестным полем — ошибка теста (`TypeError` у `replace`), а не молчаливый пропуск.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router.choice import Choice, Status
from tests.spans_oracle import INSURANCE_MARK, insurance_response, is_spans, keep_llm_breaker_closed, spans_response
from tests.test_button_actions import GATEWAY_V1_TITLES

# Конфиг деплоя 21 (прод с 29.10 15:50 Самара): известные коду ключи `cfg` прогона стенда `runs/nason_btn` (кандидат 21, valid) —
# шесть ключей кнопок, где умолчание кода ≠ проду (карта Понимания 30.10, C §0), и прочие ключи прода. Таймауты LLM здесь не нужны
# (подделка отвечает сразу), но держатся как на проде: дедлайн хода и лимит решений меняют путь.
D21_CFG: dict[str, Any] = {
    "MR_BUTTON_ANSWER_TOPIC": "frame",
    "MR_BUTTON_LEAF_FRAME": "menu.tests.booking_needed",
    "MR_BUTTON_FRAME_CHECK": "judge",
    "MR_BUTTON_TOPIC_FIRM": "on",
    "MR_BUTTON_REPLY_JUDGE": "on",
    "MR_BUTTON_SIDE_ANSWER": "on",
    "MR_BUTTON_PRICE_SCOPE": "doctor",
    "MR_PENDING_BUTTONS": "on",
    "MR_LLM_DECISION_TURN_CALLS": 4,
    "MR_LLM_DECISION_TURN_SECONDS": 25,
    "MR_LLM_TURN_DEADLINE_S": 50,
    "MR_CLASSIFIER_TIMEOUT_S": 20,
    "MR_PRICE_NOT_FOUND_BY_DATA": "on",
    "MR_POSITION_QUALIFIERS": "on",
    "MR_QUALIFIER_POOL": "on",
    "MR_QUALIFIER_MARKS": "on",
    "MR_INSURANCE_TOPIC": "on",
    "MR_TOPIC_OVERRIDE_LLM": "on",
    "MR_TOPIC_BY_LLM": (
        "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,rule_website_help,"
        "rule_address_work_hours,rule_test_result,rule_prepare,rule_address"
    ),
    "MR_TURN_UNDERSTANDING_PRICE": "on",
    "MR_TURN_UNDERSTANDING_ONE": "on",
    "MR_CHOOSE_CACHE_TTL_S": 0,
    "MR_PARSE_FAIL_VARIANTS": 3,
    "MR_P1_FACET_NONE_OFFER": "on",
    "MR_DOCTORS_BY_SERVICE_DATA": "shadow",
    "MR_PREPARE_BY_PARSE": "on",
    "MR_TEST_ASSIST_FROM_DATA": "on",
    "MR_PRICE_BY_PARSE": "on",
    "MR_BOOKING_TARGET_SLOT_LLM": "on",
    "MR_NAMES_DOCTOR_CATALOG": "on",
    "MR_NAMES_DOCTOR_NOISE": "on",
    "MR_NAMES_DOCTOR_FOLLOWUP": "on",
    "MR_NAMES_RESULT_SURNAME": "on",
    "MR_NAMES_PATIENT_FIO": "on",
    "MR_NAMES_PATIENT_RULE": "on",
    "MR_NAMES_PATIENT_FILL": "on",
    "MR_BASKET_CALC": "on",
    "MR_ADDRESS_BRANCH_FLAGS": "on",
    "MR_ADDRESS_FOLLOWUP_LLM": "on",
    "MR_TEST_ASSIST_PRICE": "on",
    "MR_WALKIN_BY_LLM": "on",
    "MR_BARE_NAME_OVERVIEW": "on",
    "MR_BOOKING_OPTIONS_PRICE": "on",
    "MR_ADDRESS_NAMED_STREET": "on",
    "MR_STRONG_FAMILY_LETTER": "on",
    "MR_BASKET_FAIL_UNPRICED": "on",
    "MR_LLM_FAIL_OFFER": "address_followup,facet",
}

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def d21_world(tmp_path_factory) -> Path:
    """Мир стенда для контракта: «мини-прод» (404 строки настоящего прайса) + выдержка справочника врачей (41 врач, без описаний —
    `tests/fixtures/doctors_excerpt`, как подставляет гейт): без врачей лист «Записаться на приём» и расписание отвечают «не удалось
    получить данные», и рамка записи не видна. Филиалов в мире нет (адресные ответы — «не нашёл филиалы»)."""

    source = tmp_path_factory.mktemp("d21_source")
    shutil.copytree(_FIXTURES / "mini_prod" / "snapshots", source, dirs_exist_ok=True)
    shutil.copy2(_FIXTURES / "doctors_excerpt" / "doctors_20261008.jsonl", source / "doctors_20261008.jsonl")
    return ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("d21_world") / "world")


CLASSIFIER_MARK = "Ты классификатор запросов пациента"
# Задачи судьи ответа на висящий вопрос со свободным ответом: `pending_free_reply` (вопрос-слот), `…_button` (вопрос кнопки,
# вариант А), `…_button_frame` (вариант Б деплоя 21 — с исходом `domain`). `Understood.pending` отвечает за все.
# Вопросы «да/нет» и выбора (`pending_reply`, `pending_reply_many`: yes / no / pause / new_topic) — тем же полем: исход, которого у
# задачи нет (`answer` на вопрос «да/нет»), — INVALID, как у живой модели, и запасной путь правил.
PENDING_TASKS = ("pending_free_reply", "pending_free_reply_button", "pending_free_reply_button_frame", "pending_reply", "pending_reply_many")
FRAME_TASK = "button_frame"  # вопрос рамки листа без своего вопроса (`choose_button_frame`: same / other)
_OPTION_LINE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


def outcome(name: str, *picked: str) -> Choice:
    """Решение задачи с исходом (`answer`, `domain`, `new_topic`, `same`…); `picked` — тексты вариантов, названных вместе с ним."""

    return Choice(Status.OUTCOME, ids=tuple(picked), outcome=name)


def picked(*texts: str) -> Choice:
    """Выбор вариантов по их тексту (тот, что видит модель): у задачи с номерами харнесс сам находит номер строки."""

    return Choice(Status.PICKED, ids=tuple(texts))


NONE_FIT = Choice(Status.NONE_FIT)
DOWN = Choice(Status.UNAVAILABLE)


@dataclass(frozen=True)
class Understood:
    """Что LLM поняла на ходе — ровно то, что тест задаёт; прочее — «недоступна».

    :param services: фразы-услуги разметки `turn_spans` (по словам реплики; пусто — «услуг нет»); None — разметка недоступна
    :param pending: исход судьи висящего вопроса со свободным ответом (`answer` / `no` / `pause` / `new_topic` / `domain`); None —
        судья недоступен (запасной путь правил)
    :param frame: вопрос рамки листа без вопроса: `same` / `other`; None — недоступен
    :param topic: метка классификатора темы; None — классификатор недоступен
    :param choices: прочие задачи выбора: имя задачи (без `choose_`) → `Choice`
    :param tu: поля `TurnUnderstanding`, которые ход «понял» поверх построенного производителем (`facet`, `names_service`; шаг 0 —
        `intent` / `frame` / `expects`); пусто — как построил производитель
    """

    services: tuple[str, ...] | None = ()
    pending: str | None = None
    frame: str | None = None
    topic: str | None = None
    choices: Mapping[str, Choice] = field(default_factory=dict)
    tu: Mapping[str, Any] = field(default_factory=dict)

    def choice_for(self, task: str) -> Choice | None:
        if task in self.choices:
            return self.choices[task]
        if task in PENDING_TASKS and self.pending is not None:
            return outcome(self.pending)
        if task == FRAME_TASK and self.frame is not None:
            return outcome(self.frame)
        return None


@dataclass(frozen=True)
class Turn:
    """Ход диалога: реплика (или подпись нажатой кнопки) и что LLM поняла."""

    text: str
    button: str = ""
    understood: Understood = Understood()


def say(text: str, **understood: Any) -> Turn:
    return Turn(text=text, understood=Understood(**understood))


def press(button_id: str, **understood: Any) -> Turn:
    """Нажатие листа меню: подпись — как шлёт шлюз (`GATEWAY_V1_TITLES`)."""

    return Turn(text=GATEWAY_V1_TITLES[button_id], button=button_id, understood=Understood(**understood))


@lru_cache(maxsize=1)
def _task_systems() -> dict[str, str]:
    """Инструкция (роль `system`) каждой задачи выбора → имя задачи: так `choose()` рендерит запрос (`choice.render`)."""

    from messengers_router import prompt_registry
    from messengers_router.choice import _DATA_MARK

    systems: dict[str, str] = {}
    for path in sorted(prompt_registry._PROMPTS_DIR.glob("choose_*.txt")):
        key = path.stem
        system, mark, _data = prompt_registry.load_prompt_text(key).partition(_DATA_MARK)
        if mark:
            systems.setdefault(system.strip(), key.removeprefix("choose_"))
    return systems


def _choice_json(choice: Choice, prompt: str, schema: Any) -> str:
    """`Choice` → ответ модели по схеме задачи (обратное `choice._judge`): номера по тексту вариантов из запроса."""

    if choice.status is Status.INVALID:
        return "не json"
    picks_schema = ((schema or {}).get("properties") or {}).get("picks") or {}
    by_text = ((picks_schema.get("items") or {}).get("type")) == "string"
    numbered = {m.group(2): int(m.group(1)) for line in prompt.splitlines() if (m := _OPTION_LINE.match(line))}
    picks: list[Any] = []
    for text in choice.ids:
        if by_text:
            picks.append(text)
        elif text in numbered:
            picks.append(numbered[text])
        else:
            raise AssertionError(f"харнесс: варианта «{text}» нет в запросе задачи")
    answer: dict[str, Any] = {"picks": picks}
    props = (schema or {}).get("properties") or {}
    if "outcome" in props:
        answer["outcome"] = choice.outcome or "none"
    elif choice.outcome:
        answer["outcome"] = choice.outcome  # у задачи нет исходов — `_judge` его не читает, как у живой модели
    if "reason" in props:
        answer["reason"] = choice.reason or "none"
    if "search_terms" in props:
        answer["search_terms"] = list(choice.terms)
    return json.dumps(answer, ensure_ascii=False)


class UnderstandingLLM:
    """Подделка LLM на границе `llm_runtime._OLLAMA_CLIENT`: отвечает пониманием текущего хода (`understood`)."""

    def __init__(self) -> None:
        self.understood = Understood()
        self.asked: list[str] = []  # места, куда пришёл ход: `spans`, `classifier`, `choose_<задача>`, `…:down`

    async def generate(self, *, prompt: str = "", system: str | None = None, format: Any = None, **_kwargs: Any) -> GenerateResponse:
        u = self.understood
        if is_spans(prompt):
            return self._answer("spans", None if u.services is None else spans_response(prompt, u.services))
        if INSURANCE_MARK in (prompt or ""):
            return self._answer("insurance", None)
        task = _task_systems().get(str(system or "").strip()) if system else None
        if task is not None:
            choice = u.choice_for(task)
            if choice is None or choice.status is Status.UNAVAILABLE:
                return self._answer(f"choose_{task}", None)
            return self._answer(f"choose_{task}", GenerateResponse(model="m", response=_choice_json(choice, prompt, format), done=True))
        if CLASSIFIER_MARK in f"{system or ''}\n{prompt}":
            topic = None
            if u.topic is not None:
                topic = GenerateResponse(
                    model="m",
                    response=json.dumps({"label": u.topic, "confidence": 0.9, "context_action": "new_topic", "entities": {}}),
                    done=True,
                )
            return self._answer("classifier", topic)
        return self._answer("other", None)

    def _answer(self, site: str, response: GenerateResponse | None) -> GenerateResponse:
        if response is None:
            self.asked.append(f"{site}:down")
            raise ConnectionError(f"тест: LLM не поняла ничего для {site}")
        self.asked.append(site)
        return response


def _install_tu_overrides(patch: Any, llm: UnderstandingLLM) -> None:
    """Поля понимания хода, заданные тестом, — поверх объекта производителя (`observe_price_turn`), до всех читателей."""

    from messengers_router import turn_understanding as tu_mod

    real = tu_mod.observe_price_turn

    async def observe(label: str, *, user_text: str, pending: Any) -> Any:
        built = await real(label, user_text=user_text, pending=pending)
        overrides = dict(llm.understood.tu)
        if not overrides:
            return built
        base = built if built is not None else tu_mod.TurnUnderstanding(reply=str(user_text or ""))
        changed = dataclasses.replace(base, **overrides)
        tu_mod._CURRENT.set(changed)
        return changed

    patch.setattr(tu_mod, "observe_price_turn", observe)
    import messengers_router.router as router_mod

    if getattr(router_mod, "turn_understanding", None) is not tu_mod:
        raise AssertionError("харнесс: роутер читает понимание не через модуль turn_understanding")


def _record(body: dict[str, Any], state: dict[str, Any], telemetry: dict[str, Any], asked: list[str]) -> dict[str, Any]:
    dbg = (body.get("state_update") or {}).get("debug") or {}
    plan = dbg.get("plan") or {}
    pending = state.get("_pending") if isinstance(state.get("_pending"), dict) else None
    return {
        "text": str(body.get("text") or ""),
        "handoff": bool(body.get("handoff")),
        "label": (dbg.get("decision") or {}).get("label"),
        "plan": [plan.get("label")] + [s.get("tool") for s in plan.get("steps") or []] if plan else None,
        "understanding": dbg.get("understanding"),
        "service_name": state.get("service_name"),
        "test_goal": state.get("test_goal"),
        "pending": {"label": pending.get("label"), "missing": pending.get("missing")} if pending else None,
        "expects": state.get("_expects"),
        "state": state,
        "telemetry": telemetry,
        "asked": list(asked),
    }


def run(world_dir: Any, monkeypatch: Any, turns: list[Turn], *, cfg: Mapping[str, Any] | None = None, session: str = "") -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта в одной сессии, конфиг деплоя 21 (`cfg` — поверх него), мир `d21_world`, день."""

    from messengers_router import choice as choice_mod
    from messengers_router import llm_runtime
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as rcfg
    from messengers_router.services import Services

    session = session or "contract-" + json.dumps([(t.button, t.text) for t in turns], ensure_ascii=False)
    llm = UnderstandingLLM()

    async def go(world: Any) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            llm.understood, start = turn.understood, len(llm.asked)
            req = MessengerGenerateRequest(session_id=session, text=turn.text, debug=True, button_id=turn.button)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            state = dict(memory.get(session).last_entities or {})
            out.append(_record(body, state, world.take_telemetry() or {}, llm.asked[start:]))
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {**D21_CFG, **dict(cfg or {})}.items():
            patch.setattr(rcfg, key, value, raising=False)
        keep_llm_breaker_closed(patch)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        _install_tu_overrides(patch, llm)
        choice_mod._CACHE.clear()
        return asyncio.run(go(world))


__all__ = [
    "D21_CFG",
    "DOWN",
    "NONE_FIT",
    "Turn",
    "Understood",
    "UnderstandingLLM",
    "d21_world",
    "insurance_response",
    "outcome",
    "picked",
    "press",
    "run",
    "say",
]
