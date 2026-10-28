"""Цель записи по смыслу: цель, филиал и дату первой реплики записи размечает LLM, значения — данные (`booking_target_llm`,
DESIGN-2026-10-28-booking-target-by-meaning; аудит С6 28.10, классы К13 и К8).

Регулярки резали реплику на роли по буквам: «на ФГДС на Ленина 5» — филиал «ФГДС», «Терапевт мне нужен на осетинской 4» — услуга
«Осетинской» и филиал «осетинской 4», которого нет; когда тему записи решала LLM-классификатор, её свободный `service_name`
(«к врачу» → «врач») ронял поток записи в «Пока не понял…». Инварианты класса (через обработчик эндпоинта, LLM — подделка настоящего
типа на границе `llm_runtime._OLLAMA_CLIENT`):

1. филиал, названный вместе с целью и датой, — в заявке значением справочника, бот не спрашивает филиал (и в свободном тексте, и в
   ответе на кнопку «Записаться на приём»); дата — тоже;
2. место, которого нет в справочнике, в слот филиала не пишется и услугой не становится; число при услуге — не филиал;
3. цели нет («к врачу») — вопрос цели, поток записи активен, и когда тему решала LLM-классификатор;
4. отмену, найденную правилом, разметка в «записаться» не превращает;
5. сбой разметки: без узла — ответ как без выключателя, байт в байт; с узлом `booking_target` — честный оффер;
6. выключатель off и шаги после цели (ответ филиалом) — разметку не спрашивают.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable

import pytest
from ollama import GenerateResponse

from messengers_router.services._common import _normalise_catalog_text
from messengers_router.services._prices_helpers import _PRICE_TOKEN_RE

_MARK = "Пронумерованные слова сообщения:"
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_BOOK_BUTTON = "menu.appointment.book"
_PROD_TOPIC_BY_LLM = (
    "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,rule_website_help,"
    "rule_address_work_hours,rule_test_result,rule_prepare"
)
_BRANCHES = [
    {"id": "branch_8882", "name": "пр.Ленина, 5", "aliases": "пр.ленина, 5, ленина 5, самара"},
    {"id": "branch_8890", "name": "ул. Победы, 83", "aliases": "ул. победы, 83, победы 83, самара"},
    {"id": "branch_8870", "name": "пр.Кирова, 223А", "aliases": "пр.кирова, 223а, кирова 223а, самара"},
]
_BRANCH_NAMES = {b["name"] for b in _BRANCHES}


def _groups(words: list[str], phrases: Iterable[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    for phrase in phrases:
        want = [_normalise_catalog_text(token) for token in _PRICE_TOKEN_RE.findall(str(phrase or ""))]
        for k in range(len(words) - len(want) + 1):
            if want and words[k : k + len(want)] == want:
                groups.append([k + 1, k + len(want)])
                break
    return groups


class _Oracle:
    """Разметка цели записи — заданные слова ролей и действие (`down` — недоступна); классификатор — `classifier` (JSON строкой);
    прочие места — «недоступна». `asked` — какие места спросили."""

    def __init__(self, roles: dict[str, tuple[str, ...]] | None = None, *, action: str = "book", classifier: str | None = None, down: bool = False):
        self.roles = roles or {}
        self.action = action
        self.classifier = classifier
        self.down = down
        self.asked: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        if _MARK in (prompt or ""):
            self.asked.append("booking")
            if self.down:
                raise ConnectionError("тест: разметка цели записи недоступна")
            numbered = prompt.split(_MARK, 1)[1].strip().splitlines()[0].split()
            words = [_normalise_catalog_text(item.partition(":")[2]) for item in numbered]
            answer: dict[str, Any] = {"action": self.action}
            answer.update({role: _groups(words, self.roles.get(role, ())) for role in ("target", "branch", "date")})
            return GenerateResponse(model="m", response=json.dumps(answer), done=True)
        seen = f"{system}\n\n{prompt}" if system else prompt
        if self.classifier is not None and _CLASSIFIER_MARK in seen:
            self.asked.append("classifier")
            return GenerateResponse(model="m", response=self.classifier, done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _dialog(monkeypatch, turns: list[str | tuple[str, str]], oracle: _Oracle, *, on: bool = True, **switches: str) -> list[dict[str, Any]]:
    from messengers_router import llm_runtime
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services
    from tests.spans_oracle import keep_llm_breaker_closed

    monkeypatch.setattr(cfg, "MR_BOOKING_TARGET_LLM", "on" if on else "off", raising=False)
    for key, value in switches.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(Services, "get_branches", lambda self: [dict(b) for b in _BRANCHES])
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    keep_llm_breaker_closed(monkeypatch)
    session = f"booking-target-{on}-{json.dumps(switches, sort_keys=True)}-{turns}"

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            dbg = (body.get("state_update") or {}).get("debug") or {}
            steps = (dbg.get("plan") or {}).get("steps") or []
            planned = dict((steps[0].get("input") or {}).get("entities") or {}) if steps else {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "entities": dict(memory.get(session).last_entities or {}),
                    "decision": dbg.get("decision") or {},
                    "planned": planned,  # слоты, с которыми ход ушёл в инструмент (состояние после заявки уже очищено)
                }
            )
        return out

    return asyncio.run(run())


def _press_then(reply: str) -> list[str | tuple[str, str]]:
    from tests.test_price_turns_golden import _button

    return [_button(_BOOK_BUTTON), reply]


# реплика → (слова цели, слова филиала, слова даты, филиал справочника). Формы — как у провалов стенда деплоя 19 (К13): адрес
# после даты, две «на …» подряд; разрез по буквам здесь переспрашивал филиал или терял услугу.
_NAMED = {
    "Нужна запись к урологу на эту субботу, лучше на Ленина": (("урологу",), ("Ленина",), ("эту субботу",), "пр.Ленина, 5"),
    "Записаться на колоноскопию на Победы 83 в среду": (("колоноскопию",), ("Победы 83",), ("среду",), "ул. Победы, 83"),
    "Записаться к гинекологу 2 ноября на Победы": (("гинекологу",), ("Победы",), ("2 ноября",), "ул. Победы, 83"),
}


def _roles(reply: str) -> dict[str, tuple[str, ...]]:
    target, branch, date, _ = _NAMED[reply]
    return {"target": target, "branch": branch, "date": date}


@pytest.mark.parametrize("via_button", [False, True], ids=["text", "button"])
@pytest.mark.parametrize("reply", list(_NAMED))
def test_named_branch_reaches_the_booking(reply, via_button, monkeypatch):
    oracle = _Oracle(_roles(reply))
    turns = _press_then(reply) if via_button else [reply]
    last = _dialog(monkeypatch, turns, oracle)[-1]
    assert "booking" in oracle.asked
    # Филиал — значение справочника: в слотах хода и в ответе, филиал не переспрашивают.
    assert last["planned"].get("branch_name") == _NAMED[reply][3], last["planned"]
    assert _NAMED[reply][3] in last["text"], last["text"]
    assert "Какой филиал" not in last["text"] and "не нашёл" not in last["text"], last["text"]


# реплика → (слова цели, слова места, которого нет в справочнике)
_UNKNOWN_PLACE = {
    "Записаться к урологу на Садовой 99": ("урологу", "Садовой 99"),
    "Хочу записаться к неврологу на Садовую 12 завтра": ("неврологу", "Садовую 12"),
}


@pytest.mark.parametrize("via_button", [False, True], ids=["text", "button"])
@pytest.mark.parametrize("reply", list(_UNKNOWN_PLACE))
def test_unknown_place_is_not_a_branch_and_not_a_service(reply, via_button, monkeypatch):
    target, place = _UNKNOWN_PLACE[reply]
    oracle = _Oracle({"target": (target,), "branch": (place,), "date": ("завтра",) if "завтра" in reply else ()})
    last = _dialog(monkeypatch, _press_then(reply) if via_button else [reply], oracle)[-1]
    for slots in (last["entities"], last["decision"].get("entities") or {}, last["planned"]):
        assert not slots.get("branch_name") and not slots.get("branch_id"), slots
        assert "садов" not in str(slots.get("service_name") or "").lower(), slots
    assert place.lower() not in last["text"].lower(), last["text"]


def test_number_inside_service_is_not_a_branch(monkeypatch):
    oracle = _Oracle({"target": ("узи 3 скрининг",)})
    last = _dialog(monkeypatch, _press_then("Запись на узи 3 скрининг"), oracle)[-1]
    for slots in (last["entities"], last["planned"]):
        assert not slots.get("branch_name"), slots
    assert "(узи 3)" not in last["text"].lower(), last["text"]


_GENERIC = "Хочу записаться к врачу"
_LLM_FREE_TEXT = json.dumps(
    {"label": "APPOINTMENT", "confidence": 0.9, "context_action": "new_topic", "entities": {"service_name": "врач", "appointment_action": None}}
)


@pytest.mark.parametrize("topic", ["rule", "llm"])
def test_no_target_asks_the_target_and_starts_booking(topic, monkeypatch):
    switches = {"MR_TOPIC_BY_LLM": f"{_PROD_TOPIC_BY_LLM},rule_appointment"} if topic == "llm" else {}
    oracle = _Oracle({}, classifier=_LLM_FREE_TEXT if topic == "llm" else None)
    [turn] = _dialog(monkeypatch, [_GENERIC], oracle, **switches)
    assert "booking" in oracle.asked
    if topic == "llm":
        assert "classifier" in oracle.asked
    assert "Пока не понял" not in turn["text"] and "оператор" not in turn["text"].lower(), turn["text"]
    assert "к какому врачу" in turn["text"].lower(), turn["text"]
    assert turn["entities"].get("appointment_flow_active") is True, turn["entities"]
    assert turn["entities"].get("appointment_action") == "book", turn["entities"]


def test_rule_cancel_is_not_turned_into_booking(monkeypatch):
    oracle = _Oracle({"target": ("урологу",), "date": ("завтра",)}, action="book")
    [turn] = _dialog(monkeypatch, ["Хочу отменить запись к урологу на завтра"], oracle)
    assert "booking" in oracle.asked
    # Отмена осталась отменой: путь своей записи (оператор), а не новая заявка.
    assert "existing_appointment_cancel" in (turn["decision"].get("flags") or []), turn["decision"]
    assert "Заявка на запись" not in turn["text"] and "оператор" in turn["text"].lower(), turn["text"]


@pytest.mark.parametrize(
    "rules,llm,want",
    [("cancel", "book", "cancel"), ("reschedule", "book", "reschedule"), ("book", "cancel", "cancel"), ("", "book", "book"),
     ("book", "none", "book"), ("", "none", ""), ("cancel", "reschedule", "cancel")],
)
def test_action_is_monotone(rules, llm, want):
    from messengers_router.booking_target import merged_action

    assert merged_action(rules, llm) == want


_FAIL_REPLY = "Нужна запись к урологу на эту субботу, лучше на Ленина"


@pytest.mark.parametrize("via_button", [False, True], ids=["text", "button"])
def test_parse_failure_without_node_is_the_answer_without_switch(via_button, monkeypatch):
    turns = _press_then(_FAIL_REPLY) if via_button else [_FAIL_REPLY]
    off = _dialog(monkeypatch, turns, _Oracle(down=True), on=False)
    down = _Oracle(down=True)
    on = _dialog(monkeypatch, turns, down)
    assert "booking" in down.asked
    assert [t["text"] for t in on] == [t["text"] for t in off]


@pytest.mark.parametrize("via_button", [False, True], ids=["text", "button"])
def test_parse_failure_under_node_is_an_honest_offer(via_button, monkeypatch):
    turns = _press_then(_FAIL_REPLY) if via_button else [_FAIL_REPLY]
    last = _dialog(monkeypatch, turns, _Oracle(down=True), MR_LLM_FAIL_OFFER="booking_target")[-1]
    assert "ответит оператор" in last["text"], last["text"]


def test_switch_off_and_later_steps_do_not_ask_the_parse(monkeypatch):
    off = _Oracle(_roles(_FAIL_REPLY))
    _dialog(monkeypatch, [_FAIL_REPLY], off, on=False)
    assert "booking" not in off.asked

    # Цель названа, филиала нет — бот спрашивает филиал; ответ филиалом — структурный шаг, разметку не спрашивают.
    oracle = _Oracle({"target": ("урологу",)})
    first, second = _dialog(monkeypatch, ["Записаться к урологу", "Ленина 5"], oracle)
    assert "Какой филиал" in first["text"], first["text"]
    assert oracle.asked.count("booking") == 1, oracle.asked

    # Нажатие кнопки — подпись кнопки, а не реплика: разметку спрашивают только на ответе пациента.
    pressed = _Oracle({"target": ("урологу",)})
    _dialog(monkeypatch, _press_then("Записаться к урологу"), pressed)
    assert pressed.asked.count("booking") == 1, pressed.asked
    assert second["planned"].get("branch_name") in _BRANCH_NAMES, second["planned"]


@pytest.mark.parametrize(
    "raw",
    [
        {"action": "book", "target": [[1, 2]], "branch": [[2, 3]], "date": []},  # роли пересекаются
        {"action": "book", "target": [[0]], "branch": [], "date": []},  # номер вне списка
        {"action": "maybe", "target": [], "branch": [], "date": []},  # действие вне контракта
        {"action": "book", "target": [[3, 1]], "branch": [], "date": []},  # первое больше последнего
        {"action": "book", "target": [[1, 3, 2]], "branch": [], "date": []},  # перечень не по возрастанию
        {"action": "book", "target": "1", "branch": [], "date": []},
    ],
)
def test_judge_rejects_off_contract(raw):
    from messengers_router.booking_target import judge

    assert judge(raw, 4) is None


def test_judge_reads_groups_by_role():
    from messengers_router.booking_target import judge

    assert judge({"action": "book", "target": [[2]], "branch": [[4, 5]], "date": [[6, 7, 8]]}, 8) == (
        "book",
        {"target": [(1, 1)], "branch": [(3, 4)], "date": [(5, 7)]},
    )
    # Перечень с пропуском предлога («8 октября в 10 утра» без «в») — сплошные группы, а не нарушение контракта.
    assert judge({"action": "book", "target": [], "branch": [], "date": [[1, 2, 4, 5]]}, 5) == (
        "book",
        {"target": [], "branch": [], "date": [(0, 1), (3, 4)]},
    )


_ABDOMEN = "Ультразвуковое исследование органов брюшной полости (комплексное)"


@pytest.fixture
def uzi_synonym(monkeypatch):
    """Синоним клиники «узи» → одна строка (как в живом срезе МИС: «запишите на узи» сильно сводится к УЗИ брюшной полости)."""

    from messengers_router.services import _biomaterial as _bio

    hermetic = _bio._vocabularies

    def _with_uzi():
        vocab = hermetic()
        vocab.synonym_candidates = {"узи": (_ABDOMEN,)}
        vocab.synonym_to_service = {"узи": _ABDOMEN}
        return vocab

    monkeypatch.setattr(_bio, "_vocabularies", _with_uzi)


@pytest.mark.parametrize(
    "reply,target",
    [("Запись на УЗИ", ("УЗИ",)), ("Могу ли я завтра пройти гинекологическое УЗИ?", ("гинекологическое УЗИ",))],
)
def test_target_value_on_the_target_step_is_as_without_parse(reply, target, uzi_synonym, monkeypatch):
    """Разметка меняет роли слов, а не значение цели: на шаге цели слот услуги — слова пациента, как у разреза по буквам; общее
    «УЗИ» не становится одной строкой синонима клиники («гинекологическое УЗИ» — не брюшной полости)."""

    roles = {"target": target, "date": ("завтра",) if "завтра" in reply else ()}
    off = _dialog(monkeypatch, _press_then(reply), _Oracle(roles), on=False)[-1]
    oracle = _Oracle(roles)
    on = _dialog(monkeypatch, _press_then(reply), oracle)[-1]
    assert "booking" in oracle.asked
    assert on["entities"].get("service_name") == off["entities"].get("service_name"), (on["entities"], off["entities"])
    assert "брюшной" not in on["text"].lower(), on["text"]


_GYN_UZI = "УЗИ гинекологическое (трансвагинальное)"


@pytest.fixture
def gyn_synonym(monkeypatch):
    """Синоним клиники на слово специальности: «гинеколог» → строка процедуры (так совпадало «гинекологу» в свипе 28.10)."""

    from messengers_router.services import _biomaterial as _bio

    hermetic = _bio._vocabularies

    def _with_gyn():
        vocab = hermetic()
        vocab.synonym_candidates = {"гинеколог": (_GYN_UZI,), "гинекологу": (_GYN_UZI,)}
        vocab.synonym_to_service = {"гинеколог": _GYN_UZI, "гинекологу": _GYN_UZI}
        return vocab

    monkeypatch.setattr(_bio, "_vocabularies", _with_gyn)


@pytest.mark.parametrize("via_button", [False, True], ids=["text", "button"])
def test_specialty_word_is_a_reception_not_a_procedure(via_button, gyn_synonym, monkeypatch):
    """Слово специальности в цели — приём врача (роль каталога МИС), а не строка процедуры по синониму клиники."""

    reply = "Хочу гинекологу на прием"
    oracle = _Oracle({"target": ("гинекологу",)})
    last = _dialog(monkeypatch, _press_then(reply) if via_button else [reply], oracle)[-1]
    assert "booking" in oracle.asked
    for slots in (last["entities"], last["decision"].get("entities") or {}, last["planned"]):
        assert str(slots.get("service_name") or "") != _GYN_UZI, slots
    assert "узи" not in last["text"].lower(), last["text"]


@pytest.mark.parametrize("via_button", [False, True], ids=["text", "button"])
def test_branch_without_the_target_is_not_booked(via_button, monkeypatch):
    """Филиал есть в справочнике, но цели там по данным хода нет (адреса терапевта — без Кирова): ложного «да, можем записать»
    и цены нет — бот показывает филиалы, где цель есть (К8)."""

    reply = "Записаться к терапевту на Кирова 223А"
    oracle = _Oracle({"target": ("терапевту",), "branch": ("Кирова 223А",)})
    last = _dialog(monkeypatch, _press_then(reply) if via_button else [reply], oracle)[-1]
    assert "booking" in oracle.asked
    assert "можем записать" not in last["text"] and "Заявка на запись" not in last["text"], last["text"]
    assert "Какой филиал" in last["text"], last["text"]
    assert not last["entities"].get("branch_name"), last["entities"]
