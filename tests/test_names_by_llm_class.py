"""Имена без списков стоп-слов: сущности LLM хода и справочник МИС (этап 3, STEP-2026-10-24-NAMES-NO-STOPWORDS).

Инварианты класса:
- имя пациента (ФИО, фамилия) — только сущность LLM-классификатора этого хода, и только подстрока реплики (регистр, «ё», знаки
  не важны); LLM не звали, сбой, модель имени не назвала — имени нет (решение владельца 22.10 (а): «уточните»);
- имена прошлого хода не читаются;
- фамилия врача — точное совпадение слова реплики с фамилией справочника МИС в любой падежной форме; нечётко — только имя,
  которое назвала LLM, и исход — вопрос, а не ответ;
- путь `on` каждого выключателя `names_*` не читает свой список.
ПДн — только выдуманные (репо публичный); фамилии врачей выбирает код из выдержки справочника в репо.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import json
from pathlib import Path

import pytest

from messengers_router import classifier, turn_names, turn_understanding
from messengers_router.services import Services


@pytest.fixture(autouse=True)
def _fresh_turn():
    """Память хода — своя на тест и не переживает его (соседи по воркеру `-n` ждут «вне хода»)."""

    token = turn_names._CURRENT.set(turn_names._Turn())
    yield
    turn_names._CURRENT.reset(token)


def _switch(monkeypatch, name: str, value: str = "on") -> None:
    monkeypatch.setattr(turn_names._cfg, "MR_" + name.upper(), value, raising=False)


# --- память имён хода ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "said"),
    [
        ("Сидорова Анна", "Сидорова Анна"),
        ("сидорова анна", "Сидорова Анна"),
        ("СИДОРОВА АННА!", "сидорова анна"),
        ("Сидорова Алёна", "Сидорова Алена"),
    ],
)
def test_llm_patient_name_is_kept_when_substring_of_reply(reply: str, said: str) -> None:
    turn_names.note_llm(reply, {"patient_name": said})
    assert turn_names.llm_patient_fio(reply, switch="names_patient_fio")


@pytest.mark.parametrize("said", ["Сидорова Анна Петровна", "Петрова Анна", "Анна Сидорова"])
def test_llm_patient_name_not_in_reply_is_dropped(said: str) -> None:
    turn_names.note_llm("Сидорова Анна", {"patient_name": said})
    assert not turn_names.llm_patient_fio("Сидорова Анна", switch="names_patient_fio")


def test_no_llm_this_turn_means_no_name() -> None:
    assert not turn_names.llm_patient_fio("Сидорова Анна", switch="names_patient_fio")
    assert not turn_names.llm_person_word("Сидорова", switch="names_result_surname")


def test_new_turn_forgets_previous_llm_names() -> None:
    def two_turns() -> bool:
        turn_names.note_llm("Сидорова Анна", {"patient_name": "Сидорова Анна"})
        turn_understanding.begin_turn()
        return turn_names.llm_patient_fio("Сидорова Анна", switch="names_patient_fio")

    assert not contextvars.copy_context().run(two_turns)  # ход — в своём контексте, как у роутера


def test_llm_names_noted_in_child_task_are_seen_by_parent() -> None:
    async def turn() -> bool:
        turn_names.begin()

        async def classify() -> None:
            turn_names.note_llm("Сидорова", {"surname": "Сидорова"})

        await asyncio.gather(classify())
        return turn_names.llm_person_word("Сидорова", switch="names_result_surname")

    assert asyncio.run(turn())


@pytest.mark.parametrize("word", ["Сидорова", "сидорова", "СИДОРОВА"])
def test_llm_surname_or_patient_name_word(word: str) -> None:
    turn_names.note_llm("Сидорова", {"surname": "Сидорова"})
    assert turn_names.llm_person_word(word, switch="names_result_surname")
    turn_names.note_llm("Сидорова Анна", {"patient_name": "Сидорова Анна"})
    assert turn_names.llm_person_word(word, switch="names_result_surname")


def test_classifier_llm_answer_is_the_producer(monkeypatch) -> None:
    """Классификатор `legacy_v2` кладёт сущности имён ответа LLM в память хода (правило не сработало)."""

    async def llm(_prompt: str, *, queue_timeout_ms: int = 30000) -> dict:
        return {"label": "TEST_RESULT", "confidence": 0.9, "entities": {"surname": "Сидорова"}}

    monkeypatch.setattr(classifier, "ollama_classify_json", llm)

    async def turn() -> bool:
        turn_understanding.begin_turn()
        await classifier.analyze("Сидорова", {}, skip_rule=True)
        return turn_names.llm_person_word("Сидорова", switch="names_result_surname")

    assert asyncio.run(turn())


def test_switch_of_other_module_is_programmer_error() -> None:
    with pytest.raises(KeyError):
        turn_names.on("pending_reply")


def test_switches_are_off_by_default() -> None:
    assert not any(turn_names.on(name) for name in _GUARDS)


# --- names_doctor_catalog: врач реплики — по справочнику МИС --------------------------------------------------------

_EXCERPT = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"


def _excerpt_doctors() -> list[dict]:
    with _EXCERPT.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _surnames() -> list[str]:
    found = [str(d.get("fio") or "").split()[0] for d in _excerpt_doctors() if str(d.get("fio") or "").strip()]
    return sorted(s for s in set(found) if found.count(s) == 1 and s.isalpha())


def _word_forms(surname: str) -> list[str]:
    """Падежные формы фамилии на -ов/-ев/-ин и -ова/-ева/-ина (как пишут пациенты: «к Иванову», «у Ивановой»)."""

    low = surname.lower()
    if low.endswith(("ова", "ева", "ина")):
        stem = surname[:-1]
        return [surname, stem + "ой", stem + "у"]
    if low.endswith(("ов", "ев", "ин")):
        return [surname, surname + "а", surname + "у", surname + "ым", surname + "е"]
    return [surname]


def _declined_surnames() -> list[str]:
    male = next(s for s in _surnames() if s.lower().endswith(("ов", "ев", "ин")))
    female = next(s for s in _surnames() if s.lower().endswith(("ова", "ева", "ина")))
    return [male, female]


def _catalog_services() -> Services:
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    async def loaded():
        return _excerpt_doctors()

    services._ensure_doctors_cache_loaded = loaded
    return services


def _match(text: str) -> dict:
    return asyncio.run(_catalog_services().match_catalog_doctor(text))


@pytest.mark.parametrize("surname", _declined_surnames())
@pytest.mark.parametrize("case", [str, str.lower, str.upper])
def test_catalog_doctor_by_data_any_word_form_and_case(monkeypatch, surname: str, case) -> None:
    _switch(monkeypatch, "names_doctor_catalog")
    for form in _word_forms(surname):
        match = _match(f"Хочу записаться на приём к врачу {case(form)}")
        assert (match["status"], match["canonical"]) == ("exact", surname), form


@pytest.mark.parametrize(
    "text",
    [
        "По каким адресам можно сделать флюорографию",
        "УЗИ органов малого таза",
        "Какова стоимость услуг",
        "Хочу записаться к врачу",
        "да, хорошо",
    ],
)
def test_catalog_doctor_by_data_no_surname_no_guess(monkeypatch, text: str) -> None:
    """Слово без фамилии справочника — не врач; прежний путь находил «Ким» в «каким» нечётким сравнением (свип 24.10)."""

    _switch(monkeypatch, "names_doctor_catalog")
    assert _match(text)["status"] == "miss"


def _typo(surname: str) -> str:
    middle = len(surname) // 2
    return surname[:middle] + ("о" if surname[middle] != "о" else "а") + surname[middle + 1 :]


def test_catalog_doctor_typo_only_in_name_llm_named(monkeypatch) -> None:
    _switch(monkeypatch, "names_doctor_catalog")
    surname = max(_surnames(), key=len)
    typo = _typo(surname)
    reply = f"Запишите к {typo}"
    assert _match(reply)["status"] == "miss"  # LLM имени не называла — по буквам не угадываем
    turn_names.note_llm(reply, {"doctor_name": typo})
    match = _match(reply)
    assert (match["status"], match["canonical"]) == ("fuzzy", surname)  # исход — вопрос «Похоже, вы имели в виду…?»


# --- names_doctor_noise: служебное слово как doctor_name — решает справочник ------------------------------------------


def _verify(services: Services, doctor_name: str, text: str):
    from messengers_router import entity_grounder
    from messengers_router.mess_types import RouteDecision

    decision = RouteDecision(label="DOCTOR_SCHEDULE", confidence=0.8, entities={"doctor_name": doctor_name}, source="llm_primary")
    return asyncio.run(entity_grounder.verify_doctor_entities_in_decision(decision, text, services))


@pytest.mark.parametrize("word", ["хочу", "Запись", "ЗАПИСАТЬСЯ", "когда", "да", "нет", "подскажите", "приёма", "записи", "хотела"])
def test_service_word_as_doctor_name_is_not_a_doctor_by_data(monkeypatch, word: str) -> None:
    """Любая форма служебного слова (и те, которых в списке не было) — не врач: справочник его не находит."""

    _switch(monkeypatch, "names_doctor_noise")
    out = _verify(_catalog_services(), word, f"{word} к врачу")
    assert "doctor_name" not in out.entities and "doctor_name_unverified" in out.flags


@pytest.mark.parametrize("surname", _declined_surnames())
def test_doctor_surname_entity_any_form_verified_by_data(monkeypatch, surname: str) -> None:
    _switch(monkeypatch, "names_doctor_noise")
    for form in _word_forms(surname):
        out = _verify(_catalog_services(), form.lower(), f"когда принимает {form.lower()}")
        assert out.entities.get("doctor_name") == surname and "doctor_name_verified" in out.flags, form


# --- names_doctor_followup: «да / хорошо» — не фамилия по справочнику -----------------------------------------------

_DOCTOR_CONTEXT = {"_last_label": "DOCTOR_SCHEDULE", "specialty": "кардиолог"}


@pytest.mark.parametrize("surname", _declined_surnames())
@pytest.mark.parametrize("case", [str, str.lower, str.upper])
def test_doctor_followup_surname_any_form_and_case(monkeypatch, surname: str, case) -> None:
    _switch(monkeypatch, "names_doctor_followup")
    for form in _word_forms(surname):
        for text in (f"{case(form)}, да", f"{case(form)} подходит"):
            assert classifier._doctor_followup_name_in_context(text, dict(_DOCTOR_CONTEXT)) == surname, text


@pytest.mark.parametrize("text", ["да, хорошо", "ладно", "подходит", "хорошо", "ок, именно", "Хорошо, спасибо", "Угу, подойдёт"])
def test_doctor_followup_without_surname_is_not_a_name(monkeypatch, text: str) -> None:
    """Без фамилии справочника правило молчит; «хорошо» ≠ «Хорошун» — сравнение точное (BUG-2026-10-08-BARE-DOCTOR-SURNAME)."""

    _switch(monkeypatch, "names_doctor_followup")
    assert classifier._doctor_followup_name_in_context(text, dict(_DOCTOR_CONTEXT)) is None


# --- names_result_surname: одно слово — фамилия, только если её назвала LLM ------------------------------------------

_RESULT_FIELDS = ["surname", "year", "filial", "number"]


def _result_after_llm(reply: str, llm_entities: dict | None) -> dict:
    from messengers_router import policies

    if llm_entities is not None:
        turn_names.note_llm(reply, llm_entities)
    out: dict = {}
    policies._fill_test_result_entities(reply, _RESULT_FIELDS, out)
    return out


@pytest.mark.parametrize("reply", ["Сидорова", "сидорова", "СИДОРОВА", "Сидорова-Белая"])
def test_result_surname_named_by_llm_any_case(monkeypatch, reply: str) -> None:
    _switch(monkeypatch, "names_result_surname")
    said = reply.title()
    assert _result_after_llm(reply, {"surname": said})["surname"] == reply.capitalize()


@pytest.mark.parametrize("reply", ["Результаты", "Результатов", "Анализы", "Анализом", "Хочу", "Подскажите", "Пожалуйста"])
def test_result_word_llm_did_not_name_is_not_surname(monkeypatch, reply: str) -> None:
    """Любая форма служебного слова (и те, которых в списке нет: «Результатов», «Анализом», «Пожалуйста») — не фамилия."""

    _switch(monkeypatch, "names_result_surname")
    assert "surname" not in _result_after_llm(reply, {"test_name": "анализы"})


def test_result_no_llm_no_surname(monkeypatch) -> None:
    _switch(monkeypatch, "names_result_surname")
    assert "surname" not in _result_after_llm("Сидорова", None)


def test_result_ordered_fields_do_not_need_llm(monkeypatch) -> None:
    """Порядок «фамилия, год, код, номер» — механика формы, не имя по смыслу: LLM не нужна."""

    _switch(monkeypatch, "names_result_surname")
    out = _result_after_llm("Сидорова, 1985, Бг, 123", None)
    assert (out["surname"], out["year"], out["number"]) == ("Сидорова", 1985, 123)


def _result_dialog(monkeypatch, second: str, llm_entities: dict | None) -> tuple[str, dict]:
    from messengers_router import router as router_mod
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import SessionState

    if llm_entities is not None:
        async def llm(_prompt: str, *, queue_timeout_ms: int = 30000) -> dict:
            return {"label": "TEST_RESULT", "confidence": 0.9, "context_action": "continue", "entities": llm_entities}

        monkeypatch.setattr(classifier, "ollama_classify_json", llm)
    state = SessionState(session_id=f"names-result-{second}-{bool(llm_entities)}")
    memory = MemoryStore()
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    async def go() -> str:
        answer = ""
        for text in ("Хочу получить результаты анализов", second):
            out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
            answer = "".join(env.text for env in out if env.text)
        return answer

    return asyncio.run(go()), dict(state.last_entities)


def test_result_dialog_surname_from_llm(monkeypatch) -> None:
    _switch(monkeypatch, "names_result_surname")
    _, entities = _result_dialog(monkeypatch, "Сидорова", {"surname": "Сидорова"})
    assert entities.get("surname") == "Сидорова"


def test_result_dialog_without_llm_asks_again(monkeypatch) -> None:
    """LLM недоступна (сеть в тестах закрыта): одно слово фамилией не становится — бот снова просит реквизиты."""

    _switch(monkeypatch, "names_result_surname")
    answer, entities = _result_dialog(monkeypatch, "Сидорова", None)
    assert "surname" not in entities
    assert "фамилия, год рождения, код анализа, номер анализа" in answer


# --- ФИО пациента: names_patient_fio / names_patient_rule / names_patient_fill --------------------------------------

# Служебные слова в разных формах: форма ФИО (2–3 слова) их принимает; списки знали не все формы.
_NOT_NAMES = [
    "Общий анализ крови",
    "Анализом крови",
    "Результатов анализов",
    "Расписание гинеколога",
    "Врачу терапевту",
    "Записи нет",
    "Стоимость приёма",
    "Хотела уточнить",
]


def _fio_on_path(switch: str, text: str) -> bool:
    from messengers_router import flow_policy, policies

    if switch == "names_patient_fio":
        return flow_policy._looks_like_patient_fio(text)
    if switch == "names_patient_rule":
        return classifier._looks_like_patient_name_only(text)
    out: dict = {}
    policies._fill_patient_entities(text, ["patient_name"], out)
    return bool(out.get("patient_name"))


_PATIENT_SWITCHES = ["names_patient_fio", "names_patient_rule", "names_patient_fill"]


@pytest.mark.parametrize("switch", _PATIENT_SWITCHES)
@pytest.mark.parametrize("text", ["Сидорова Анна", "сидорова анна петровна", "СИДОРОВА АННА", "Сидорова Алёна"])
def test_patient_fio_named_by_llm_any_case(monkeypatch, switch: str, text: str) -> None:
    _switch(monkeypatch, switch)
    turn_names.note_llm(text, {"patient_name": text.title().replace("ё", "е")})
    assert _fio_on_path(switch, text)


@pytest.mark.parametrize("switch", _PATIENT_SWITCHES)
@pytest.mark.parametrize("text", _NOT_NAMES)
def test_service_words_in_any_form_are_not_a_patient_name(monkeypatch, switch: str, text: str) -> None:
    """LLM хода имени не назвала — форма ФИО не делает реплику именем, какой бы формой ни было служебное слово."""

    _switch(monkeypatch, switch)
    turn_names.note_llm(text, {"service_name": text})
    assert not _fio_on_path(switch, text)


@pytest.mark.parametrize("switch", _PATIENT_SWITCHES)
def test_no_llm_this_turn_no_patient_name(monkeypatch, switch: str) -> None:
    _switch(monkeypatch, switch)
    assert not _fio_on_path(switch, "Сидорова Анна")


@pytest.mark.parametrize("switch", _PATIENT_SWITCHES)
def test_llm_name_not_in_reply_is_not_accepted(monkeypatch, switch: str) -> None:
    _switch(monkeypatch, switch)
    turn_names.note_llm("Сидорова Анна", {"patient_name": "Петрова Анна"})
    assert not _fio_on_path(switch, "Сидорова Анна")



def test_patient_fio_unbookable_target_needs_llm(monkeypatch) -> None:
    """Вид недоступной цели «ФИО пациента» (вопрос о цели, не «такого врача нет») — по LLM хода, не по форме."""

    from messengers_router import planner

    _switch(monkeypatch, "names_patient_fio")
    assert planner._unbookable_target_kind("Сидорова Анна Петровна", set()) != "patient_name"
    turn_names.note_llm("Сидорова Анна Петровна", {"patient_name": "Сидорова Анна Петровна"})
    assert planner._unbookable_target_kind("Сидорова Анна Петровна", set()) == "patient_name"


def test_patient_rule_waits_for_llm(monkeypatch) -> None:
    """Правило шага ФИО зовут до LLM: под `on` оно молчит, и ход решает LLM-классификатор."""

    _switch(monkeypatch, "names_patient_rule")
    pending = {"_pending": {"label": "APPOINTMENT", "missing": ["patient_name"]}}
    decision = asyncio.run(classifier.deterministic_rule_decision("Сидорова Анна", pending))
    assert decision is None or "rule_appointment_patient_name" not in decision.flags


def test_patient_fill_after_prefix_takes_llm_name(monkeypatch) -> None:
    from messengers_router import policies

    _switch(monkeypatch, "names_patient_fill")
    reply = "Меня зовут Сидорова Анна"
    turn_names.note_llm(reply, {"patient_name": "Сидорова Анна"})
    out: dict = {}
    policies._fill_patient_entities(reply, ["patient_name"], out)
    assert out["patient_name"] == "Сидорова Анна"


@pytest.mark.parametrize("text", ["Общий анализ крови", "Анализ крови"])
def test_patient_fill_does_not_take_service(monkeypatch, text: str) -> None:
    """Известный дефект пути off (характеристика): услуга в графе ФИО. Под `on` её туда кладёт только LLM, а она назвала услугу."""

    from messengers_router import policies

    _switch(monkeypatch, "names_patient_fill")
    turn_names.note_llm(text, {"service_name": text})
    out: dict = {}
    policies._fill_patient_entities(text, ["patient_name"], out)
    assert "patient_name" not in out


# --- путь `on` не читает список (AST) --------------------------------------------------------------------------------

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"
# Выключатель → (файл, функция-развилка, что читает только путь off): список или единственный его читатель.
_GUARDS = {
    "names_patient_fio": ("flow_policy.py", "_looks_like_patient_fio", "_PATIENT_FIO_STOPWORDS"),
    "names_patient_rule": ("classifier.py", "_looks_like_patient_name_only", "_PATIENT_NAME_STOPWORDS"),
    "names_patient_fill": ("policies.py", "_is_valid_patient_name_candidate", "_QF_PATIENT_NAME_STOPWORDS"),
    "names_result_surname": ("policies.py", "_fill_test_result_entities", "_QF_RESULT_SURNAME_STOPWORDS"),
    "names_doctor_followup": ("classifier.py", "_doctor_followup_name_in_context", "_DOCTOR_FOLLOWUP_FILLERS"),
    "names_doctor_noise": ("entity_grounder.py", "_sanitize_raw_doctor_name", "_DOCTOR_NOISE_TOKENS"),
    "names_doctor_catalog": ("services/doctors.py", "match_catalog_doctor", "_doctor_catalog_query_candidates"),
}


def _is_on(node: ast.AST, switch: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "on"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "turn_names"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == switch
    )


def _is_off(node: ast.AST, switch: str) -> bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and _is_on(node.operand, switch)


def _returns_on(stmt: ast.stmt, switch: str) -> bool:
    """`if turn_names.on(switch): … return …` — дальше по этому списку операторов идёт только путь off."""

    return isinstance(stmt, ast.If) and _is_on(stmt.test, switch) and bool(stmt.body) and isinstance(stmt.body[-1], ast.Return)


def _off_only(func: ast.AST, target: ast.AST, switch: str) -> bool:
    parents = {child: parent for parent in ast.walk(func) for child in ast.iter_child_nodes(parent)}
    node = target
    while node is not func:
        parent = parents[node]
        if isinstance(parent, ast.If) and node in parent.body and _is_off(parent.test, switch):
            return True
        if isinstance(parent, ast.If) and node in parent.orelse and _is_on(parent.test, switch):
            return True
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if isinstance(block, list) and node in block and any(_returns_on(s, switch) for s in block[: block.index(node)]):
                return True
        node = parent
    return False


def _function(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


@pytest.mark.parametrize("switch", sorted(_GUARDS))
def test_on_path_does_not_read_the_list(switch: str) -> None:
    file, function, symbol = _GUARDS[switch]
    func = _function(_CORE / file, function)
    reads = [n for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == symbol]
    assert reads, f"{function} не читает {symbol} — развилка устарела"
    assert all(_off_only(func, n, switch) for n in reads), f"{function}: {symbol} читает путь on"
    assert any(_is_on(n, switch) for n in ast.walk(func)), f"{function} не смотрит на выключатель {switch}"


def _readers(symbol: str) -> set[str]:
    """Функции ядра, которые читают имя (`ast.Name` в теле), — кто читает список, по всему `messengers_router`."""

    found: set[str] = set()
    for path in sorted(_CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(n, ast.Name) and n.id == symbol for n in ast.walk(func)
            ):
                found.add(f"{path.relative_to(_CORE)}::{func.name}")
    return found


@pytest.mark.parametrize("switch", sorted(_GUARDS))
def test_list_has_no_reader_besides_the_guarded_one(switch: str) -> None:
    """Новый читатель списка обошёл бы выключатель — опись (журнал) и развилка должны его знать."""

    file, function, symbol = _GUARDS[switch]
    if symbol == "_doctor_catalog_query_candidates":
        assert _readers("_CATALOG_DOCTOR_STOPWORDS") == {"services/_doctors_helpers.py::_doctor_catalog_query_candidates"}
    assert _readers(symbol) == {f"{file}::{function}"}


def test_names_module_reads_no_list() -> None:
    tree = ast.parse((_CORE / "turn_names.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    listed = {sym for _, _, sym in _GUARDS.values()} | {
        "_CATALOG_DOCTOR_STOPWORDS", "STOP_WORDS", "_EXTRA_STOPWORDS",
    }
    assert names.isdisjoint(listed)


def test_every_switch_is_a_config_field() -> None:
    """Выключатель читается из `config.ini` (`[MESSENGER_ROUTER] names_*`), а не только атрибутом на прогон."""

    from agent_logic_2 import config

    assert all(getattr(config, "MR_" + name.upper()) == "off" for name in _GUARDS)
