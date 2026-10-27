"""Контекст сессии = показанное — L-14, шаг 3. Единственный путь с 17.10: выключатель `[MESSENGER_ROUTER] context_shown`
снят (STEP-2026-10-17-ONE-PATH-CONTEXT-SHOWN; на проде `on` с 15.10).

Было: услугу в состояние писало только заземление правил по буквам. Строку, которую показал выбор LLM («ггтп» →
«ГГТ гамма-глютамилтрансфераза»), состояние не получало — следующее «сколько стоит?» не знало, о чём речь; «а
цена?» после ответа с услугой в контексте искала в прайсе букву «а» («По услуге «а»…» — цены аллергенов и шерсти);
сводка диалога называла услугу, которую состояние уже сменило или сняло.

Стало: ответ с ценами пишет `_shown_services` — строки, которые увидел пациент, — и `service_name`, если
показана одна услуга; срок из МИС — только `_shown_services` (ответ без цен: по решению владельца 13.10 услуги в
контексте нет); корзина снимает `_shown_services`; реплика, в понимании которой услуга не названа, не перебивает
контекст фразой вырезателя; сводка обновляется, как только называет не ту услугу.

Инвариант класса: контекст — то, что пациент видел; короткий вопрос не показывает цен строк, которых пациент не видел,
и не печатает слова вопроса названием услуги; ключ `context_shown` в конфиге ни на что не влияет.
"""

from __future__ import annotations

import json

import pytest

from messengers_router.services._common import _normalise_catalog_text
from tests.spans_oracle import SpansOracle
from tests.test_l14_two_turn_consumers import _FOUND_GGT, _QUOTED, _Scripted, _play, _priced, _shown
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

SHOWN = ("deploy", "next")
_GGT = "ГГТ гамма-глютамилтрансфераза"


def test_one_path_without_a_switch():
    """Выключателя нет: ни ключа в настройках, ни режима у модуля (решение владельца 16.10, один путь)."""

    from agent_logic_2 import config
    from messengers_router import shown_context

    assert not hasattr(config, "MR_CONTEXT_SHOWN") and not hasattr(config.settings.messenger_router, "context_shown")
    assert not hasattr(shown_context, "mode") and not hasattr(shown_context, "MODES")


def _row(name: str, price: int = 100) -> dict:
    return {"serviceName": name, "price": price}


@pytest.mark.parametrize(
    ("payload", "names"),
    [
        ({"clarify_text": "Уточните название", "prices": [_row("A")]}, []),  # уточнение цен не показывает
        ({"prices": [_row("A"), _row("A", 200), _row("B")]}, ["A", "B"]),  # повторы названия — одно
        ({"prices": [_row(str(i)) for i in range(8)]}, ["0", "1", "2", "3", "4"]),  # рендер печатает пять
        ({"service_kind": "family_query", "family_variants": [_row(str(i)) for i in range(12)], "visible_limit": 3}, ["0", "1", "2"]),
        ({"service_kind": "family_query", "family_variants": [_row(str(i)) for i in range(12)], "visible_limit": 3, "showing_all": True}, [str(i) for i in range(12)]),
        ({"prices": []}, []),
    ],
    ids=["clarify", "dedup", "five", "family_visible", "family_all", "empty"],
)
def test_shown_names_are_what_the_renderer_prints(payload, names):
    from messengers_router import shown_context

    assert shown_context.shown_price_names(payload) == names


def test_stale_summary_is_detected_only_when_it_names_another_service():
    from messengers_router.context_summary import _names_stale_context

    summary = "last_turns:\nuser: сколько стоит ферритин\n\nstate:\n_last_label=PRICE, service_name=Ферритин"
    assert not _names_stale_context(summary, {"service_name": "Ферритин"})
    assert _names_stale_context(summary, {})  # услугу сняли
    assert _names_stale_context(summary, {"service_name": "Гастрин"})  # услугу сменили
    assert not _names_stale_context("last_turns:\nuser: привет", {"service_name": "Ферритин"})  # сводка услуги не называет


@pytest.mark.parametrize("config", SHOWN)
def test_llm_picked_row_is_the_context_and_the_next_question_shows_it(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода через эндпоинт: строку выбрала LLM по термину — она в контексте; «сколько стоит?» — её цена, ничего
    другого."""

    first, second = _play(["сколько стоит ггтп", "сколько стоит?"], "shown-ggt", mini_prod_world, monkeypatch, config, llm=_FOUND_GGT())
    assert _priced(first.text) == [_normalise_catalog_text(_GGT)]
    assert (first.service_name, first.shown) == (_GGT, [_GGT])
    assert set(_priced(second.text)) <= _shown(first.text) and _priced(second.text), second.text


def test_old_off_value_in_the_config_changes_nothing(mini_prod_world, monkeypatch):  # noqa: F811
    """Прежнее значение `context_shown = off` в `config.ini` (прод до деплоя 6 держит ключ) не читается: два хода —
    как без него."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_CONTEXT_SHOWN", "off", raising=False)
    first, second = _play(["сколько стоит ггтп", "сколько стоит?"], "shown-ggt-old-off", mini_prod_world, monkeypatch, "deploy", llm=_FOUND_GGT())
    assert (first.service_name, first.shown) == (_GGT, [_GGT])
    assert set(_priced(second.text)) <= _shown(first.text) and _priced(second.text), second.text


@pytest.mark.parametrize("config", SHOWN)
@pytest.mark.parametrize("first", ["сколько стоит ферритин", "сколько стоит оак", "сколько стоит ггтп"])
def test_a_price_after_an_answer_does_not_search_the_letter(first, config, mini_prod_world, monkeypatch):  # noqa: F811
    """«а цена?» без названия услуги — о показанном: ни «По услуге «а»», ни цен строк, которых пациент не видел."""

    llm = _FOUND_GGT() if "ггтп" in first else None
    played = _play([first, "а цена?"], f"shown-a-{first}", mini_prod_world, monkeypatch, config, llm=llm)
    seen = _shown(played[0].text)
    assert set(_priced(played[1].text)) <= seen, played[1].text
    assert not [q for q in _QUOTED.findall(played[1].text) if _normalise_catalog_text(q) in {"а", "а цена"}], played[1].text


@pytest.mark.parametrize("config", SHOWN)
def test_readiness_names_one_service_and_holds_it(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Срок из МИС назвал одну услугу — она в контексте (правило владельца 14.10, уточнение решения 13.10: в
    контексте — услуга, которую ответ назвал пациенту одной строкой; догадка — никогда). Два хода: «сколько стоит?» —
    её цена, ничего другого."""

    first, second = _play(["сколько готовится ферритин", "сколько стоит?"], "shown-readiness", mini_prod_world, monkeypatch, config)
    assert first.text == "Срок готовности «Ферритин» — 1–2 дня."
    assert (first.service_name, first.shown) == ("Ферритин", ["Ферритин"])
    assert _priced(second.text) == ["ферритин"], second.text


@pytest.mark.parametrize("config", SHOWN)
def test_policy_answer_without_prices_still_holds_nothing(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Граница правила 14.10: ответ политики без цен («не нашёл») ничего не назвал — услуги в контексте нет."""

    first, second = _play(["сколько стоит гастроскопия", "сколько стоит?"], "shown-policy", mini_prod_world, monkeypatch, config, llm=_Scripted(terms=("ФГДС",)))
    if not _priced(first.text):
        assert first.service_name is None and not first.shown
        assert not _priced(second.text), second.text


@pytest.mark.parametrize(("config", "first"), [("next", "сколько стоит глюкоза"), ("deploy", "сколько стоит оак")])
@pytest.mark.parametrize("again", ["сколько стоит?", "а цена?"])
def test_repeat_shows_exactly_what_was_shown(again, config, first, mini_prod_world, monkeypatch):  # noqa: F811
    """Находка 4: повтор вопроса без названия услуги — ровно показанные строки (одна строка «Глюкоза» — не семья
    «Глюкозы в моче» и «в синовиальной жидкости»; два варианта ОАК — эти два, не «ещё 6»), без слов вопроса."""

    played = _play([first, again], f"shown-repeat-{first}-{again}", mini_prod_world, monkeypatch, config)
    assert _priced(played[1].text) == _priced(played[0].text), played[1].text
    assert not [q for q in _QUOTED.findall(played[1].text) if _normalise_catalog_text(q) in {"а", "сколько стоит", "а цена"}]


# --------------------------------------------------------------------------- подготовка (находка 2)

_MEMO_TEXT = "Сдавать натощак, после 8–12 часов голода; за сутки исключить алкоголь."
_PREPARE_MARK = "как подготовиться к анализу или процедуре"
_CONTEXT_MARK = "Перед этим вопросом бот показал пациенту услугу"


class _PrepareLLM:
    """LLM выбора памятки: выбирает памятку «Ферритин», только если промпт называет показанную услугу и в вопросе
    нет своего предмета (как решила бы модель по смыслу); прочие места — недоступна."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        from ollama import GenerateResponse

        if _PREPARE_MARK not in prompt:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.prompts.append(prompt)
        lines = [line for line in prompt.splitlines() if line[:1].isdigit() and ". " in line]
        own = "ттг" in prompt.split("Вопрос пациента:", 1)[1].split("\n", 1)[0].lower()
        match = None if own or _CONTEXT_MARK not in prompt else next((int(line.split(".", 1)[0]) for line in lines if "Ферритин" in line), None)
        return GenerateResponse(model="m", response=json.dumps({"match": match}), done=True)


def _with_memo(monkeypatch) -> None:
    from messengers_router.services import prepare as prep
    from messengers_router.services._prepare_select_llm import SOURCE_KB, PrepareMemo

    memo = PrepareMemo(source=SOURCE_KB, title="Ферритин", text=_MEMO_TEXT)
    monkeypatch.setattr(prep, "kb_patient_memos", lambda: (memo,))


def _prod_parse(monkeypatch) -> None:
    """Разбор хода подготовки — разметка LLM, как на проде: `prepare_by_parse` и `turn_spans_llm` включены вместе с деплоя 17
    (24.10). Пока умолчание `turn_spans_llm` — off, гейт иначе строил бы разбор старым генератором окон (он режет без LLM), а
    этого сочетания на проде не было."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_SPANS_LLM", "on", raising=False)


@pytest.mark.parametrize("config", SHOWN)
@pytest.mark.parametrize("first", ["сколько стоит ферритин", "сколько готовится ферритин"])
@pytest.mark.parametrize("ask", ["а как подготовиться?", "подскажите пожалуйста как подготовиться", "а натощак?"])
def test_prepare_without_own_subject_is_about_the_shown_service(ask, first, config, mini_prod_world, monkeypatch):  # noqa: F811
    """LLM видит показанную услугу и по смыслу выбирает её памятку; в ответе нет слов вопроса."""

    _with_memo(monkeypatch)
    _prod_parse(monkeypatch)
    llm = _PrepareLLM()
    oracle = SpansOracle(("ферритин",), inner=llm)  # первый ход называет ферритин, вопрос о подготовке — ничего
    played = _play([first, ask], f"shown-prep-{first}-{ask}", mini_prod_world, monkeypatch, config, llm=oracle)
    assert _MEMO_TEXT[:20] in played[1].text, played[1].text
    assert llm.prompts and _CONTEXT_MARK in llm.prompts[-1] and "«Ферритин»" in llm.prompts[-1]


@pytest.mark.parametrize("config", SHOWN)
@pytest.mark.parametrize(
    ("ask", "expected"),
    [
        # LLM нет — разбор хода не удался: без LLM предмет не угадываем ни по буквам, ни по показанному (решение владельца
        # 22.10 (а); до пути прода — «Памятки по подготовке к «Ферритин» у меня нет» / «к «Ферритин» или к другому»)
        ("а как подготовиться?", "Уточните, пожалуйста, к какому анализу или процедуре нужна подготовка"),
        ("а натощак?", "Уточните, пожалуйста, к какому анализу или процедуре нужна подготовка"),
        ("а как подготовиться к ттг?", "Уточните, пожалуйста, к какому анализу или процедуре нужна подготовка"),
    ],
)
def test_prepare_without_llm_never_prints_question_words(ask, expected, config, mini_prod_world, monkeypatch):  # noqa: F811
    _prod_parse(monkeypatch)
    played = _play(["сколько стоит ферритин", ask], f"shown-prep-nollm-{ask}", mini_prod_world, monkeypatch, config)
    assert expected in played[1].text, played[1].text
    assert not [q for q in _QUOTED.findall(played[1].text) if _normalise_catalog_text(q) in {_normalise_catalog_text(ask.strip("?")), "а натощак"}]


@pytest.mark.parametrize("config", SHOWN)
def test_prepare_after_several_shown_asks_which(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Показаны два варианта ОАК — «а как подготовиться?» спрашивает, к какому из них (принцип 5)."""

    _prod_parse(monkeypatch)
    oracle = SpansOracle(("оак",))  # первый ход называет ОАК, вопрос о подготовке — ничего; выбор строки — недоступна
    played = _play(["сколько стоит оак", "а как подготовиться?"], "shown-prep-family", mini_prod_world, monkeypatch, config, llm=oracle)
    assert played[1].text.startswith("К какому из анализов нужна подготовка: «Общий анализ крови (Le, Er, Hb, СОЭ)»"), played[1].text


def test_prepare_prompt_without_context_is_unchanged():
    """Без показанной услуги промпт выбора памятки — байт в байт прежний."""

    from messengers_router.prompt_registry import load_prompt_text
    from messengers_router.services._prepare_select_llm import PrepareMemo, _build_prompt, _excerpt

    memos = [PrepareMemo(source="x", title="Ферритин", text=_MEMO_TEXT)]
    template = load_prompt_text("prepare_document_select")
    head, _, rest = template.partition("<<CONTEXT_START>>")
    assert "<<CONTEXT_END>>" in rest
    before = (
        (head + rest.partition("<<CONTEXT_END>>")[2])
        .replace("<<QUESTION>>", "как подготовиться к ферритину")
        .replace("<<DOCUMENTS>>", f"1. Ферритин — {_excerpt(_MEMO_TEXT)}")
    ).strip()
    assert _build_prompt("как подготовиться к ферритину", memos) == before
    assert _CONTEXT_MARK in _build_prompt("а натощак?", memos, "Ферритин")
